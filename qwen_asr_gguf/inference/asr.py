# coding=utf-8
import os
import time
import re
import codecs
import dataclasses
import numpy as np
import multiprocessing as mp
from pathlib import Path
from collections import deque
from typing import Optional, List

from .schema import MsgType, StreamingMessage, DecodeResult, ASREngineConfig, TranscribeResult, ForcedAlignItem, ForcedAlignResult
from .utils import normalize_language_name, validate_language
from .encoder import QwenAudioEncoder
from . import llama

@dataclasses.dataclass
class ASRS_Segment:
    """管理分片记忆及其物理时间坐标"""
    idx: int
    audio_start: float
    audio_end: float
    text: str = ""
    items: List[ForcedAlignItem] = None   

def _tail_char_loop(s: str) -> bool:
    n = len(s)
    if n < 12:
        return False
    tail = s[-400:] if n > 400 else s
    m = len(tail)
    for plen in range(1, min(80, m // 2) + 1):
        if plen <= 4:
            need = 6
        elif plen <= 10:
            need = 4
        elif plen <= 30:
            need = 3
        else:
            need = 2
        if m < plen * need:
            continue
        pat = tail[-plen:]
        ok = True
        for r in range(1, need):
            if tail[m - (r + 1) * plen: m - r * plen] != pat:
                ok = False
                break
        if ok:
            return True
    return False


def _has_loop_text(text: str) -> bool:
    if not text:
        return False
    if _tail_char_loop(text):
        return True
    nospace = re.sub(r'\s', '', text)
    if nospace is not text and _tail_char_loop(nospace):
        return True
    low = text.lower()
    if _tail_char_loop(low):
        return True
    lines = [l.strip() for l in re.split(r'[\n。？！!?；;]+', text) if l.strip()]
    if len(lines) >= 4:
        if lines[-1] == lines[-3] and lines[-2] == lines[-4]:
            return True
        if lines[-1] == lines[-2] == lines[-3]:
            return True
    words = re.findall(r"[a-zA-Z']+", low)
    if len(words) >= 8:
        tail = words[-24:] if len(words) > 24 else words
        tlen = len(tail)
        for wn in range(1, 7):
            need = 6 if wn == 1 else 3
            if tlen < wn * need:
                continue
            pat = tail[-wn:]
            ok = True
            for r in range(1, need):
                if tail[tlen - (r + 1) * wn: tlen - r * wn] != pat:
                    ok = False
                    break
            if ok:
                return True
    return False


def _has_token_loop(toks) -> bool:
    n = len(toks)
    if n < 12:
        return False
    tail = toks[-60:] if n > 60 else toks
    m = len(tail)
    if m >= 20 and len(set(tail[-20:])) == 1:
        return True
    for p in range(1, min(24, m // 2) + 1):
        need = 8 if p == 1 else 4 if p == 2 else 3
        if m < p * need:
            continue
        pat = tail[m - p:]
        ok = True
        for r in range(1, need):
            if tail[m - (r + 1) * p: m - r * p] != pat:
                ok = False
                break
        if ok:
            return True
    return False


def _find_repeat_start(t: str):
    n = len(t)
    end = min(100, n // 2)
    for plen in range(end, 0, -1):
        need = 2 if plen >= 10 else 3
        if plen == 1:
            need = 6
        if n < plen * need:
            continue
        pat = t[n - plen:]
        cnt = 1
        pos = n - plen
        while pos >= plen and t[pos - plen:pos] == pat:
            cnt += 1
            pos -= plen
        if cnt >= need and (plen >= 2 or cnt >= 6):
            return pos
    return -1


class QwenASREngine:
    """Qwen3-ASR 流式转录引擎 (GGUF 后端) - 统一辅助进程架构"""
    def __init__(self, config: ASREngineConfig):
        self.config = config
        self.verbose = config.verbose
        if self.verbose: print(f"--- [QwenASR] 初始化引擎 (Provider: {config.onnx_provider}) ---")
        
        # 路径解析
        llm_gguf = os.path.join(config.model_dir, config.llm_fn)
        frontend_path = os.path.join(config.model_dir, config.encoder_frontend_fn)
        backend_path = os.path.join(config.model_dir, config.encoder_backend_fn)

        # 1. 初始化 Encoder
        self.encoder = QwenAudioEncoder(
            frontend_path=frontend_path,
            backend_path=backend_path,
            onnx_provider=config.onnx_provider,
            dml_pad_to=config.dml_pad_to,
            verbose=self.verbose
        )

        # 2. 初始化 Aligner (可选)
        self.aligner = None
        if config.enable_aligner and config.align_config:
            from .aligner import QwenForcedAligner
            self.aligner = QwenForcedAligner(config.align_config)
        
        # 3. 加载识别 LLM
        self.model = llama.LlamaModel(llm_gguf, use_gpu=config.llm_use_gpu)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)
        self.ctx = llama.LlamaContext(self.model, n_ctx=config.n_ctx, n_batch=config.n_ctx * 2, embeddings=False)

        # 缓存 Token ID
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    def shutdown(self):
        if self.verbose: print("--- [QwenASR] 引擎已关闭 ---")

    def _build_prompt_embd(self, audio_embd: np.ndarray, prefix_text: str, context: Optional[str], language: Optional[str]):
        """构造用于 LLM 输入的 Embedding 序列 (区块化打包模式)"""
        def tk(t): return self.model.tokenize(t)

        # 1. 区块 A: 音频之前的所有内容 (System + User Header)
        prefix_str = f"system\n{context or 'You are a helpful assistant.'}"
        prefix_tokens = [self.ID_IM_START] + tk(prefix_str) + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk("user\n") + [self.ID_AUDIO_START]
        
        # 2. 区块 B: 音频之后的所有内容 (Instruction + Assistant Header + History)
        suffix_head = f"assistant\n"
        if language: suffix_head += f"language {language}"
        
        suffix_tokens = [self.ID_AUDIO_END] + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk(suffix_head) + [self.ID_ASR_TEXT] + tk(prefix_text)

        # 3. 截断音频防止超过 n_ctx
        n_pre, n_aud, n_suf = len(prefix_tokens), audio_embd.shape[0], len(suffix_tokens)
        max_audio = self.config.n_ctx - n_pre - n_suf - 64  # 预留64token安全空间
        if max_audio < 100:
            max_audio = 100  # 至少保留100个音频token
        if n_aud > max_audio:
            audio_embd = audio_embd[-max_audio:]
            n_aud = max_audio
            print(f"\n[警告] 音频过长，已截断至 {n_aud} tokens")
        
        # 4. 统计并拼接
        total_embd = np.zeros((n_pre + n_aud + n_suf, self.model.n_embd), dtype=np.float32)
        
        total_embd[:n_pre] = self.embedding_table[prefix_tokens]
        total_embd[n_pre : n_pre + n_aud] = audio_embd
        total_embd[n_pre + n_aud:] = self.embedding_table[suffix_tokens]
        
        return total_embd

    def _decode(
        self, 
        full_embd: np.ndarray,
        prefix_text: str, 
        rollback_num: int,
        is_last_chunk: bool = False, 
        temperature: float = 0.4, 
        streaming: bool = True, 
    ) -> DecodeResult:
        """底层方法：执行单次 LLM 生成循环（物理推理）"""
        result = DecodeResult()
        
        total_len = full_embd.shape[0]
        
        if total_len > self.config.n_ctx:
            full_embd = full_embd[-self.config.n_ctx:]
            total_len = self.config.n_ctx
        
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
        batch = llama.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        # 1. Prefill
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start
        
        # 2. Generation Loop（使用新采样器和随机种子）
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue = deque()
        stable_tokens = []
        stable_text_acc = ""
        text_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        
        # 每次解码使用新的随机种子
        seed = int(np.random.randint(0, 2**31 - 1))
        sampler = llama.LlamaSampler(temperature=temperature, seed=seed)
        last_sampled_token = sampler.sample(self.ctx.ptr)
        for _ in range(512): # Max new tokens per chunk
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break
            
            if self.ctx.decode_token(last_sampled_token) != 0:
                    break
            
            display_queue.append(last_sampled_token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = text_decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    if streaming: print(re.sub(r'([，。？！：,\.])', r'\1\n', piece), end='', flush=True)
                    stable_text_acc += piece
            
            # 熔断检查：token 周期重复（含 display_queue 前视，加速 5 token）
            if len(stable_tokens) + len(display_queue) >= 12:
                _all_toks = stable_tokens + list(display_queue) if display_queue else stable_tokens
                if _has_token_loop(_all_toks):
                    result.is_aborted = True
                    setattr(result, 'abort_reason', 'loop')
                    break

            # 熔断检查：文本周期重复（中英 ABAB / i wish / i will，3 次即熔断）
            if n_gen_tokens % 2 == 0 and stable_text_acc and len(stable_text_acc) >= 12:
                try:
                    _pend_b = b''.join(self.model.token_to_bytes(t) for t in display_queue) if display_queue else b''
                    _pend_s = _pend_b.decode('utf-8', errors='ignore') if _pend_b else ''
                except Exception:
                    _pend_s = ''
                if _has_loop_text(stable_text_acc + _pend_s):
                    result.is_aborted = True
                    setattr(result, 'abort_reason', 'loop')
                    break

            # 熔断检查：检测超长无标点句子（按标点/换行分割后最长段超过200字符）
            if stable_text_acc and len(stable_text_acc) > 250:
                recent_text = stable_text_acc[-500:] if len(stable_text_acc) > 500 else stable_text_acc
                # 按句子结束标点和换行分割，取最长段落检查
                segments = re.split(r'[。？！.!?\n]', recent_text)
                longest_seg = max((s for s in segments if s), key=len, default='')
                if len(longest_seg) > 200:
                    result.is_aborted = True
                    setattr(result, 'abort_reason', 'long')
                    break
            
            last_sampled_token = sampler.sample(self.ctx.ptr)
            n_gen_tokens += 1
            
        gen_time = time.time() - t_gen_start
        del sampler  # 释放采样器资源
        del batch
            
        if is_last_chunk and not result.is_aborted:
            while display_queue:
                t = display_queue.popleft()
                stable_tokens.append(t)
                piece = text_decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    if streaming: print(re.sub(r'([，。？！：,\.])', r'\1\n', piece), end="", flush=True)
                    stable_text_acc += piece
            final_p = text_decoder.decode(b"", final=True)
            if final_p: 
                if streaming: print(final_p, end='', flush=True)
                stable_text_acc += final_p
        
        if result.is_aborted and display_queue:
            try:
                _tail_b = b''.join(self.model.token_to_bytes(t) for t in display_queue)
                stable_text_acc += _tail_b.decode('utf-8', errors='ignore')
                stable_tokens.extend(list(display_queue))
            except Exception:
                pass
        # 填充结果（内核输出标准化）
        result.text = stable_text_acc
        result.stable_tokens = stable_tokens
        result.t_prefill = prefill_time
        result.t_generate = gen_time
        result.n_prefill = total_len
        result.n_generate = n_gen_tokens
        result.n_generate = n_gen_tokens
        return result

    def _safe_decode(
        self,
        full_embd: np.ndarray,
        prefix_text: str,
        rollback_num: int,
        is_last_chunk: bool,
        temperature: float,
        streaming: bool = True,
    ) -> DecodeResult:
        """带熔断加温重试的高层推理封装（loop 重试，long 直接截断不重试，省 GPU）"""
        original_temp = temperature
        res = None
        for i in range(3):
            res = self._decode(full_embd, prefix_text, rollback_num, is_last_chunk, temperature, streaming=streaming)
            if not res.is_aborted:
                break
            reason = getattr(res, 'abort_reason', 'loop')
            if reason == 'long':
                res.text = self._truncate_long(res.text)
                res.is_aborted = False
                print(f"[!] 超长无标点内容，已截断（{len(res.text)}字符）")
                return res
            if res.text:
                cleaned = self._remove_tail_repeat(res.text)
                if cleaned is not None and (not res.text or len(cleaned) >= len(res.text) * 0.3):
                    res.text = cleaned
                    res.is_aborted = False
                    print(f"[!] 末尾重复已清理，保留前置内容（{len(res.text)}字符）")
                    return res
            if i >= 1 and res.text:
                cleaned = self._remove_tail_repeat(res.text)
                if cleaned is not None:
                    res.text = cleaned
                else:
                    res.text = self._truncate_long(res.text) if len(res.text) > 200 else res.text
                res.is_aborted = False
                return res
            temperature = min(original_temp + 0.4 * (i + 1), 1.5)
            print(f"\n\n[!] 触发重试 (尝试 {i+1}/3, Temp -> {temperature:.1f})\n")

        if res is not None and res.is_aborted and res.text:
            cleaned = self._remove_tail_repeat(res.text)
            if cleaned is not None:
                res.text = cleaned
            else:
                res.text = self._truncate_long(res.text) if len(res.text) > 200 else res.text
            res.is_aborted = False
        return res

    def _truncate_long(self, text: str) -> str:
        if not text or len(text) <= 200:
            return text
        if re.search(r'[。？！.!?]', text):
            return text[:250]
        break_pos = -1
        for sep in ('\n', '\r', ' '):
            pos = text.rfind(sep, 100, min(len(text), 250))
            if pos > break_pos:
                break_pos = pos
        return text[:break_pos] if break_pos > 100 else text[:200]

    def _remove_tail_repeat(self, text):
        """检测并移除末尾重复（字符/行/单词级，2-3 次即清理），无重复返回 None"""
        if not text or len(text) < 12:
            return None
        t = text.lower()
        pos = _find_repeat_start(t)
        if pos > 0:
            return text[:pos]
        if pos == 0:
            return ""
        lines = [l.strip() for l in re.split(r'[\n。？！!?；;]+', text) if l.strip()]
        if len(lines) >= 4 and lines[-1] == lines[-3] and lines[-2] == lines[-4]:
            keep = lines[:-4]
            head_len = len(text) - len(''.join(lines[-4:]))
            return text[:max(head_len, 0)].rstrip() if keep else ""
        if len(lines) >= 3 and lines[-1] == lines[-2] == lines[-3]:
            idx = text.rfind(lines[-1], 0, len(text) - len(lines[-1]))
            return text[:idx].rstrip() if idx > 0 else ""
        words = re.findall(r"[a-zA-Z']+", t)
        if len(words) >= 9:
            for wn in range(1, 7):
                need = 6 if wn == 1 else 3
                if len(words) < wn * need:
                    continue
                pat = words[-wn:]
                if all(words[len(words) - (r + 1) * wn: len(words) - r * wn] == pat for r in range(1, need)):
                    cut = len(words) - wn * need
                    head_words = words[:cut]
                    if not head_words:
                        return ""
                    tail_str = ' '.join(head_words)
                    if len(text) > len(tail_str):
                        return text[:text.lower().rfind(pat[0].lower())].rstrip() if cut > 0 else ""
                    return tail_str
        return None

    def _print_stats(self, stats: dict, audio_duration: float, t_total: float):
        """打印转录过程的性能统计指标"""
        rtf = t_total / audio_duration if audio_duration > 0 else 0
        pre_speed = stats["prefill_tokens"] / stats["prefill_time"] if stats["prefill_time"] > 0 else 0
        gen_speed = stats["decode_tokens"] / stats["decode_time"] if stats["decode_time"] > 0 else 0
        
        print(f"\n\n📊 性能统计:")
        print(f"  🔹 RTF (实时率) : {rtf:.3f} (越小越快)")
        print(f"  🔹 音频时长    : {audio_duration:.2f} 秒")
        print(f"  🔹 总处理耗时  : {t_total:.2f} 秒")
        if stats.get("align_time"):
            print(f"  🔹 对齐耗时    : {stats['align_time']:.3f} 秒")
        print(f"  🔹 编码耗时    : {stats['encode_time']:.3f} 秒")
        print(f"  🔹 LLM 预填充  : {stats['prefill_time']:.3f} 秒 ({stats['prefill_tokens']} tokens, {pre_speed:.1f} tokens/s)")
        print(f"  🔹 LLM 生成    : {stats['decode_time']:.3f} 秒 ({stats['decode_tokens']} tokens, {gen_speed:.1f} tokens/s)")

    def transcribe(
        self, 
        audio_file: str, 
        language: Optional[str] = None, 
        context: Optional[str] = None, 
        start_second: float = 0.0,
        duration: float = 0.0,
        temperature: float = 0.4,
        rollback_num: int = 5
    ) -> TranscribeResult:
        """运行完整转录流水线 (从文件加载音频)"""
        from .audio import load_audio
        audio = load_audio(audio_file, start_second=start_second, duration=duration)
        
        return self.asr(
            audio=audio,
            context=context or "",
            language=language,
            chunk_size_sec=self.config.chunk_size,
            memory_chunks=self.config.memory_num,
            temperature=temperature,
            rollback_num=rollback_num
        )

    def asr(
        self, 
        audio: np.ndarray,
        context: Optional[str],
        language: Optional[str],
        chunk_size_sec: float = 40.0,
        memory_chunks: int = 2,
        temperature: float = 0.4,
        rollback_num: int = 5
    ) -> TranscribeResult:
        """运行完整转录流水线 (三级流水线：i+1 预取, i 识别, i-1 对齐)"""
        # 语言归一化与校验
        if language:
            language = normalize_language_name(language)
            validate_language(language)

        sr = 16000
        samples_per_chunk = int(chunk_size_sec * sr)
        total_len = len(audio)
        num_chunks = int(np.ceil(total_len / samples_per_chunk))
        total_duration = total_len / sr
        
        # 记忆管理 (预定义所有分片的物理边界)
        all_segments: List[ASRS_Segment] = [
            ASRS_Segment(
                idx=i,
                audio_start=i * chunk_size_sec,
                audio_end=min((i + 1) * chunk_size_sec, total_duration)
            ) for i in range(num_chunks)
        ]
        asr_memory = deque(maxlen=memory_chunks) # 存储 (embd, text)
        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []
        
        # 统计指标
        stats = {
            "prefill_time": 0.0, "decode_time": 0.0,
            "prefill_tokens": 0, "decode_tokens": 0,
            "encode_time": 0.0, "align_time": 0.0,
        }
        t_main_start = time.time()

        # --- 顺序同步处理循环 ---
        for i in range(num_chunks):
            # 1. 编码第 i 片段
            s, e = i * samples_per_chunk, min((i + 1) * samples_per_chunk, total_len)
            chunk_data = audio[s:e]
            if len(chunk_data) < samples_per_chunk: 
                chunk_data = np.pad(chunk_data, (0, samples_per_chunk - len(chunk_data)))
            
            audio_feature, enc_time = self.encoder.encode(chunk_data)
            stats["encode_time"] += enc_time
            was_last = (i == num_chunks - 1)

            # 2. 识别第 i 片段文字
            prefix_text = "".join([m[1] for m in asr_memory])
            combined_audio = np.concatenate([m[0] for m in asr_memory] + [audio_feature], axis=0)
            full_embd = self._build_prompt_embd(combined_audio, prefix_text, context, language)
            
            # 带熔断加温重试的解码调用
            res = self._safe_decode(full_embd, prefix_text, rollback_num, was_last, temperature)

            # 检测问题文本，跳过记忆防止污染后续分片
            is_clean = bool(re.search(r'[。？！.!?]', res.text)) or len(res.text) < 100
            if not is_clean:
                print(f"[!] 当前分片文本无完整断句，跳过记忆")

            all_segments[i].text = res.text
            asr_memory.append((audio_feature, res.text if is_clean else ""))
            
            total_full_text += res.text
            stats["prefill_tokens"] += res.n_prefill; stats["prefill_time"] += res.t_prefill
            stats["decode_tokens"] += res.n_generate; stats["decode_time"] += res.t_generate

            # 3. 对齐第 i 片段 (同步)
            if self.aligner and res.text.strip():
                t_align_start = time.time()
                # 计算偏移（同步版本逻辑简化：直接使用片起点，不考虑前片动态边界）
                offset_sec = all_segments[i].audio_start
                s_smpl, e_smpl = int(offset_sec * sr), int(all_segments[i].audio_end * sr)
                audio_slice = audio[s_smpl:e_smpl]
                
                align_res = self.aligner.align(
                    audio_slice, 
                    res.text, 
                    language=language, 
                    offset_sec=float(offset_sec)
                )
                all_segments[i].items = align_res.items
                all_aligned_items.extend(align_res.items)
                stats["align_time"] += (time.time() - t_align_start)

        # 4. 结果整理
        all_aligned_items.sort(key=lambda x: x.start_time)
        t_total = time.time() - t_main_start
        if self.verbose: self._print_stats(stats, total_duration, t_total)
            
        return TranscribeResult(
            text=total_full_text,
            alignment=ForcedAlignResult(items=all_aligned_items) if all_aligned_items else None,
            performance=stats
        )
