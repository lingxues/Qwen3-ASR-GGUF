# coding=utf-8
import re
from datetime import timedelta
from typing import List, Optional
import srt
import json
from .schema import ForcedAlignResult, ForcedAlignItem, TranscribeResult
from .chinese_itn import chinese_to_num as itn

def alignment_to_srt(items: Optional[List[ForcedAlignItem]], max_chars: int = 40) -> str:
    """
    将对齐结果转换为 SRT 格式内容。
    按逗号、句号、问号、感叹号以及换行符进行分行。
    """
    if not items:
        return ""

    subtitles = []
    current_texts = []
    start_time = None
    
    # 匹配分割符号：中文全角标点、英文半角标点、以及可能存在的换行符
    # 特别注意：在有的 ASR 引擎中，标点后可能跟有空格，也一并匹配
    split_pattern = re.compile(r'[，。？！、\n]|[,.?!]\s*')
    
    for item in items:
        # 记录每一行字幕的开始时间
        if start_time is None:
            start_time = item.start_time
        
        current_texts.append(item.text)
        
        # 聚合当前已有的文本
        current_content = "".join(current_texts)
        
        # 触发分割的条件：
        # 1. 遇到了分割标点符号
        # 2. 或者当前行字符数超过了 max_chars (防止单行过长)
        if split_pattern.search(item.text) or len(current_content) >= max_chars:
            content = current_content.strip()
            if content:
                # 移除末尾标点
                content = content.rstrip("，。？！、,.?!")
                # 应用 ITN 处理
                itn_content = itn(content)
                subtitles.append(srt.Subtitle(
                    index=len(subtitles) + 1,
                    start=timedelta(seconds=start_time),
                    end=timedelta(seconds=item.end_time),
                    content=itn_content
                ))
            current_texts = []
            start_time = None
            
    # 处理末尾残余文本
    if current_texts:
        content = "".join(current_texts).strip()
        if content:
            # 移除末尾标点
            content = content.rstrip("，。？！：、,.?!")
            # 应用 ITN 处理
            itn_content = itn(content)
            end_time = items[-1].end_time
            subtitles.append(srt.Subtitle(
                index=len(subtitles) + 1,
                start=timedelta(seconds=start_time),
                end=timedelta(seconds=end_time),
                content=itn_content
            ))
            
    return srt.compose(subtitles)

def alignment_to_json(items: Optional[List[ForcedAlignItem]]) -> List[dict]:
    """将对齐结果转换为可序列化的字典列表"""
    if not items:
        return []
    return [
        {
            "text": it.text,
            "start": round(it.start_time, 3),
            "end": round(it.end_time, 3)
        }
        for it in items
    ]

def export_to_srt(path: str, result: TranscribeResult):
    """将对齐结果保存为 SRT 文件"""
    if not result.alignment:
        with open(path, "w", encoding="utf-8") as f: f.write("")
        return
    
    content = alignment_to_srt(result.alignment.items)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ 已生成字幕文件: {path}")

def export_to_json(path: str, result: TranscribeResult):
    """将对齐结果保存为 JSON 文件"""
    if not result.alignment:
        with open(path, "w", encoding="utf-8") as f: f.write("[]")
        return

    data = alignment_to_json(result.alignment.items)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已导出时间戳: {path}")

def smart_line_break(text: str, max_chars: int = 80) -> str:
    """
    智能换行：对超长无标点句子，寻找自然断句点换行。
    支持口语化直播场景的断句。
    """
    if len(text) <= max_chars:
        return text
    
    # 中文口语填充词/语气词（直播场景高频出现，表示停顿）
    cn_fillers = [
        # 语气词（表示停顿/思考/感叹）
        '嗯', '啊', '哎', '哦', '呃', '呀', '啦', '吧', '呢', '嘛', '哈',
        '哇', '唉', '喂', '嘿', '哟', '唉呀', '哎呀', '哎呦', '哎哟',
        '我靠', '我擦', '我勒个', '我的天', '天哪', '我去', '我晕',
        '妈呀', '卧槽', '牛逼', '绝了', '服了', '醉了', '晕',
        '噗', '啧', '嘁', '哼', '呵', '嘻', '嚯', '嘎', '嘭',
        '啊啊啊', '哈哈哈', '嘿嘿嘿', '哇哇哇',
        # 口头禅/连接词
        '就是说', '怎么说呢', '那个', '这个', '然后', '接着', '所以',
        '反正', '总之', '对吧', '是吧', '对不对', '你知道',
        '不是', '没有', '真的', '确实', '其实', '毕竟', '究竟',
        '话说回来', '说到底', '归根结底', '总而言之',
        # 话题转换
        '好了', '行了', '算了', '得了', '够了',
        '对了', '话说', '说到', '提到', '想起',
        '另外', '还有', '顺便说一下',
        # 直播互动
        '老师', '朋友', '兄弟', '姐妹', '观众', '大家', '各位',
        '老铁', '家人们', '宝子', '亲', '亲们', '兄弟们', '姐妹们',
        '小姐姐', '小哥哥', '大佬', '老板', '大哥', '大姐',
        '谢谢', '感谢', '欢迎', '拜托', '求求',
        '恭喜', '厉害', '牛', '666', '绝绝子',
        # 直播常用语
        '关注', '点赞', '分享', '收藏', '订阅',
        '刷礼物', '打赏', '送礼', '火箭', '飞机',
        '上车', '下车', '上票', '冲', '干', '奥利给',
    ]
    
    # 中文断句词（按优先级排序）
    cn_breaks = [
        # 转折/承接
        '但是', '不过', '然而', '可是', '只是', '就是', '然后', '接着', '于是',
        '却', '倒是', '反而', '相反', '要不然', '不然', '否则',
        # 因果
        '因为', '所以', '因此', '由于', '既然', '之所以', '怪不得', '难怪',
        # 条件/假设
        '如果', '要是', '假如', '假设', '假使', '倘若', '倘使',
        '只要', '只有', '除非', '万一', '一旦',
        # 让步/转折
        '虽然', '虽说', '尽管', '固然', '即使', '即便', '哪怕', '就是',
        '纵然', '纵使', '无论', '不管', '不论',
        # 并列/选择
        '或者', '还是', '要么', '不是', '而是', '不但', '不仅',
        '而且', '并且', '况且', '何况', '以及', '还有', '同时',
        # 否定/肯定
        '没有', '不要', '不能', '不会', '不应该', '不可能',
        '可以', '应该', '需要', '必须', '一定', '肯定',
        # 语气词/助词
        '的话', '来说', '而言', '至于', '关于',
        '时候', '之后', '之前', '以前', '以后', '以来',
        '同时', '当时', '那时', '这时',
        # 常见动词搭配
        '觉得', '认为', '知道', '希望', '相信', '发现', '看到', '听到',
        '感觉', '感觉', '想要', '打算', '准备', '决定',
        '承认', '否认', '确认', '同意', '反对', '支持',
        # 连词
        '和', '与', '或', '但', '而', '也', '还', '就', '都', '才', '又', '再',
        # 助词/介词
        '的', '了', '着', '过', '地', '得', '在', '把', '被', '让', '给', '对',
        '从', '向', '往', '到', '比', '跟', '同',
    ]
    
    # 英文自然断句词
    en_breaks = [
        # 连词
        'and', 'but', 'or', 'so', 'yet', 'nor',
        # 从属连词
        'because', 'although', 'though', 'while', 'when', 'where', 'if', 'unless',
        'until', 'after', 'before', 'since', 'as', 'that', 'which', 'who', 'whom',
        # 副词/过渡词
        'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
        'meanwhile', 'otherwise', 'instead', 'still', 'also', 'then',
        # 介词
        'with', 'from', 'into', 'through', 'during', 'between', 'about',
        'against', 'among', 'along', 'across', 'behind', 'below', 'above',
    ]
    
    # 合并所有断句词，按优先级排序
    all_breaks = []
    for w in cn_fillers:
        all_breaks.append((w, 0))  # 最高优先级
    for w in cn_breaks:
        all_breaks.append((w, 1))
    for w in en_breaks:
        all_breaks.append((w, 2))
    
    result_lines = []
    remaining = text
    
    while len(remaining) > max_chars:
        # 在整个 [0, max_chars] 范围内寻找所有断句点
        search_end = min(max_chars, len(remaining))
        search_region = remaining[:search_end]
        
        best_pos = -1
        best_priority = 999
        best_word_len = 0
        
        # 遍历所有断句词，寻找最佳断点（位置越靠后越好，但不超过max_chars）
        for word, priority in all_breaks:
            # 从后向前搜索，优先使用靠后的断句点
            pos = search_region.rfind(word)
            if pos != -1:
                # 位置必须在合理范围内（至少留10个字符）
                if pos < 10:
                    continue
                # 优先级更高，或相同优先级但位置更靠后
                if priority < best_priority or (priority == best_priority and pos > best_pos):
                    best_priority = priority
                    best_pos = pos
                    best_word_len = len(word)
        
        # 如果找到断句点，在该点换行
        if best_pos > 0:
            break_line = remaining[:best_pos].rstrip()
            if break_line:
                result_lines.append(break_line)
            remaining = remaining[best_pos:].lstrip()
        else:
            # 没找到断句词，尝试检测重复模式断句
            repeat_break = find_repeat_break(remaining, search_end)
            if repeat_break > 10:
                result_lines.append(remaining[:repeat_break])
                remaining = remaining[repeat_break:]
            else:
                # 最后手段：在空格处断行
                space_pos = remaining.rfind(' ', max_chars // 2, search_end)
                if space_pos > 10:
                    result_lines.append(remaining[:space_pos])
                    remaining = remaining[space_pos + 1:]
                else:
                    # 强制断行
                    result_lines.append(remaining[:max_chars])
                    remaining = remaining[max_chars:]
    
    if remaining:
        result_lines.append(remaining)
    
    return '\n'.join(result_lines)


def find_repeat_break(text: str, max_pos: int) -> int:
    """
    检测文本中的重复模式，在重复边界处返回断句位置。
    例如：'早上好早上好早上好' -> 在第一个'早上好'后断句
    """
    # 尝试不同长度的子串（2-10个字符）
    for substr_len in range(2, min(11, max_pos // 3)):
        # 检查从位置0开始的子串是否重复
        substr = text[:substr_len]
        count = 0
        pos = 0
        while pos + substr_len <= len(text) and text[pos:pos + substr_len] == substr:
            count += 1
            pos += substr_len
        
        # 如果重复3次以上，在第一次重复结束后断句
        if count >= 3:
            return substr_len
    
    # 尝试从后向前检测重复（处理'abcabcabc'类型）
    for substr_len in range(2, min(11, max_pos // 3)):
        # 在max_pos范围内查找重复
        check_region = text[:max_pos]
        substr = check_region[-substr_len:]
        if len(substr) < substr_len:
            continue
        
        # 检查这个子串在前面是否也出现
        count = 0
        pos = max_pos - substr_len
        while pos >= 0 and text[pos:pos + substr_len] == substr:
            count += 1
            pos -= substr_len
        
        # 如果重复3次以上，断在重复开始前
        if count >= 3:
            return pos + substr_len
    
    return -1


def export_to_txt(path: str, result: TranscribeResult):
    """将转录结果处理后保存为 TXT 文件 (含 ITN 和标点换行)"""
    # 1. ITN 处理
    final_text = itn(result.text)
    # 2. 按照标点符号换行，保留标点
    formatted_text = re.sub(r'([，。？！：])', r'\1\n', final_text)
    # 3. 对于英文字母后面的逗号空格、句号空格，也要换行
    formatted_text = re.sub(r'(?<=[a-zA-Z])([,\.] )', r'\1\n', formatted_text)
    # 4. 对超长行进行智能换行（口语场景用60字符更易读）
    lines = formatted_text.split('\n')
    processed_lines = [smart_line_break(line, max_chars=60) for line in lines]
    formatted_text = '\n'.join(processed_lines)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    print(f"✅ 已保存文本文件: {path}")
