
from pathlib import Path

# [源模型路径] 官方下载好的 SafeTensors 模型文件夹
ASR_MODEL_DIR = Path(r'D:\ai\Qwen3-ASR-GGUF-main\Qwen3-ASR-1.7B')
ALIGNER_MODEL_DIR = Path(r'D:\ai\Qwen3-ASR-GGUF-main\Qwen3-ForcedAligner-0.6B')

# [导出目标路径] 转换后的 ONNX, GGUF 和权重汇总目录
EXPORT_DIR = r'D:\ai\Qwen3-ASR-GGUF-main\model'
