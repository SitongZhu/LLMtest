#!/usr/bin/env bash
# 在已激活的 conda/venv 中安装 LLMtest 所需 Python 包（torch、LLaMA-Factory、vLLM、评测与 GPTQ 等）。
#
# 用法：
#   conda create -n llmtest python=3.10 -y
#   conda activate llmtest
#   bash /path/to/LLMtest/scripts/setup_env.sh
#
# 环境变量（可选）：
#   SKIP_TORCH_INSTALL=1     跳过 torch/torchvision/torchaudio（已装好 CUDA 版时）
#   TORCH_INDEX_URL=...      指定 PyTorch wheel 源，例如：
#                            https://download.pytorch.org/whl/cu126
#   SKIP_LLAMAFACTORY_EDITABLE=1  跳过 pip install -e LLaMA-Factory（仅补装缺失包时）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LF_SRC="$ROOT/LLaMA-Factory/LLaMA-Factory"

if [[ ! -f "$LF_SRC/pyproject.toml" ]]; then
  echo "Error: 未找到 LLaMA-Factory 源码: $LF_SRC"
  exit 1
fi

echo "Python: $(command -v python3) -> $(python3 -V 2>&1)"

_pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
  :
else
  echo "Error: 需要 Python >= 3.9，当前 $_pyver"
  exit 1
fi

echo ">>> pip 升级 pip / wheel"
python3 -m pip install -U pip wheel

if [[ "${SKIP_TORCH_INSTALL:-0}" != "1" ]]; then
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    echo ">>> 安装 torch（索引: $TORCH_INDEX_URL）"
    python3 -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
  else
    echo ">>> 安装 torch（默认 PyPI）。若需匹配 CUDA，请先:"
    echo "    export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126"
    echo "    然后重新运行本脚本，或设 SKIP_TORCH_INSTALL=1 后手动安装 torch。"
    python3 -m pip install torch torchvision torchaudio
  fi
else
  echo ">>> 跳过 torch 安装（SKIP_TORCH_INSTALL=1）"
fi

if [[ "${SKIP_LLAMAFACTORY_EDITABLE:-0}" != "1" ]]; then
  echo ">>> 可编辑安装 LLaMA-Factory: $LF_SRC"
  python3 -m pip install -e "$LF_SRC"
else
  echo ">>> 跳过 LLaMA-Factory 可编辑安装（SKIP_LLAMAFACTORY_EDITABLE=1）"
fi

echo ">>> metrics / vLLM / GPTQ / huggingface_hub / scikit-learn"
python3 -m pip install -r "$ROOT/requirements-llmtest.txt"

echo ""
echo "完成。建议自检:"
echo "  python3 -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\""
echo "  python3 -c \"import vllm; print('vllm ok')\""
echo "  python3 -c \"from llamafactory.extras.packages import is_vllm_available; print('lf vllm', is_vllm_available())\""
