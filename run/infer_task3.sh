#!/usr/bin/env bash
# LLMtest：自包含仓库；权重默认在 LLMTEST_ROOT/LLaMA-Factory/models，task3 数据在 LLMTEST_ROOT/data/...
# 首次环境：conda activate 你的环境后执行
#   bash LLMtest/scripts/setup_env.sh
#   （可选 CUDA）export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 && bash LLMtest/scripts/setup_env.sh
# 用法（在 GPU 节点、已 conda activate 后）：
#   export PATH="/path/to/env/bin:$PATH"
#   bash /path/to/LLMtest/run/infer_task3.sh
#
# 可选环境变量：
#   TASK3_DATA_DIR       默认：$LLMTEST_ROOT/data/enitre_pipeline/task3（可指向任意含 task3 JSON 的目录）
#   MODEL_ROOT           默认：$LLMTEST_ROOT/LLaMA-Factory/models
#   SAVES_LINK_TARGET    若设置且为目录，则将 LLaMA-Factory/saves 符号链接到该路径（适配器权重；替代旧版「链接到某仓库 saves」用法）
#   HF_TOKEN             传给 huggingface_hub（gated 模型需要）
#   SKIP_MODEL_DOWNLOAD=1  跳过自动下载基座（infer 与 test/task3 均尊重此变量）
#   DEFAULT_TEST_SFT_MODELS / TEST_SFT_MODELS  见 task3.sh
#   其余：TEST_DATASETS、VLLM_*、MASTER_PORT 等

set -euo pipefail

LLMTEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$LLMTEST_ROOT/LLaMA-Factory/models}"
export MODEL_ROOT

mkdir -p "$MODEL_ROOT"

export DEFAULT_TEST_SFT_MODELS="${DEFAULT_TEST_SFT_MODELS:-Qwen2.5-7B-Instruct-GPTQ-Int4,Meta-Llama-3.1-8B-Instruct,Mistral-7B-Instruct-v0.3}"

# 1) 按 TEST_SFT_MODELS（或默认 trio）预拉 HF 基座（与 scripts/ensure_hf_sft_model.py 一致）；失败仅告警，test 内会再尝试
_ENSURE_PY="$LLMTEST_ROOT/scripts/ensure_hf_sft_model.py"
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  _sft_csv="${TEST_SFT_MODELS:-$DEFAULT_TEST_SFT_MODELS}"
  IFS=',' read -r -a _sft_want <<< "$_sft_csv"
  for _m in "${_sft_want[@]}"; do
    _m="${_m//[[:space:]]/}"
    [[ -z "$_m" ]] && continue
    if python3 "$_ENSURE_PY" --model-key "$_m" --model-root "$MODEL_ROOT"; then
      :
    else
      echo "Warning: 预下载失败 $_m（运行 test 时将再尝试或跳过该模型）"
    fi
  done
fi

# 2) task3 数据：仅使用本仓库或 TASK3_DATA_DIR
TASK3_LOCAL="$LLMTEST_ROOT/data/enitre_pipeline/task3"
if [[ -n "${TASK3_DATA_DIR:-}" ]]; then
  export TASK3_DATA_DIR
else
  export TASK3_DATA_DIR="$TASK3_LOCAL"
  if [[ ! -d "$TASK3_DATA_DIR" ]] || ! compgen -G "$TASK3_DATA_DIR"/*.json > /dev/null 2>&1; then
    echo "Warning: 未在 $TASK3_DATA_DIR 找到 task3 JSON。请放入数据或 export TASK3_DATA_DIR=/path/to/task3"
  fi
fi

# 3) SFT adapter：默认使用仓库内 saves/；可选 SAVES_LINK_TARGET 指向外部适配器目录
_LOCAL_SAVES="$LLMTEST_ROOT/LLaMA-Factory/saves"
if [[ -n "${SAVES_LINK_TARGET:-}" ]]; then
  if [[ -d "$SAVES_LINK_TARGET" ]]; then
    ln -sfn "$SAVES_LINK_TARGET" "$_LOCAL_SAVES"
    echo ">>> saves -> $SAVES_LINK_TARGET"
  else
    echo "Warning: SAVES_LINK_TARGET 不是目录，忽略: $SAVES_LINK_TARGET"
  fi
fi
if [[ ! -e "$_LOCAL_SAVES" ]]; then
  mkdir -p "$_LOCAL_SAVES/task3"
  echo ">>> 已创建本地 saves: $_LOCAL_SAVES"
fi

cd "$LLMTEST_ROOT/LLaMA-Factory" || exit 1
exec bash "$LLMTEST_ROOT/LLaMA-Factory/run/test/task3.sh" "$@"
