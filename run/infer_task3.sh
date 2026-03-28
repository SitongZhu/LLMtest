#!/usr/bin/env bash
# LLMtest：自包含仓库；权重默认在 LLMTEST_ROOT/LLaMA-Factory/models，task3 数据在 LLMTEST_ROOT/data/...
# 用法（在 GPU 节点、已 conda activate 后）：
#   export PATH="/path/to/env/bin:$PATH"
#   bash /path/to/LLMtest/run/infer_task3.sh
#
# 可选环境变量：
#   TASK3_DATA_DIR       默认：$LLMTEST_ROOT/data/enitre_pipeline/task3（可指向任意含 task3 JSON 的目录）
#   MODEL_ROOT           默认：$LLMTEST_ROOT/LLaMA-Factory/models
#   SAVES_LINK_TARGET    若设置且为目录，则将 LLaMA-Factory/saves 符号链接到该路径（适配器权重；替代旧版「链接到某仓库 saves」用法）
#   HF_TOKEN             传给 huggingface_hub（gated 模型需要）
#   SKIP_MODEL_DOWNLOAD=1  跳过自动下载基座
#   DEFAULT_TEST_SFT_MODELS / TEST_SFT_MODELS  见 task3.sh
#   其余：TEST_DATASETS、VLLM_*、MASTER_PORT 等

set -euo pipefail

LLMTEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$LLMTEST_ROOT/LLaMA-Factory/models}"
export MODEL_ROOT

mkdir -p "$MODEL_ROOT"

export DEFAULT_TEST_SFT_MODELS="${DEFAULT_TEST_SFT_MODELS:-Qwen2.5-7B-Instruct-GPTQ-Int4,Meta-Llama-3.1-8B-Instruct,Mistral-7B-Instruct-v0.3}"

# 1) 按 TEST_SFT_MODELS（或默认 trio）按需下载 HF 基座到 MODEL_ROOT（与 task3.sh 中 sft_models 路径一致）
download_sft_base_if_needed() {
  local model_key="$1"
  local repo="" dest=""
  case "$model_key" in
    Qwen2.5-7B-Instruct-GPTQ-Int4)
      repo="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
      dest="$MODEL_ROOT/Qwen2.5-7B-Instruct-GPTQ-Int4"
      ;;
    Meta-Llama-3.1-8B-Instruct)
      repo="meta-llama/Meta-Llama-3.1-8B-Instruct"
      dest="$MODEL_ROOT/Llama-3.1-8B"
      ;;
    Mistral-7B-Instruct-v0.3)
      repo="mistralai/Mistral-7B-Instruct-v0.3"
      dest="$MODEL_ROOT/Mistral-7B-Instruct-v0.3"
      ;;
    DeepSeek-R1-Distill-Qwen-7B)
      repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
      dest="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-7B"
      ;;
    *)
      return 0
      ;;
  esac
  if [[ -f "$dest/config.json" ]]; then
    echo ">>> 已存在: $model_key -> $dest"
    return 0
  fi
  echo ">>> 下载 $model_key ($repo) -> $dest"
  HF_TOKEN_ARG=()
  [[ -n "${HF_TOKEN:-}" ]] && HF_TOKEN_ARG=(--token "$HF_TOKEN")
  python3 "$LLMTEST_ROOT/scripts/download_hf_snapshot.py" \
    --repo "$repo" \
    --local-dir "$dest" \
    "${HF_TOKEN_ARG[@]}"
}

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  _sft_csv="${TEST_SFT_MODELS:-$DEFAULT_TEST_SFT_MODELS}"
  IFS=',' read -r -a _sft_want <<< "$_sft_csv"
  for _m in "${_sft_want[@]}"; do
    _m="${_m//[[:space:]]/}"
    [[ -n "$_m" ]] && download_sft_base_if_needed "$_m"
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
