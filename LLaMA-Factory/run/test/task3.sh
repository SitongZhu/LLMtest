#!/usr/bin/env bash
set -euo pipefail

# Change to LLaMA-Factory directory to ensure relative paths work correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$LLAMA_FACTORY_DIR" || exit 1

# 本地/迁移环境：权重放在 LLaMA-Factory/models（可用 MODEL_ROOT 覆盖）
MODEL_ROOT="${MODEL_ROOT:-$LLAMA_FACTORY_DIR/models}"

# Add LLaMA-Factory src to PYTHONPATH so llamafactory module can be found
if [[ -d "$LLAMA_FACTORY_DIR/LLaMA-Factory/src" ]]; then
    export PYTHONPATH="$LLAMA_FACTORY_DIR/LLaMA-Factory/src${PYTHONPATH:+:$PYTHONPATH}"
elif [[ -d "$LLAMA_FACTORY_DIR/src" ]]; then
    export PYTHONPATH="$LLAMA_FACTORY_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
fi

# GPUs for inference
# Convert MIG UUIDs to device indices if needed
# If CUDA_VISIBLE_DEVICES contains MIG UUIDs, we need to map them to indices
if [[ -n "$CUDA_VISIBLE_DEVICES" ]] && [[ "$CUDA_VISIBLE_DEVICES" == *"MIG-"* ]]; then
    echo "Warning: MIG UUIDs detected in CUDA_VISIBLE_DEVICES. Converting to device indices..."
    export CUDA_VISIBLE_DEVICES="0"
    echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

# Force single GPU by default (stability over multi-GPU experiments).
# To restore old behavior, run with: FORCE_SINGLE_GPU=0
FORCE_SINGLE_GPU="${FORCE_SINGLE_GPU:-1}"
if [[ "${FORCE_SINGLE_GPU}" == "1" ]]; then
  # If multiple GPUs are visible, keep only the first one.
  if [[ "$CUDA_VISIBLE_DEVICES" == *","* ]]; then
    echo "Warning: FORCE_SINGLE_GPU=1: CUDA_VISIBLE_DEVICES has multiple GPUs, using first only."
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export LOCAL_RANK=0
  export RANK=0
else
  # 单卡时强制 rank=0，避免 Transformers 使用 device 1 导致 device=1,num_gpus=1 报错
  if [[ "$CUDA_VISIBLE_DEVICES" == *","* ]]; then
      :  # 多卡时不覆盖 RANK/LOCAL_RANK
  else
      export LOCAL_RANK=0
      export RANK=0
  fi
fi
export DISABLE_VERSION_CHECK=1

# 避免 load_dataset 缓存写满 home 配额（OSError: [Errno 122] Disk quota exceeded）
# 若未设置 HF_DATASETS_CACHE，则用 TMPDIR 或 /tmp，减少占用 home 空间
if [[ -z "${HF_DATASETS_CACHE:-}" ]]; then
  _hf_cache="${TMPDIR:-/tmp}/hf_datasets_cache_$$"
  mkdir -p "$_hf_cache" 2>/dev/null && export HF_DATASETS_CACHE="$_hf_cache" && echo "Using HF_DATASETS_CACHE=$HF_DATASETS_CACHE (avoid disk quota on home)"
fi

# Tensor parallel size:
# - Default is single GPU (TENSOR_PARALLEL_SIZE=1) to avoid CUDA OOM / device mismatch.
# - If you really want multi-GPU, run with FORCE_SINGLE_GPU=0 and set USE_4GPU=1.
if [[ "${FORCE_SINGLE_GPU}" == "1" ]]; then
  TENSOR_PARALLEL_SIZE=1
  if [[ "${USE_4GPU:-0}" == "1" ]]; then
    echo "Warning: FORCE_SINGLE_GPU=1: ignoring USE_4GPU=1 and using tensor_parallel_size=1."
  fi
else
  # 4-GPU 推理：设置 USE_4GPU=1 时使用 tensor_parallel_size=4，充分利用 4 张卡跑大模型
  # 可同时设置 NOSFT_MODELS_4GPU="Qwen2.5-32B-Instruct,Qwen2.5-72B-Instruct,..." 仅跑大模型
  # 注意：必须保证实际有 4 张 GPU（如 srun --gres=gpu:4），否则会报 device=1, num_gpus=1
  if [[ "${USE_4GPU:-0}" == "1" ]]; then
      # 根据 CUDA_VISIBLE_DEVICES 判断当前可见 GPU 数量，避免申请了 4 卡却只看到 1 卡
      _gpu_count=0
      if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
          _gpu_count=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c . || true)
          [[ "$_gpu_count" -lt 1 ]] && _gpu_count=0
      fi
      if [[ "$_gpu_count" -ge 4 ]]; then
          TENSOR_PARALLEL_SIZE=4
          echo "USE_4GPU=1: using tensor_parallel_size=4 (${_gpu_count} GPUs visible)"
      else
          TENSOR_PARALLEL_SIZE="${_gpu_count:-1}"
          [[ "$TENSOR_PARALLEL_SIZE" -lt 1 ]] && TENSOR_PARALLEL_SIZE=1
          echo "Warning: USE_4GPU=1 but only ${_gpu_count:-1} GPU(s) visible (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES). Using tensor_parallel_size=$TENSOR_PARALLEL_SIZE. For 4-GPU run, use e.g. srun --gres=gpu:4 bash run/test/task3.sh"
      fi
      if [[ -n "${NOSFT_MODELS_4GPU:-}" ]]; then
          IFS=',' read -r -a nosft_models_4gpu <<< "$NOSFT_MODELS_4GPU"
          nosft_models=("${nosft_models_4gpu[@]}")
          echo "Using NOSFT_MODELS_4GPU: ${nosft_models[*]}"
      fi
  else
      TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
  fi
fi

echo "Using single-GPU inference: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"

# Zero-shot 基座目录：默认与 SFT 相同（MODEL_ROOT）；需要单独 HF 缓存树时再 export HF_MODEL_ROOT
HF_MODEL_ROOT="${HF_MODEL_ROOT:-$MODEL_ROOT}"

# Task3 数据目录：自动发现 dataset 时扫描此目录下所有 .json（可覆盖 TASK3_DATA_DIR）
# 默认：<LLMtest>/data/enitre_pipeline/task3（与 LLMtest/LLaMA-Factory 平级的 data/）
TASK3_DATA_DIR="${TASK3_DATA_DIR:-$LLAMA_FACTORY_DIR/../data/enitre_pipeline/task3}"
DATASET_INFO_FILE="${DATASET_INFO_FILE:-$LLAMA_FACTORY_DIR/data/dataset_info.json}"
SYNC_TASK3_SCRIPT="$LLAMA_FACTORY_DIR/run/test/sync_task3_datasets.py"

# Dirs
mkdir -p generate/task3 generate/task3_group
mkdir -p log/task3/infer log/task3_group/infer

# Zero-shot models (no adapter)
nosft_models=(
  "gpt-oss-120b"
  "gpt-oss-20b"
  "Qwen3-0.6B"
  "Qwen3-1.7B"
  "Qwen3-8B"
  "Qwen3-14B"
  "Qwen3-32B"
  "Qwen3-72B"
  "Qwen2.5-0.5B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-3B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
  "Qwen2.5-32B-Instruct"
  "Qwen2.5-72B-Instruct"
  "DeepSeek-R1-Distill-Qwen-14B"
  "Meta-Llama-3.1-8B-Instruct"
  "Mistral-7B-Instruct-v0.3"
)

# Models with SFT adapters; base model paths.
# Qwen GPTQ Int4 only (fits 16GB). Uncomment DeepSeek for 24GB+ GPU.
# Must match training outputs:
# - train_attitude:      saves/task3/${model_tag}/lora/sft
# - train_attitude_group: saves/task3/${model_tag}_group/lora/sft
declare -A sft_models=(
  ["DeepSeek-R1-Distill-Qwen-7B"]="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-7B"
  ["Qwen2.5-7B-Instruct-GPTQ-Int4"]="$MODEL_ROOT/Qwen2.5-7B-Instruct-GPTQ-Int4"
  ["Meta-Llama-3.1-8B-Instruct"]="$MODEL_ROOT/Llama-3.1-8B"
  ["Mistral-7B-Instruct-v0.3"]="$MODEL_ROOT/Mistral-7B-Instruct-v0.3"
  # ["Qwen2.5-7B-Instruct"]="$HF_MODEL_ROOT/Qwen2.5-7B-Instruct"
  # ["DeepSeek-R1-Distill-Qwen-14B"]="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-14B"
)

sanitize() {
  local s="$1"
  s="${s//\//_}"
  s="${s//./_}"
  s="${s// /_}"
  echo "$s"
}

parse_datasets() {
  local raw="${1:-}"
  local -a out=()
  if [[ -z "$raw" ]]; then
    echo ""
    return 0
  fi
  IFS=',' read -r -a out <<<"$raw"
  for i in "${!out[@]}"; do
    out[$i]="${out[$i]//[[:space:]]/}"
  done
  echo "${out[*]}"
}

check_dataset_exists() {
  local dataset_name="$1"
  local dataset_info_file="$LLAMA_FACTORY_DIR/data/dataset_info.json"
  if [[ ! -f "$dataset_info_file" ]]; then
    return 1
  fi
  if python3 -c "import json; d=json.load(open('$dataset_info_file')); exit(0 if '$dataset_name' in d else 1)" 2>/dev/null; then
    local file_path
    file_path=$(python3 -c "import json; d=json.load(open('$dataset_info_file')); print(d.get('$dataset_name', {}).get('file_name', ''))" 2>/dev/null)
    if [[ "$file_path" == ../../* ]]; then
      file_path="$LLAMA_FACTORY_DIR/../../${file_path#../../}"
    elif [[ "$file_path" == ../* ]]; then
      file_path="$LLAMA_FACTORY_DIR/$file_path"
    elif [[ "$file_path" != /* ]]; then
      file_path="$LLAMA_FACTORY_DIR/$file_path"
    fi
    [[ -f "$file_path" ]] && return 0
  fi
  return 1
}

get_template() {
  local name="$1"
  if [[ "$name" == gpt-oss-* ]]; then
    echo "gpt"
  elif [[ "$name" == Qwen3-* ]]; then
    echo "qwen3"
  elif [[ "$name" == Qwen2.5-* ]]; then
    echo "qwen"
  elif [[ "$name" == *"DeepSeek-R1"* ]]; then
    echo "deepseekr1"
  elif [[ "$name" == *"Meta-Llama-3.1"* ]]; then
    echo "llama3"
  elif [[ "$name" == *"Mistral"* ]]; then
    echo "mistral"
  else
    echo "qwen"
  fi
}

run_infer() {
  local model_name="$1"
  local model_path="$2"
  local template="$3"
  local dataset="$4"
  local save_name="$5"
  local log_file="$6"
  local adapter_path="${7:-}"

  echo "Inference started for ${model_name}..."
  # Configure vLLM based on model type
  local vllm_config="{}"
  if [[ "$model_name" == *"GPTQ"* ]]; then
    # Prefer gptq_marlin over standard gptq
    # Note: Use device index instead of MIG UUID to avoid parsing errors
    vllm_config='{"enforce_eager": true, "quantization": "gptq_marlin", "gpu_memory_utilization": 0.9}'
    echo "Using enforce_eager=True and quantization=gptq_marlin for GPTQ model"
  elif [[ "$model_name" == *"Llama"* ]] && [[ "$model_name" == *"8B"* ]]; then
    # For Llama-3.1-8B model on single GPU, use conservative memory settings
    # Reduce max_num_seqs to avoid OOM during warmup (default is 1024)
    # Lower max_model_len to reduce KV cache memory requirements
    # Use enforce_eager to disable CUDA graph to avoid OOM during initialization
    vllm_config='{"gpu_memory_utilization": 0.7, "max_model_len": 2048, "max_num_seqs": 256, "enforce_eager": true, "swap_space": 4}'
    echo "Using gpu_memory_utilization=0.7, max_model_len=2048, max_num_seqs=256, enforce_eager=True, swap_space=4GB for Llama-3.1-8B model"
    echo "Note: Lower max_num_seqs reduces memory during warmup phase"
    echo "Note: Lower max_model_len reduces KV cache memory requirements"
    echo "Note: swap_space enables CPU offload for KV cache if needed"
    echo "Note: tensor_parallel_size will be automatically calculated by vllm_infer.py"
  elif [[ "$model_name" == *"Mistral"* ]] && [[ "$model_name" == *"7B"* ]]; then
    vllm_config='{"gpu_memory_utilization": 0.7, "max_model_len": 4096, "max_num_seqs": 128, "enforce_eager": true, "swap_space": 4}'
    echo "Using conservative vLLM settings for Mistral-7B-Instruct"
  elif [[ "$model_name" == *"DeepSeek-R1"* ]] && [[ "$model_name" == *"7B"* ]]; then
    # DeepSeek-R1-7B：max_num_seqs 加大可并行更多条、推理更快；max_model_len 需≥650（部分 prompt 较长）
    max_len="${VLLM_MAX_MODEL_LEN:-1024}"
    max_seqs="${VLLM_MAX_NUM_SEQS:-64}"
    gpu_util="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
    swap_space_gb="${VLLM_SWAP_SPACE_GB:-2}"
    vllm_config="{\"gpu_memory_utilization\": ${gpu_util}, \"max_model_len\": ${max_len}, \"max_num_seqs\": ${max_seqs}, \"enforce_eager\": true, \"swap_space\": ${swap_space_gb}}"
    echo "Using gpu_memory_utilization=${gpu_util}, max_model_len=${max_len}, max_num_seqs=${max_seqs}, enforce_eager=True, swap_space=${swap_space_gb}GB for DeepSeek-R1-7B (tuned)"
    echo "Note: If OOM set VLLM_MAX_NUM_SEQS=16. Long prompts need max_model_len≥650."
    # Reduce fragmentation on tight memory (PyTorch suggestion from OOM message)
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  elif [[ "$model_name" == *"DeepSeek-R1"* ]] || [[ "$model_name" == *"14B"* ]] || [[ "$model_name" == *"32B"* ]] || [[ "$model_name" == *"72B"* ]]; then
    # For large models (14B+), use lower memory utilization; with 4 GPUs use tensor_parallel_size=4
    vllm_config='{"gpu_memory_utilization": 0.90, "max_model_len": 4096, "max_num_seqs": 64}'
    echo "Using gpu_memory_utilization=0.90, max_model_len=4096 for large model ${model_name}"
  fi

  # 注入 tensor_parallel_size，使 vLLM 使用多卡（USE_4GPU=1 时为 4）
  local tp="${TENSOR_PARALLEL_SIZE:-1}"
  vllm_config=$(echo "$vllm_config" | python3 -c "import sys,json; d=json.load(sys.stdin); d['tensor_parallel_size']=$tp; print(json.dumps(d))")
  echo "Using tensor_parallel_size=$tp"

  # Determine the correct path to vllm_infer.py
  if [[ -f "$LLAMA_FACTORY_DIR/LLaMA-Factory/scripts/vllm_infer.py" ]]; then
    VLLM_INFER_SCRIPT="$LLAMA_FACTORY_DIR/LLaMA-Factory/scripts/vllm_infer.py"
  elif [[ -f "$LLAMA_FACTORY_DIR/scripts/vllm_infer.py" ]]; then
    VLLM_INFER_SCRIPT="$LLAMA_FACTORY_DIR/scripts/vllm_infer.py"
  else
    echo "Error: vllm_infer.py not found"
    return 1
  fi

  if [[ -n "$adapter_path" ]]; then
    python "$VLLM_INFER_SCRIPT" \
      --model_name_or_path "$model_path" \
      --adapter_name_or_path "$adapter_path" \
      --template "$template" \
      --dataset "$dataset" \
      --save_name "$save_name" \
      --vllm_config "$vllm_config" \
      > "$log_file" 2>&1
  else
    python "$VLLM_INFER_SCRIPT" \
      --model_name_or_path "$model_path" \
      --template "$template" \
      --dataset "$dataset" \
      --save_name "$save_name" \
      --vllm_config "$vllm_config" \
      > "$log_file" 2>&1
  fi
  echo "Inference finished for ${model_name}"
}

# Multi-wave / multi-dataset support:
# - Default: 自动扫描 TASK3_DATA_DIR 下所有 .json，同步到 dataset_info.json 并用于 test
# - TASK3_ALL=1: 使用固定列表（旧行为，不含 group）
# - TASK3_GROUP_ALL=1: 仅 test_attitude_group, train_attitude_group
# - TEST_DATASETS="a,b,c": 显式指定 dataset 列表
# - TASK3_INCLUDE_GROUP=1: 自动扫描时同时包含 task3/group/*.json
# - TEST_ONLY_SFT=1（默认）: 只跑现有 SFT 模型，不跑 zero-shot（本地/HF 模型未下载时用）。要跑 zero-shot 请设 TEST_ONLY_SFT=0
# - TEST_SFT_MODELS="ModelA,ModelB": 只跑列表中的 SFT 模型。未设置时默认跑 DEFAULT_TEST_SFT_MODELS（见下方）
# - DEFAULT_TEST_SFT_MODELS: 逗号分隔；未设置时内置为 Qwen GPTQ + Llama-3.1-8B + Mistral-7B
raw_datasets="${TEST_DATASETS:-${DATASETS:-}}"
if [[ -n "$raw_datasets" ]]; then
  read -r -a datasets <<<"$(parse_datasets "$raw_datasets")"
  echo "Using TEST_DATASETS: ${datasets[*]}"
elif [[ "${TASK3_ALL:-0}" == "1" ]]; then
  datasets=("prompt_w22" "prompt_w22_tanchored" "prompt_w25" "prompt_w25_tanchored" "prompt_w25_trajectory" "prompt_w26" "prompt_w26_tanchored" "prompt_w26_trajectory" "prompt_w28" "prompt_w28_tanchored" "prompt_w28_trajectory")
  echo "TASK3_ALL=1: using fixed task3 list (no group): ${datasets[*]}"
elif [[ "${TASK3_GROUP_ALL:-0}" == "1" ]]; then
  datasets=("test_attitude_group" "train_attitude_group")
  echo "TASK3_GROUP_ALL=1: using group datasets: ${datasets[*]}"
else
  # Default: 自动读取 TASK3_DATA_DIR 下所有 .json，同步到 dataset_info.json 并作为 test 的 dataset 列表
  TASK3_DATA_ABS=""
  [[ -d "$TASK3_DATA_DIR" ]] && TASK3_DATA_ABS="$(cd "$TASK3_DATA_DIR" 2>/dev/null && pwd)"
  if [[ -f "$SYNC_TASK3_SCRIPT" ]] && [[ -n "$TASK3_DATA_ABS" ]]; then
    python3 "$SYNC_TASK3_SCRIPT" --task3_dir "$TASK3_DATA_ABS" --dataset_info "$DATASET_INFO_FILE" ${TASK3_INCLUDE_GROUP:+--include_group} 2>/dev/null || true
    mapfile -t datasets < <(python3 "$SYNC_TASK3_SCRIPT" --task3_dir "$TASK3_DATA_ABS" --print-names-only ${TASK3_INCLUDE_GROUP:+--include_group} 2>/dev/null)
    if [[ ${#datasets[@]} -eq 0 ]]; then
      echo "Warning: no JSON files found in $TASK3_DATA_DIR, falling back to fixed list"
      datasets=("prompt_w22" "prompt_w22_tanchored" "prompt_w25" "prompt_w25_tanchored" "prompt_w25_trajectory" "prompt_w26" "prompt_w26_tanchored" "prompt_w26_trajectory" "prompt_w28" "prompt_w28_tanchored" "prompt_w28_trajectory")
    fi
  else
    datasets=("prompt_w22" "prompt_w22_tanchored" "prompt_w25" "prompt_w25_tanchored" "prompt_w25_trajectory" "prompt_w26" "prompt_w26_tanchored" "prompt_w26_trajectory" "prompt_w28" "prompt_w28_tanchored" "prompt_w28_trajectory")
  fi
  echo "Default: using all JSON in task3 dir ($TASK3_DATA_DIR): ${datasets[*]}"
fi

# 未指定 TEST_SFT_MODELS 时，默认只跑 trio（不是 sft_models 全部键，避免误跑 DeepSeek 等）
DEFAULT_TEST_SFT_MODELS="${DEFAULT_TEST_SFT_MODELS:-Qwen2.5-7B-Instruct-GPTQ-Int4,Meta-Llama-3.1-8B-Instruct,Mistral-7B-Instruct-v0.3}"
if [[ -z "${TEST_SFT_MODELS:-}" ]]; then
  TEST_SFT_MODELS="$DEFAULT_TEST_SFT_MODELS"
fi
export TEST_SFT_MODELS
echo "TEST_SFT_MODELS=${TEST_SFT_MODELS}"

[[ "${TEST_ONLY_SFT:-1}" == "1" ]] && echo "TEST_ONLY_SFT=1 (default): only running existing SFT models (no zero-shot)."

# Run each dataset in a separate invocation (one dataset per run; good for parallel jobs or clear logs)
if [[ "${RUN_EACH_DATASET_SEPARATELY:-0}" == "1" ]] && [[ ${#datasets[@]} -gt 0 ]]; then
  echo "RUN_EACH_DATASET_SEPARATELY=1: running test once per dataset (${#datasets[@]} runs)."
  for one_dataset in "${datasets[@]}"; do
    echo "=== Test for dataset: $one_dataset ==="
    TEST_DATASETS="$one_dataset" RUN_EACH_DATASET_SEPARATELY=0 bash "$SCRIPT_DIR/task3.sh" || true
  done
  echo "All per-dataset test runs finished."
  exit 0
fi

for dataset in "${datasets[@]}"; do
  echo ""
  echo "========== Dataset: $dataset =========="
  if ! check_dataset_exists "$dataset"; then
    echo "Dataset '$dataset' not found, skipping."
    continue
  fi

  # All task3/group datasets (e.g. test_attitude_group, train_attitude_group) -> task3_group output
  if [[ "$dataset" == *"attitude_group" ]] || [[ "$dataset" == *"_group" ]]; then
    gen_dir="generate/task3_group"
    log_dir="log/task3_group/infer"
    group_suffix="_group"
  else
    gen_dir="generate/task3"
    log_dir="log/task3/infer"
    group_suffix=""
  fi

  mkdir -p "$gen_dir" "$log_dir"
  dataset_tag="$(sanitize "$dataset")"
  multi_dataset=0
  if [[ ${#datasets[@]} -gt 1 ]]; then
    multi_dataset=1
  fi
  # Optional: use adapter from a specific training run (e.g. TRAIN_RUN=wave11train_20260218_223831)
  train_run_tag=""
  if [[ -n "${TRAIN_RUN:-}" ]]; then
    train_run_tag="$(sanitize "$TRAIN_RUN")"
    echo "Using adapter from training run: TRAIN_RUN=$TRAIN_RUN -> $train_run_tag"
  fi

  # USE_BASE_ONLY=1: run SFT base models without adapter (original pretrained model only)
  use_base_only=0
  if [[ "${USE_BASE_ONLY:-0}" == "1" ]]; then
    use_base_only=1
    echo "USE_BASE_ONLY=1: will run SFT models without adapter (base/original model only)"
  fi

  # 1) SFT inference (adapter_name_or_path must equal training output_dir); or base-only if USE_BASE_ONLY=1
  for model_name in "${!sft_models[@]}"; do
    # 若设置了 TEST_SFT_MODELS，只跑列表中的模型
    if [[ -n "${TEST_SFT_MODELS:-}" ]]; then
      if [[ ",${TEST_SFT_MODELS}," != *",${model_name},"* ]]; then
        echo "Skipping ${model_name} (not in TEST_SFT_MODELS=${TEST_SFT_MODELS})"
        continue
      fi
    fi
    model_path="${sft_models[$model_name]}"
    template="$(get_template "$model_name")"
    model_tag="$(sanitize "$model_name")"
    adapter_dir=""

    if [[ $use_base_only -eq 1 ]]; then
      # No adapter: save as nosft_* so you can compare base vs finetuned (sft_*). Always include dataset in name.
      save_file="${gen_dir}/nosft_${model_tag}__${dataset_tag}.jsonl"
      log_file="${log_dir}/nosft_${model_tag}__${dataset_tag}.log"
      echo "Base-only inference for ${model_name} (dataset=${dataset}) -> $save_file"
    else
      if [[ -n "$train_run_tag" ]]; then
        adapter_dir="saves/task3/${model_tag}/${train_run_tag}/lora/sft"
        save_file="${gen_dir}/sft_${model_tag}__train_${train_run_tag}__${dataset_tag}.jsonl"
        log_file="${log_dir}/sft_${model_tag}__train_${train_run_tag}__${dataset_tag}.log"
      else
        adapter_dir="saves/task3/${model_tag}${group_suffix}/lora/sft"
        if [[ $multi_dataset -eq 1 ]]; then
          save_file="${gen_dir}/sft_${model_tag}__${dataset_tag}.jsonl"
          log_file="${log_dir}/sft_${model_tag}__${dataset_tag}.log"
        else
          save_file="${gen_dir}/sft_${model_tag}.jsonl"
          log_file="${log_dir}/sft_${model_tag}.log"
        fi
      fi

      if [[ ! -d "$adapter_dir" ]]; then
        if [[ -z "$train_run_tag" ]]; then
          alt_adapter_dir="saves/task3/${model_tag}${group_suffix}/${dataset_tag}/lora/sft"
          if [[ -d "$alt_adapter_dir" ]]; then
            adapter_dir="$alt_adapter_dir"
          else
            # No adapter: run base-only so something is written to generate/task3 (or task3_group)
            echo "Adapter not found ($adapter_dir), running base-only for ${model_name} (dataset=${dataset}) -> ${gen_dir}/nosft_${model_tag}__${dataset_tag}.jsonl"
            save_file="${gen_dir}/nosft_${model_tag}__${dataset_tag}.jsonl"
            log_file="${log_dir}/nosft_${model_tag}__${dataset_tag}.log"
            adapter_dir=""
          fi
        else
          echo "Adapter directory not found: $adapter_dir"
          echo "  -> Run first: TRAIN_DATASETS=${TRAIN_RUN} bash run/train/task3.sh"
          echo "  -> skipping SFT inference for ${model_name} (TRAIN_RUN=${TRAIN_RUN})"
          continue
        fi
      fi
    fi

    run_infer "$model_name" "$model_path" "$template" "$dataset" "$save_file" "$log_file" "$adapter_dir" || echo "Warning: inference failed for ${model_name} on ${dataset}, see $log_file"
  done

  # 2) Zero-shot inference (no adapter)；默认 TEST_ONLY_SFT=1 跳过，只跑本机已有的 SFT 模型
  if [[ "${TEST_ONLY_SFT:-1}" == "1" ]]; then
    echo "TEST_ONLY_SFT=1 (default): skipping zero-shot (only SFT models run). Set TEST_ONLY_SFT=0 to run zero-shot."
  else
  for model_name in "${nosft_models[@]}"; do
    template="$(get_template "$model_name")"
    model_tag="$(sanitize "$model_name")"
    model_path="$HF_MODEL_ROOT/$model_name"
    if [[ $multi_dataset -eq 1 ]]; then
      save_file="${gen_dir}/nosft_${model_tag}__${dataset_tag}.jsonl"
      log_file="${log_dir}/nosft_${model_tag}__${dataset_tag}.log"
    else
      save_file="${gen_dir}/nosft_${model_tag}.jsonl"
      log_file="${log_dir}/nosft_${model_tag}.log"
    fi

    run_infer "$model_name" "$model_path" "$template" "$dataset" "$save_file" "$log_file" || echo "Warning: inference failed for ${model_name} on ${dataset}, see $log_file"
  done
  fi
done

echo "All task3 inference tasks have been completed/submitted."
