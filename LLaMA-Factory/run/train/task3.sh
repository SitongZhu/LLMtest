#!/usr/bin/env bash
set -euo pipefail

# Change to LLaMA-Factory directory to ensure relative paths work correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$LLAMA_FACTORY_DIR" || exit 1

MODEL_ROOT="${MODEL_ROOT:-$LLAMA_FACTORY_DIR/models}"

# Add LLaMA-Factory src to PYTHONPATH so llamafactory module can be found
if [[ -d "$LLAMA_FACTORY_DIR/LLaMA-Factory/src" ]]; then
    export PYTHONPATH="$LLAMA_FACTORY_DIR/LLaMA-Factory/src${PYTHONPATH:+:$PYTHONPATH}"
elif [[ -d "$LLAMA_FACTORY_DIR/src" ]]; then
    export PYTHONPATH="$LLAMA_FACTORY_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
fi

export CUDA_VISIBLE_DEVICES="0,1"
export DISABLE_VERSION_CHECK=1

# Create necessary directories
mkdir -p saves/task3
mkdir -p log/task3

# Define common training parameters
TRAIN_SCRIPT="examples/train_lora/llama3_lora_sft.yaml"
MAX_SAMPLES="200000"
# For 7B models on single GPU, use smaller batch size and more gradient accumulation
PER_DEVICE_BATCH_SIZE="2"
GRADIENT_ACCUMULATION_STEPS="32"
NUM_TRAIN_EPOCHS="3"
SAVE_STEPS="1000"

# Multi-wave / multi-dataset support:
# - Put each wave as a separate dataset key in data/dataset_info.json
# - Then pass multiple dataset keys via:
#     TRAIN_DATASETS="train_attitude_group,train_attitude_group_wave_1994_1998,train_attitude_group_wave_2010_2014"
#   (comma-separated)
parse_datasets() {
    local raw="${1:-}"
    local -a out=()
    if [[ -z "$raw" ]]; then
        echo ""
        return 0
    fi
    IFS=',' read -r -a out <<<"$raw"
    for i in "${!out[@]}"; do
        # trim spaces
        out[$i]="${out[$i]//[[:space:]]/}"
    done
    echo "${out[*]}"
}

sanitize() {
    local s="$1"
    s="${s//\//_}"
    s="${s//./_}"
    s="${s// /_}"
    echo "$s"
}

# Function to check if dataset exists
check_dataset_exists() {
    local dataset_name=$1
    local dataset_info_file="$LLAMA_FACTORY_DIR/data/dataset_info.json"
    
    if [[ ! -f "$dataset_info_file" ]]; then
        return 1
    fi
    
    # Check if dataset is defined in dataset_info.json
    if python3 -c "import json; d=json.load(open('$dataset_info_file')); exit(0 if '$dataset_name' in d else 1)" 2>/dev/null; then
        # Get the file path from dataset_info.json
        local file_path=$(python3 -c "import json; d=json.load(open('$dataset_info_file')); print(d.get('$dataset_name', {}).get('file_name', ''))" 2>/dev/null)
        
        # Resolve relative paths（相对 LLaMA-Factory 根目录，如 ../data/、../../data/）
        if [[ "$file_path" == ../../* ]]; then
            file_path="$LLAMA_FACTORY_DIR/../../${file_path#../../}"
        elif [[ "$file_path" == ../* ]]; then
            file_path="$LLAMA_FACTORY_DIR/$file_path"
        elif [[ "$file_path" != /* ]]; then
            file_path="$LLAMA_FACTORY_DIR/$file_path"
        fi
        
        # Check if the actual file exists
        if [[ -f "$file_path" ]]; then
            return 0
        fi
    fi
    
    return 1
}

# Define a function for training
train_model() {
    local model_name=$1       # e.g., "Meta-Llama-3.1-8B-Instruct"
    local model_path=$2       # e.g., "$MODEL_ROOT/Llama-3.1-8B"
    local dataset=$3
    local output_dir=$4
    local log_file=$5

    # Check if dataset exists
    if ! check_dataset_exists "$dataset"; then
        echo "Dataset '$dataset' not found, skipping training for $model_name"
        return 0
    fi

    # Select template based on model name
    local template
    if [[ "$model_name" == *"Qwen"* ]]; then
        template="qwen"
    elif [[ "$model_name" == *"Meta-Llama"* ]]; then
        template="llama3"
    elif [[ "$model_name" == *"Mistral"* ]]; then
        template="mistral"
    elif [[ "$model_name" == *"DeepSeek-R1"* ]]; then
        template="deepseekr1"
    else
        template="qwen"
    fi

    # Check if base adapter exists, only use it if it does
    local base_adapter_path="saves/base/${model_name//./_}/lora/sft"
    
    # Adjust batch size and gradient accumulation for large models on single GPU
    local per_device_batch_size=$PER_DEVICE_BATCH_SIZE
    local gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS
    
    if [[ "$model_name" == *"Llama"* ]] && [[ "$model_name" == *"8B"* ]]; then
        # For Llama-3.1-8B model on single GPU, use small batch size
        per_device_batch_size="1"
        gradient_accumulation="64"
        echo "Using optimized settings for Llama-3.1-8B model: batch_size=1, gradient_accumulation=64"
    elif [[ "$model_name" == *"GPTQ"* ]]; then
        # GPTQ quantized models use less memory, can use larger batch size
        per_device_batch_size="4"
        gradient_accumulation="16"
        echo "Using optimized settings for GPTQ quantized model: batch_size=4, gradient_accumulation=16"
    elif [[ "$model_name" == *"DeepSeek-R1"* ]] && [[ "$model_name" == *"7B"* ]]; then
        # For 7B model on single GPU, use very small batch size
        per_device_batch_size="1"
        gradient_accumulation="64"
        echo "Using optimized settings for 7B model: batch_size=1, gradient_accumulation=64"
    elif [[ "$model_name" == *"DeepSeek-R1"* ]] && [[ "$model_name" == *"14B"* ]]; then
        # For 14B model, use even smaller settings
        per_device_batch_size="1"
        gradient_accumulation="128"
        echo "Using optimized settings for 14B model: batch_size=1, gradient_accumulation=128"
    fi
    
    local train_cmd_args=(
        "model_name_or_path=$model_path"
        "dataset=$dataset"
        "template=$template"
        "max_samples=$MAX_SAMPLES"
        "per_device_train_batch_size=$per_device_batch_size"
        "gradient_accumulation_steps=$gradient_accumulation"
        "num_train_epochs=$NUM_TRAIN_EPOCHS"
        "save_steps=$SAVE_STEPS"
        "output_dir=$output_dir"
    )
    
    if [[ -d "$base_adapter_path" ]] && [[ -f "$base_adapter_path/adapter_config.json" ]]; then
        train_cmd_args+=("adapter_name_or_path=$base_adapter_path")
        echo "Found existing adapter at $base_adapter_path, will resume training"
    else
        echo "No existing adapter found, starting new training"
    fi

    echo "$model_name LoRA-SFT training started..."
    #nohup llamafactory-cli train "$TRAIN_SCRIPT" "${train_cmd_args[@]}" > "$log_file" 2>&1
    #echo "$model_name LoRA-SFT training finished"
    echo "Log file: $log_file"
    llamafactory-cli train "$TRAIN_SCRIPT" "${train_cmd_args[@]}" > "$log_file" 2>&1
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "$model_name LoRA-SFT training finished successfully"
    else
        echo "$model_name LoRA-SFT training failed with exit code $exit_code"
        return $exit_code
    fi
}

# Define models and datasets in an associative array (keys are model names)
declare -A models
#models["Qwen2.5-0.5B-Instruct"]="$MODEL_ROOT/Qwen2.5-0.5B-Instruct"
#models["DeepSeek-R1-Distill-Qwen-7B"]="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-7B"

models["Qwen2.5-7B-Instruct-GPTQ-Int4"]="$MODEL_ROOT/Qwen2.5-7B-Instruct-GPTQ-Int4"
#models["Meta-Llama-3.1-8B-Instruct"]="$MODEL_ROOT/Llama-3.1-8B"
#models["Qwen2.5-7B-Instruct"]="$MODEL_ROOT/Qwen2.5-7B-Instruct"
#models["Mistral-7B-Instruct-v0.3"]="$MODEL_ROOT/Mistral-7B-Instruct-v0.3"
#models["DeepSeek-R1-Distill-Qwen-14B"]="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-14B"

# Datasets to train (default keeps original behavior)
raw_datasets="${TRAIN_DATASETS:-${DATASETS:-train_attitude_group}}"
read -r -a datasets <<<"$(parse_datasets "$raw_datasets")"
if [[ ${#datasets[@]} -eq 0 ]]; then
    datasets=("train_attitude_group")
fi
echo "Datasets to train: ${datasets[*]}"

# Execute training for each model and each dataset
for model_name in "${!models[@]}"; do
    model_path="${models[$model_name]}"
    model_tag="$(sanitize "$model_name")"

    for dataset in "${datasets[@]}"; do
        # Keep legacy dir layout for the canonical two datasets
        # so existing scripts keep working:
        #  - train_attitude -> saves/task3/${model_tag}/lora/sft
        #  - train_attitude_group -> saves/task3/${model_tag}_group/lora/sft
        if [[ "$dataset" == "train_attitude" ]]; then
            output_dir="saves/task3/${model_tag}/lora/sft"
            log_file="log/task3/${model_tag}.log"
        elif [[ "$dataset" == "train_attitude_group" ]]; then
            output_dir="saves/task3/${model_tag}_group/lora/sft"
            log_file="log/task3/${model_tag}_group.log"
        else
            # For wave-specific datasets, avoid collisions by nesting under dataset tag.
            dataset_tag="$(sanitize "$dataset")"
            # Heuristic: group_suffix follows dataset name
            group_suffix=""
            if [[ "$dataset" == *"_group"* ]]; then
                group_suffix="_group"
            fi
            output_dir="saves/task3/${model_tag}${group_suffix}/${dataset_tag}/lora/sft"
            log_file="log/task3/${model_tag}${group_suffix}__${dataset_tag}.log"
        fi

        train_model "$model_name" "$model_path" "$dataset" "$output_dir" "$log_file"
    done
done
