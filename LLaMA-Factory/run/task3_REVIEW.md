# Task3 pipeline review: pretrained → finetuning → test

## 1. Is this finetuning on a pretrained model?

**Yes.** The pipeline is:

1. **Base model**: A pretrained model (e.g. `DeepSeek-R1-Distill-Qwen-7B`, `Qwen2.5-7B-Instruct-GPTQ-Int4`) is loaded from disk (`model_name_or_path`).
2. **Training** (`run/train/task3.sh`): LoRA-SFT (supervised finetuning with LoRA) is run on **labeled data** (e.g. `train_attitude_group`, or wave-specific `wave11train_20260218_223831`). The script uses `llamafactory-cli train` with:
   - `model_name_or_path` = base model path  
   - `dataset` = your labeled dataset  
   - Output = **LoRA adapter** only (e.g. `saves/task3/DeepSeek-R1-Distill-Qwen-7B_group/lora/sft`).
3. **Testing** (`run/test/task3.sh`): For “SFT models”, inference loads **base model + adapter** (`--model_name_or_path` and `--adapter_name_or_path`). So you are testing the **finetuned** model (base + LoRA), not the raw pretrained model.

So: **pretrained base → finetune with your labels (LoRA-SFT) → test with base + adapter.**

---

## 2. Can you test with the original (base) model only?

**Yes.** You have two ways:

### Option A: Use `USE_BASE_ONLY=1` with the **same test dataset** as training

To run the **pure (unfinetuned) model** on the **same test dataset** you use for finetuned evaluation, set both `TEST_DATASETS` and `USE_BASE_ONLY=1`:

```bash
cd /path/to/LLaMA-Factory/run

# Same test set as your training experiment (e.g. wave11test)
TEST_DATASETS=wave11test_20260218_223831 USE_BASE_ONLY=1 bash test/task3.sh
```

- Output: `generate/task3/nosft_<model_tag>__wave11test_20260218_223831.jsonl`
- Then run `python result.py` in `generate/task3` to include base-only results in the same Excel.

Other examples:

```bash
# Default group test set (task3_group)
TEST_DATASETS=test_attitude_group USE_BASE_ONLY=1 bash test/task3.sh

# No env: default is test_attitude_group
USE_BASE_ONLY=1 bash test/task3.sh
```

- Inference runs with **no adapter** for every model in `sft_models`.
- Outputs are always `nosft_<model_tag>__<dataset_tag>.jsonl`, so you can compare:
  - `sft_*.jsonl` = finetuned (base + LoRA)
  - `nosft_*.jsonl` = original base only

### Option B: Zero-shot list (`nosft_models`)

The script already has a **zero-shot** list `nosft_models` (no adapter). Those models are run with the base model only. They use `model_path="$HF_MODEL_ROOT/$model_name"`. To test another base model (e.g. a local path) as zero-shot, add it to `nosft_models` and set `HF_MODEL_ROOT` so that `$HF_MODEL_ROOT/$model_name` points to the correct directory, or extend the script to support a separate list with explicit paths.

---

## 3. Summary

| Question | Answer |
|----------|--------|
| Finetuning on pretrained model? | **Yes.** LoRA-SFT on labeled data; adapter saved under `saves/task3/...`. |
| Using labeled data for training? | **Yes.** Dataset comes from `dataset_info.json` (e.g. `train_attitude_group`, wave datasets). |
| Test with original model only? | **Yes.** Use `USE_BASE_ONLY=1 bash run/test/task3.sh` for base-only inference of your SFT models, or use models in `nosft_models` (zero-shot). |
