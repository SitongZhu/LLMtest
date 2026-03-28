#!/usr/bin/env python3
"""
若 MODEL_ROOT 下缺少已知 SFT 基座，则从 Hugging Face 下载。
与 run/test/task3.sh 中 sft_models 路径一致（目录名 = mapping 第二项）。

退出码: 0 已存在或下载成功, 1 失败或未知 model-key, 2 缺少 huggingface_hub
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# (HuggingFace repo_id, 本地目录名，相对 MODEL_ROOT)
SFT_BASE_MAP: dict[str, tuple[str, str]] = {
    "Qwen2.5-7B-Instruct-GPTQ-Int4": ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "Qwen2.5-7B-Instruct-GPTQ-Int4"),
    "Meta-Llama-3.1-8B-Instruct": ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    "Mistral-7B-Instruct-v0.3": ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"),
    "DeepSeek-R1-Distill-Qwen-7B": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B"),
}


def main() -> int:
    p = argparse.ArgumentParser(description="Ensure SFT base model under MODEL_ROOT")
    p.add_argument("--model-key", required=True, help="e.g. Qwen2.5-7B-Instruct-GPTQ-Int4")
    p.add_argument("--model-root", type=Path, required=True, help="e.g. LLMtest/LLaMA-Factory/models")
    p.add_argument("--token", default=None, help="HF token (default: env HF_TOKEN)")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("需要 huggingface_hub：pip install huggingface_hub", file=sys.stderr)
        return 2

    key = args.model_key.strip()
    if key not in SFT_BASE_MAP:
        print(f"No Hugging Face mapping for model key: {key}", file=sys.stderr)
        return 1

    repo_id, dirname = SFT_BASE_MAP[key]
    model_root = args.model_root.resolve()
    dest = model_root / dirname
    token = args.token or os.environ.get("HF_TOKEN")

    if (dest / "config.json").is_file():
        print(f"Already present: {dest}")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {dest}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
        )
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    if not (dest / "config.json").is_file():
        print(f"Download finished but config.json missing under {dest}", file=sys.stderr)
        return 1

    print("Done:", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
