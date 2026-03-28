#!/usr/bin/env python3
"""Download a Hugging Face repo snapshot into a local directory (no symlinks by default)."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    raise SystemExit("需要 huggingface_hub：pip install huggingface_hub") from e


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="e.g. Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    p.add_argument("--local-dir", required=True, type=Path, help="target directory")
    p.add_argument("--token", default=None, help="HF token (optional; or huggingface-cli login)")
    args = p.parse_args()

    dest = args.local_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "config.json").is_file():
        print(f"Already present (config.json exists): {dest}")
        return

    print(f"Downloading {args.repo} -> {dest}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=args.token,
    )
    print("Done:", dest)


if __name__ == "__main__":
    main()
