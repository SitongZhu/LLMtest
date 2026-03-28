#!/usr/bin/env python3
"""
扫描 task3 目录下所有 .json 文件，同步到 dataset_info.json，并可选输出 dataset 名称列表。
用法:
  python sync_task3_datasets.py --task3_dir /path/to/task3 --dataset_info /path/to/dataset_info.json
  python sync_task3_datasets.py --print-names-only --task3_dir /path/to/task3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _default_paths() -> tuple[Path, Path, Path]:
    """本脚本位于 LLMtest/LLaMA-Factory/run/test/ → 解析出 task3、dataset_info、LLaMA-Factory 根目录。"""
    run_test = Path(__file__).resolve().parent
    llama_factory_root = run_test.parent.parent
    llmtest_root = llama_factory_root.parent
    task3_dir = llmtest_root / "data" / "enitre_pipeline" / "task3"
    dataset_info = llama_factory_root / "data" / "dataset_info.json"
    return task3_dir, dataset_info, llama_factory_root


def main() -> None:
    default_task3, default_dataset_info, llama_factory_root = _default_paths()
    parser = argparse.ArgumentParser(description="Sync task3/*.json into dataset_info.json")
    parser.add_argument(
        "--task3_dir",
        type=str,
        default=str(default_task3),
        help="Directory containing task3 JSON files (default: <LLMtest>/data/enitre_pipeline/task3)",
    )
    parser.add_argument(
        "--dataset_info",
        type=str,
        default=str(default_dataset_info),
        help="Path to dataset_info.json (default: <LLMtest>/LLaMA-Factory/data/dataset_info.json)",
    )
    parser.add_argument(
        "--include_group",
        action="store_true",
        help="Also include JSON files from task3/group/ subdirectory",
    )
    parser.add_argument(
        "--print-names-only",
        action="store_true",
        help="Only print dataset names (one per line), do not modify dataset_info.json",
    )
    args = parser.parse_args()

    task3_dir = Path(args.task3_dir).resolve()
    if not task3_dir.is_dir():
        raise SystemExit(f"Task3 dir not found: {task3_dir}")

    dataset_info_path = Path(args.dataset_info).resolve()
    if not dataset_info_path.is_file() and not args.print_names_only:
        raise SystemExit(f"dataset_info.json not found: {dataset_info_path}")

    # 收集 task3 下所有 .json（仅顶层）
    names_to_path: dict[str, Path] = {}
    for f in sorted(task3_dir.glob("*.json")):
        name = f.stem
        names_to_path[name] = f.resolve()

    if args.include_group:
        group_dir = task3_dir / "group"
        if group_dir.is_dir():
            for f in sorted(group_dir.glob("*.json")):
                name = f.stem
                names_to_path[name] = f.resolve()

    if not names_to_path:
        if args.print_names_only:
            return
        raise SystemExit(f"No JSON files found in {task3_dir}")

    names_sorted = sorted(names_to_path.keys())

    if args.print_names_only:
        for n in names_sorted:
            print(n)
        return

    # 读取现有 dataset_info.json
    if dataset_info_path.is_file():
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    # 更新 task3 相关条目：路径相对 LLaMA-Factory 根目录（可用 .. 指向同级的 data/，便于整包迁移）
    lf_root = os.path.abspath(llama_factory_root)
    for name in names_sorted:
        path = names_to_path[name]
        file_name = os.path.relpath(os.path.abspath(path), start=lf_root).replace("\\", "/")
        info[name] = {"file_name": file_name}

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Synced {len(names_sorted)} task3 datasets to {dataset_info_path}")
    for n in names_sorted:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
