#!/usr/bin/env bash
# Compare: (1) train wave11 + test wave11  vs  (2) train wave10 + test wave11
# 1) Train on wave11train, test on wave11test
# 2) Train on wave10train, test on wave11test
# 3) Run result.py in generate/task3 to get Excel with two result rows

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "========== Experiment 1: Train wave11train, test wave11test =========="
TRAIN_DATASETS=wave11train_20260218_223831 bash train/task3.sh
TEST_DATASETS=wave11test_20260218_223831 TRAIN_RUN=wave11train_20260218_223831 bash test/task3.sh

echo ""
echo "========== Experiment 2: Train wave10train, test wave11test =========="
TRAIN_DATASETS=wave10train_20260218_223831 bash train/task3.sh
TEST_DATASETS=wave11test_20260218_223831 TRAIN_RUN=wave10train_20260218_223831 bash test/task3.sh

echo ""
echo "========== Base-only (no adapter) on same test set =========="
echo "Run pure model on same test dataset as above (wave11test):"
TEST_DATASETS=wave11test_20260218_223831 USE_BASE_ONLY=1 bash test/task3.sh
echo "Output: generate/task3/nosft_*__wave11test_20260218_223831.jsonl"

echo ""
echo "========== Summary: run result.py to generate Excel =========="
GEN_DIR="$(cd ".." && pwd)/generate/task3"
if [[ -f "$GEN_DIR/result.py" ]]; then
  (cd "$GEN_DIR" && python result.py)
  echo "Results: $GEN_DIR/all_files_prediction_analysis_*.xlsx"
else
  echo "result.py not found; run manually: cd generate/task3 && python result.py"
fi

echo ""
echo "Done. Compare sft_*__train_wave11* vs sft_*__train_wave10* vs nosft_* in generate/task3."
echo "To run step by step:"
echo "  TRAIN_DATASETS=wave11train_20260218_223831 bash run/train/task3.sh"
echo "  TEST_DATASETS=wave11test_20260218_223831 TRAIN_RUN=wave11train_20260218_223831 bash run/test/task3.sh"
echo "  TRAIN_DATASETS=wave10train_20260218_223831 bash run/train/task3.sh"
echo "  TEST_DATASETS=wave11test_20260218_223831 TRAIN_RUN=wave10train_20260218_223831 bash run/test/task3.sh"
echo "  TEST_DATASETS=wave11test_20260218_223831 USE_BASE_ONLY=1 bash run/test/task3.sh   # same test set, no adapter"
echo "  cd LLaMA-Factory/generate/task3 && python result.py"
