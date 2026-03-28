conda create -n llmtest python=3.10 -y
conda activate llmtest
cd ~/LLMtest

# A100 / 常见 CUDA 12.x（按你集群实际改 cu118/cu121/cu126）
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
bash scripts/setup_env.sh
