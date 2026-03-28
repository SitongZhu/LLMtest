conda create -n llmtest python=3.10 -y
conda activate llmtest
cd ~/LLMtest

export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
bash scripts/setup_env.sh

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e LLaMA-Factory/LLaMA-Factory
pip install -r requirements-llmtest.txt
