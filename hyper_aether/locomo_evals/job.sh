#!/bin/bash

# Load python and install dependencies
module load python/3.10.4
pip install vllm openai nltk numpy

# Export your HuggingFace token so vLLM can download Llama 3
export HF_TOKEN=""

# 1. Start the vLLM server in the background
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype auto \
    --port 8000 &
    
VLLM_PID=$!

# 2. Wait for the server to spin up and load the model into VRAM
echo "Waiting 2 minutes for vLLM to start..."
sleep 120

# 3. Run your evaluation script
python3 judge_locomo_vllm.py --input_file results/locomo_results_gemini.json

# 4. Stop the server
kill $VLLM_PID