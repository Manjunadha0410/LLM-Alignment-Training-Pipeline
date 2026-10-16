#!/usr/bin/env bash
set -e
echo "Step 4: Launching vLLM with Speculative Decoding"

TARGET_MODEL="./models/merged_target_model"
DRAFT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
NUM_SPEC_TOKENS=5
PORT=8000
HOST="0.0.0.0"

vllm serve "$TARGET_MODEL" 

--host "$HOST" 

--port "$PORT" 

--speculative-model "$DRAFT_MODEL" 

--num-speculative-tokens "$NUM_SPEC_TOKENS" 

--gpu-memory-utilization 0.75 

--dtype bfloat16