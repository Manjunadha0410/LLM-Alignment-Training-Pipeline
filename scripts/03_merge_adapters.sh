#!/usr/bin/env bash
set -e
echo "Step 3:Merging DPO Weights into Base Model"
mkdir -p ./models/merged_target_model
python -m src.utils.merge_weights
echo "Model weights successfully merged"