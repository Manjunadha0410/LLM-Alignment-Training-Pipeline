#!/usr/bin/env bash
set -e
echo "Step 1:Running Supervised Fine-Tuning(SFT)"
mkdir -p ./models/sft_adapter
python -m src.training.train_sft
echo "SFT training complete"