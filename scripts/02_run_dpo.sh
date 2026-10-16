#!/usr/bin/env bash
set -e
echo "Step 2:Running Direct Preference Optimization (DPO) Alignment"
mkdir -p ./models/dpo_adapter
python -m src.training.train_dpo
echo "DPO alignment complete"
