#!/usr/bin/env bash
set -e
echo "Step 5:Benchmarking and Launching Dashboard"

echo "Running async TTFT and TPS benchmark tests..."
python -m evaluation.async_bench

echo "Launching Streamlit visual dashboard..."
streamlit run dashboard/app.py