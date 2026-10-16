# LLM Alignment Training Pipeline

Training scripts for SFT and DPO fine-tuning.

## setup

```bash
pip install -r requirements.txt
```

## Training

### 1) Supervised fine tuning (SFT) Training

```bash
python src/training/train_sft.py
```

### 2) Direct Preference Optimization (DPO) Training

```bash
python src/training/train_dpo.py
```

## Project Structure

```
LLM-Alignment-Training-Pipeline/
├── src/
│ └── training/
│ ├── train_sft.py # SFT training script
│ └── train_dpo.py # DPO training script
├── models/ # Model checkpoints (gitignored)
├── requirements.txt # Python dependencies
└── README.md

