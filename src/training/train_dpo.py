from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

SFT_ADAPTER_PATH = Path("models/sft_adapter")
OUTPUT_DIR = Path("models/dpo_qwen1.5b")
NUM_EXAMPLES = 2500
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
use_bf16 = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=f"train_prefs[:{NUM_EXAMPLES}]")

def is_valid_example(example):
    chosen = example.get("chosen")
    rejected = example.get("rejected")

    if not isinstance(chosen, list) or not isinstance(rejected, list):
        return False
    if len(chosen) < 2 or len(rejected) < 1:
        return False
    if chosen[-1].get("role") != "assistant":
        return False
    if rejected[-1].get("role") != "assistant":
        return False
    if chosen[:-1] != rejected[:-1]:
        return False
    if not chosen[-1].get("content") or not rejected[-1].get("content"):
        return False

    return True

raw_dataset = raw_dataset.filter(is_valid_example)

def format_example(example):
    return {
        "prompt": example["chosen"][:-1],
        "chosen": [example["chosen"][-1]],
        "rejected": [example["rejected"][-1]],
    }
dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

if len(dataset) == 0:
    raise RuntimeError("No valid preference examples remain after filtering.")

print(f"Training examples: {len(dataset)}")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

print("Loading quantized base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", quantization_config=quantization_config,
    torch_dtype=compute_dtype, device_map={"": 0}, attn_implementation="sdpa",)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training( base_model, use_gradient_checkpointing=True,)

print("Loading SFT adapter...")

model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER_PATH), is_trainable=True)
model.enable_input_require_grads()
model.config.use_cache = False

training_args = DPOConfig(output_dir=str(OUTPUT_DIR),per_device_train_batch_size=2,
    gradient_accumulation_steps=4, num_train_epochs=1, learning_rate=5e-7,
    lr_scheduler_type="linear", warmup_steps=10,
    weight_decay=0.0, max_grad_norm=1.0,
    bf16=use_bf16, fp16=not use_bf16,
    tf32=False,optim="paged_adamw_8bit", gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    beta=0.1, loss_type=["sigmoid"],
    max_length=2048, truncation_mode="keep_start",
    precompute_ref_log_probs=True, precompute_ref_batch_size=1,
    dataloader_num_workers=0, dataloader_pin_memory=False,
    dataset_num_proc=None, pad_to_multiple_of=8,
    remove_unused_columns=False,
    logging_strategy="steps", logging_steps=10,
    logging_first_step=True, report_to="none",
    save_strategy="epoch", save_total_limit=1,
    save_only_model=True, seed=42,
    data_seed=42, use_cache=False,
)

print("Initializing DPO trainer...")

trainer = DPOTrainer(
    model=model, ref_model=None, args=training_args, train_dataset=dataset, processing_class=tokenizer,
)

print("Starting DPO training...")
trainer.train()

print("Saving DPO adapter and tokenizer...")
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"Training complete. Output: {OUTPUT_DIR.resolve()}")