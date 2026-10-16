import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import types
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import trl.trainer.sft_trainer
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
USE_BF16 = False
USE_FP16 = True
USE_4BIT = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if hasattr(trl.trainer.sft_trainer,"_patch_chunked_ce_lm_head"):
    _orig_patch_chunked_ce = (trl.trainer.sft_trainer._patch_chunked_ce_lm_head)
    def _safe_patch_chunked_ce(target, *args, **kwargs):
        def ensure_func_attribute(module):
            if module is None or not hasattr(module, "forward"):
                return
            fw = module.forward
            if hasattr(fw, "__func__"):
                return
            def fw_wrapper(self, *a, **kw):
                return fw(*a, **kw)
            module.forward = types.MethodType(
                fw_wrapper,
                module
            )
        ensure_func_attribute(target)

        if hasattr(target, "modules"):
            for module in target.modules():
                ensure_func_attribute(module)

        return _orig_patch_chunked_ce(
            target,
            *args,
            **kwargs
        )

    trl.trainer.sft_trainer._patch_chunked_ce_lm_head = (
        _safe_patch_chunked_ce
    )
def train():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True,)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True,)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,quantization_config=bnb_config,torch_dtype=torch.float16,
            device_map={ "": LOCAL_RANK},attn_implementation="sdpa",trust_remote_code=True,low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,torch_dtype=torch.float16,device_map={"": LOCAL_RANK},attn_implementation="sdpa",
            trust_remote_code=True,low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
    peft_config = LoraConfig(r=16,lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",],
        lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",)
    raw_dataset = load_dataset(dataset_name,split="train_sft[:4000]")
    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(example["messages"],tokenize=False,add_generation_prompt=False,)}
    dataset = raw_dataset.map(format_chat,remove_columns=raw_dataset.column_names,desc="Formatting chat dataset",)
    torch.cuda.empty_cache()
    training_args = SFTConfig(
        output_dir="./models/sft_adapter", dataset_text_field="text", max_length=2048,
        per_device_train_batch_size=4, gradient_accumulation_steps=2,
        learning_rate=2e-4, num_train_epochs=1, fp16=True,
        bf16=False, gradient_checkpointing_kwargs={"use_reentrant": False}, packing=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True, logging_steps=10, logging_first_step=True,
        save_strategy="epoch", save_total_limit=2, ddp_find_unused_parameters=False, remove_unused_columns=False,
        report_to="none",
        seed=42, data_seed=42,
    )
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset,
        peft_config=peft_config, processing_class=tokenizer,
    )
    trainer.train()
    if RANK == 0:
        print("\nSaving adapter...")
        trainer.save_model(
            "./models/sft_adapter"
        )
        tokenizer.save_pretrained(
            "./models/sft_adapter"
        )
        print("\nTraining complete.")
        print(
            "Adapter saved to: ./models/sft_adapter"
        )
if __name__ == "__main__":
    train()
