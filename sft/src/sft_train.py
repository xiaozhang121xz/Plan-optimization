"""SFT training pipeline extracted from FSFT.ipynb."""

from __future__ import annotations

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load base model/tokenizer with optional 4-bit quantization."""
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": torch.cuda.current_device()},
            quantization_config=bnb_config,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": torch.cuda.current_device()},
            dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    return model, tokenizer


def build_peft_model(model):
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "v_proj", "q_proj"],
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    return model, peft_config


def load_and_format_dataset(dataset_path: str, tokenizer):
    def format_prompt(example):
        chat = [
            {"role": "system", "content": "?????????????"},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        return {"text": prompt}

    dataset_dict = load_dataset(dataset_path)
    train_dataset = dataset_dict["train"].map(format_prompt)
    return train_dataset


def run_sft_training() -> None:
    model_name = os.getenv("SFT_MODEL_NAME", "")
    dataset_path = os.getenv("SFT_DATASET_PATH", "")
    output_dir = os.getenv("SFT_OUTPUT_DIR", "")
    save_dir = os.getenv("SFT_SAVE_DIR", "")

    model, tokenizer = load_model_and_tokenizer(model_name=model_name, use_4bit=True)
    model, peft_config = build_peft_model(model)
    train_dataset = load_and_format_dataset(dataset_path=dataset_path, tokenizer=tokenizer)

    training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_length=4096,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_arguments,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.model.save_pretrained(save_dir)


if __name__ == "__main__":
    run_sft_training()
