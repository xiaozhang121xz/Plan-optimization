"""SFT inference script extracted from FSFT.ipynb."""

from __future__ import annotations

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """ 
"""


def load_finetuned_model():
    base_model_path = os.getenv("BASE_MODEL_PATH", "/home/pc/matrad/zhangxue/models/Qwen3-0.6B")
    lora_model_path = os.getenv("LORA_MODEL_PATH", "/home/pc/matrad/zhangxue/models/SFT/Qwen3-0.6B")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )

    finetuned_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto",
    )
    return finetuned_model, tokenizer


def answer_question(finetuned_model, tokenizer, user_question: str) -> str:
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to("cuda")

    with torch.no_grad():
        outputs = finetuned_model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            min_new_tokens=100,
            suppress_tokens=None,
        )

    input_ids = inputs["input_ids"]
    generated_tokens = outputs[0][len(input_ids[0]):]
    assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return assistant_response


def run_demo_inference() -> None:
    finetuned_model, tokenizer = load_finetuned_model()

    test_question = """
"""

    response = answer_question(finetuned_model, tokenizer, test_question)
    print("===== 临床报告 =====")
    print(f"{response}")


if __name__ == "__main__":
    run_demo_inference()
