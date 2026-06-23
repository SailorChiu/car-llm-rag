"""车控指令 QLoRA 微调：Qwen2.5-1.5B + 4bit(nf4) + LoRA。

要点：
  - 4bit 量化基座（省显存），只训练 LoRA 适配器（参数量极小）。
  - label mask：prompt 部分 label 设 -100，只在「答案 JSON」上算 loss，
    让模型学"怎么答"而不是"复读题目"。
  - 显存友好：batch=2 + 梯度累积 + gradient_checkpointing + paged_adamw_8bit。
适配器存到 sft/out。
"""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,  # noqa: E402
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402

MAX_LEN = 256


class SFTDataset(Dataset):
    """每条样本：prompt(系统+用户) + target(答案JSON)。labels 只在 target 上有效。"""

    def __init__(self, path, tok):
        self.tok = tok
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = [{"role": "system", "content": config.SFT_SYSTEM},
                {"role": "user", "content": r["messages_user"]}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tok(r["target"], add_special_tokens=False)["input_ids"]
        target_ids = target_ids + [self.tok.eos_token_id]

        input_ids = (prompt_ids + target_ids)[:MAX_LEN]
        # prompt 部分用 -100 屏蔽，只在答案算 loss
        labels = ([-100] * len(prompt_ids) + target_ids)[:MAX_LEN]
        return {"input_ids": input_ids, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def main():
    tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print(f"加载 4bit 基座: {config.LLM_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    train_ds = SFTDataset(config.TRAIN_PATH, tok)
    print(f"训练样本: {len(train_ds)} 条")

    args = TrainingArguments(
        output_dir=os.path.join(config.SFT_OUT_DIR, "_trainer"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    os.makedirs(config.SFT_OUT_DIR, exist_ok=True)
    model.save_pretrained(config.SFT_OUT_DIR)
    tok.save_pretrained(config.SFT_OUT_DIR)
    print(f"\nLoRA 适配器已保存到 {config.SFT_OUT_DIR}")


if __name__ == "__main__":
    main()
