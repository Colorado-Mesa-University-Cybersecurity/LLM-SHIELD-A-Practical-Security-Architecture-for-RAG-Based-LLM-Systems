#!/usr/bin/env python3
"""Fine-tune Qwen2-0.5B on tiny-codes (resumes from latest checkpoint automatically)."""
import os
import glob
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA = os.path.join(EXPERIMENTS_DIR, "tokenized_tiny_codes_qwen2.5_0.5b_train")
VAL_DATA = os.path.join(EXPERIMENTS_DIR, "tokenized_tiny_codes_qwen2.5_0.5b_val")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")
BASE_MODEL = "Qwen/Qwen2-0.5B"


def latest_checkpoint(results_dir: str) -> str | None:
    checkpoints = glob.glob(os.path.join(results_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.rsplit("-", 1)[-1]))


def main():
    print("Loading datasets...")
    train_dataset = load_from_disk(TRAIN_DATA)
    val_dataset = load_from_disk(VAL_DATA) if os.path.exists(VAL_DATA) else None
    print(f"  train: {len(train_dataset):,} examples")
    if val_dataset:
        print(f"  val:   {len(val_dataset):,} examples")

    resume = latest_checkpoint(RESULTS_DIR)
    print(f"Loading model from: {resume or BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"  parameters: {model.num_parameters():,}")

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        save_steps=500,
        logging_steps=100,
        lr_scheduler_type="linear",
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        seed=42,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        report_to=[],
        average_tokens_across_devices=True,
        save_total_limit=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    print(f"Resuming from checkpoint: {resume}" if resume else "Starting fresh")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = os.path.join(RESULTS_DIR, "final")
    print(f"Saving final model to {final_dir}")
    trainer.save_model(final_dir)


if __name__ == "__main__":
    main()
