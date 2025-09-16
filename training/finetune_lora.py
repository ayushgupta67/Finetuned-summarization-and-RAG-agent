import argparse
import json
import os
from typing import Tuple

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# Base summarization model (small & fast)
BASE_MODEL = os.environ.get("BASE_MODEL", "sshleifer/distilbart-cnn-12-6")


def load_jsonl_dataset(path: str) -> Tuple[Dataset, Dataset]:
    """
    Expects JSONL with fields: {"split": "train"/"test", "text": "...", "summary": "..."}
    """
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    train = [r for r in rows if r.get("split") == "train"]
    test = [r for r in rows if r.get("split") == "test"]
    if not train:
        raise ValueError(f"No train examples found in {path}. Ensure 'split':'train' rows exist.")
    if not test:
        # Allow train-only; just eval on train as a fallback
        test = train[: max(1, len(train) // 5)]
    return Dataset.from_list(train), Dataset.from_list(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/sample_dataset.jsonl",
                    help="JSONL file with fields: split/text/summary")
    ap.add_argument("--output_dir", type=str, default="models/lora-distilbart")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_src_len", type=int, default=1024)
    ap.add_argument("--max_tgt_len", type=int, default=128)
    args = ap.parse_args()

    # Tokenizer & base model
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # LoRA config (correct projection names for DistilBART/BART family)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_cfg)

    # Data
    train_ds, test_ds = load_jsonl_dataset(args.data_path)

    def preprocess(ex):
        model_inputs = tok(ex["text"], truncation=True, max_length=args.max_src_len)
        with tok.as_target_tokenizer():
            labels = tok(ex["summary"], truncation=True, max_length=args.max_tgt_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    test_tok = test_ds.map(preprocess, remove_columns=test_ds.column_names)

    # Training
    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=False,  # safe default on CPU; enable if you use a CUDA GPU that supports fp16
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()

    # Save adapters + tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"✅ LoRA adapters saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
