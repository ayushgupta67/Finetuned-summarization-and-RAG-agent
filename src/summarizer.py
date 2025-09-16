# src/summarizer.py
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# LoRA is optional; if not present, we just run base
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

BASE_MODEL = "sshleifer/distilbart-cnn-12-6"

class Summarizer:
    def __init__(self, lora_dir: Optional[str] = None):
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

        if lora_dir and PEFT_AVAILABLE:
            try:
                _ = PeftConfig.from_pretrained(lora_dir)
                self.model = PeftModel.from_pretrained(self.model, lora_dir)
                print(f"[Summarizer] Loaded LoRA from {lora_dir}")
            except Exception as e:
                print(f"[Summarizer] LoRA not loaded: {e}")

        self.model.to(torch.device("cpu"))
        self.model.eval()

    def summarize(self, text: str, max_new_tokens: int = 160) -> str:
    
        inputs = self.tok([text], max_length=1024, truncation=True, return_tensors="pt")
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,   # fixed cap
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        return self.tok.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
