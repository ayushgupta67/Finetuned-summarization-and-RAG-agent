import argparse, json, os
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "sshleifer/distilbart-cnn-12-6")

def load_test(path="data/sample_dataset.jsonl"):
    rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    return [r for r in rows if r.get("split")=="test"]

def build_pipe(lora_dir=None):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    if lora_dir and os.path.isdir(lora_dir) and os.listdir(lora_dir):
        model = PeftModel.from_pretrained(model, lora_dir)
    return pipeline("summarization", model=model, tokenizer=tok)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["base","lora"], default="base")
    ap.add_argument("--lora_dir", type=str, default="models/lora-distilbart")
    args = ap.parse_args()

    lora_dir = args.lora_dir if args.model_type=="lora" else None
    pipe = build_pipe(lora_dir=lora_dir)
    data = load_test()

    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    sums = {"r1":0.0,"r2":0.0,"rL":0.0,"comp":0.0}
    for ex in data:
        pred = pipe(ex["text"], truncation=True, max_new_tokens=128)[0]["summary_text"]
        sc = scorer.score(ex["summary"], pred)
        sums["r1"] += sc["rouge1"].fmeasure
        sums["r2"] += sc["rouge2"].fmeasure
        sums["rL"] += sc["rougeL"].fmeasure
        sums["comp"] += max(1,len(pred))/len(ex["text"])

    n = max(1,len(data))
    print(f"ROUGE-1: {sums['r1']/n:.4f}  ROUGE-2: {sums['r2']/n:.4f}  ROUGE-L: {sums['rL']/n:.4f}")
    print(f"Compression Ratio: {sums['comp']/n:.4f}")

if __name__ == "__main__":
    main()
