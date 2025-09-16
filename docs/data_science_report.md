# Data Science Report — LoRA Summarization

**Base model**: sshleifer/distilbart-cnn-12-6  
**Adapter**: LoRA (rank=8, alpha=16, dropout=0.05)

**Dataset**: `data/sample_dataset.jsonl` (tiny synthetic; replace with your own).  
**Tokenization**: 1024 input, 128 target.  
**Metrics**: ROUGE 1/2/L + compression.

Training example:
```
python training/finetune_lora.py --epochs 1 --batch_size 2 --lr 2e-4   --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --output_dir models/lora-distilbart
```
