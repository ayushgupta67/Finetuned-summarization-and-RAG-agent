import re
from typing import List
from pypdf import PdfReader

def normalize_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return normalize_text("\n".join(texts))

def chunk_text(text: str, max_tokens: int = 700) -> List[str]:
    # Rough token proxy: ~1 token ≈ 4 chars (English). Chunk on sentences.
    if not text: return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, cur_len = [], [], 0
    limit_chars = max_tokens * 4
    for s in sentences:
        s_len = len(s)
        if cur_len + s_len > limit_chars and cur:
            chunks.append(" ".join(cur).strip())
            cur, cur_len = [s], s_len
        else:
            cur.append(s); cur_len += s_len
    if cur: chunks.append(" ".join(cur).strip())
    return chunks

def compression_ratio(src: str, summary: str) -> float:
    if not src: return 0.0
    return max(1, len(summary)) / len(src)
