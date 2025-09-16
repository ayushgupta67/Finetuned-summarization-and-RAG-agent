import os, json, re, uuid
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

def clean_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return clean_text("\n".join(texts))

def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) > max_chars and cur:
            chunks.append(" ".join(cur).strip())
            cur, cur_len = [s], len(s)
        else:
            cur.append(s); cur_len += len(s)
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks

class SimpleFAISSRetriever:
    def __init__(self, kb_path: str = "knowledge_base", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        os.makedirs(kb_path, exist_ok=True)
        self.kb_path = kb_path
        self.index_file = os.path.join(kb_path, "faiss.index")
        self.meta_file = os.path.join(kb_path, "meta.jsonl")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self._load()

    def _load(self):
        self.index = faiss.read_index(self.index_file)
        with open(self.meta_file, "r", encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f]

    def _save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    def add_texts(self, docs: List[Tuple[str, str]]):
        texts = [t for _, t in docs]
        if not texts: return 0
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._ensure_index(embs.shape[1])
        self.index.add(embs.astype("float32"))
        for (doc_id, t) in docs:
            self.meta.append({"doc_id": doc_id, "text": t})
        self._save()
        return len(texts)

    def ingest_path(self, path: str):
        added = 0
        if os.path.isdir(path):
            for fn in os.listdir(path):
                full = os.path.join(path, fn)
                if fn.lower().endswith(".pdf"):
                    txt = pdf_to_text(full)
                    for ch in chunk_text(txt):
                        added += self.add_texts([(str(uuid.uuid4()), ch)])
                elif fn.lower().endswith(".txt"):
                    txt = clean_text(open(full, "r", encoding="utf-8", errors="ignore").read())
                    for ch in chunk_text(txt):
                        added += self.add_texts([(str(uuid.uuid4()), ch)])
        else:
            if path.lower().endswith(".pdf"):
                txt = pdf_to_text(path)
                for ch in chunk_text(txt):
                    added += self.add_texts([(str(uuid.uuid4()), ch)])
            elif path.lower().endswith(".txt"):
                txt = clean_text(open(path, "r", encoding="utf-8", errors="ignore").read())
                for ch in chunk_text(txt):
                    added += self.add_texts([(str(uuid.uuid4()), ch)])
        return added

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or not self.meta:
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q, k)
        out = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.meta): continue
            m = self.meta[idx].copy()
            m["score"] = float(score)
            out.append(m)
        return out
