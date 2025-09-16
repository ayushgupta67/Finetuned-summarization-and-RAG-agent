from typing import Dict, Any, Optional
from src.summarizer import Summarizer
from src.retriever import SimpleFAISSRetriever

PROMPT_TEMPLATE = """Answer the question using only the context.

Context:
{context}

Question:
{question}

Answer:
"""

class RAGAgent:
    def __init__(self, lora_dir: Optional[str] = None, kb_path: str = "knowledge_base"):
        self.summarizer = Summarizer(lora_dir=lora_dir)
        self.retriever = SimpleFAISSRetriever(kb_path)

    def ingest(self, path: str) -> int:
        return self.retriever.ingest_path(path)

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        hits = self.retriever.search(question, k=k)
        context = "\n\n".join([h["text"] for h in hits])
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.summarizer.summarize(prompt, max_new_tokens=180)
        return {"answer": answer, "hits": hits}
