# src/agent.py
from typing import Dict, Any, Optional, List
from src.utils import chunk_text
from src.summarizer import Summarizer

class SummarizationAgent:
    def __init__(self, lora_dir: Optional[str] = None):
        self.summarizer = Summarizer(lora_dir=lora_dir)

    def plan(self, text: str) -> Dict[str, Any]:
        
        strategy = "single" if len(text) < 1500 else "hierarchical"
        return {"strategy": strategy}

    def execute(self, text: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan using fixed-length summarization."""
        if plan["strategy"] == "single":
            summary = self.summarizer.summarize(text)  # defaults to ~160 new tokens
            return {"strategy": "single", "summary": summary, "chunks": 1}

        # hierarchical: summarize chunks, then combine and summarize again
        chunks: List[str] = chunk_text(text)
        # keep a modest cap for partials so final isn’t too long
        partials = [self.summarizer.summarize(c, max_new_tokens=120) for c in chunks]
        combined = "\n".join(partials)
        final = self.summarizer.summarize(combined, max_new_tokens=180)
        return {"strategy": "hierarchical", "summary": final, "chunks": len(chunks)}
