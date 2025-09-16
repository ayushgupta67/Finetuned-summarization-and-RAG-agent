import argparse
from src.rag_agent import RAGAgent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="File or directory with PDFs/TXT to ingest")
    args = ap.parse_args()

    agent = RAGAgent()
    n = agent.ingest(args.path)
    print(f"Ingested chunks: {n}")
