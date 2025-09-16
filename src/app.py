import streamlit as st
import tempfile, os
from src.utils import pdf_to_text
from src.agent import SummarizationAgent
from src.rag_agent import RAGAgent


st.set_page_config(page_title="Summarization and RAG Agent", page_icon="🧠", layout="wide")
st.title("🧠 AI Summarization and RAG Agent")
st.caption("DistilBART base with optional LoRA adapters")

with st.sidebar:
    st.header("Settings")
    use_lora = st.toggle("Use LoRA adapter (models/lora-distilbart)", True)
    target_words = st.slider("Target summary length (words)", 10, 20, 50, 60)

def build_agent():
    lora_dir = "models/lora-distilbart" if use_lora else None
    if use_lora and not (os.path.isdir(lora_dir) and os.listdir(lora_dir)):
        st.info("LoRA directory not found or empty; using base model.")
        lora_dir = None
    return SummarizationAgent(lora_dir=lora_dir)

tab_text, tab_pdf, tab_rag = st.tabs(["📝 Paragraph", "📄 PDF", "🔎 Ask RAG"])

with tab_text:
    text_input = st.text_area("Input text", height=220, placeholder="Paste long text here...")
    if st.button("Summarize Text", type="primary"):
        if not text_input.strip():
            st.warning("Please paste some text.")
        else:
            agent = build_agent()
            plan = agent.plan(text_input)
            with st.spinner(f"Running plan: {plan['strategy']}"):
                out = agent.execute(text_input, plan)
            st.success("Done!")
            st.write("**Strategy:**", out.get("strategy"))
            if "chunks" in out: st.write("**#Chunks:**", out["chunks"])
            st.text_area("Summary", out["summary"], height=220)

with tab_pdf:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Summarize PDF"):
        if not pdf_file:
            st.warning("Please upload a PDF.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name
            raw = pdf_to_text(tmp_path)
            os.unlink(tmp_path)
            if not raw.strip():
                st.error("No text extracted from PDF.")
            else:
                agent = build_agent()
                plan = agent.plan(raw)
                with st.spinner(f"Running plan: {plan['strategy']}"):
                    out = agent.execute(raw, plan)
                st.success("Done!")
                st.write("**Strategy:**", out.get("strategy"))
                if "chunks" in out: st.write("**#Chunks:**", out["chunks"])
                st.text_area("Summary", out["summary"], height=240)


with tab_rag:
    st.subheader("Ask a question with Retrieval-Augmented Generation (RAG)")
    st.caption("Build a small knowledge base by ingesting PDFs/TXT, then query it.")
    kb_path = st.text_input("Knowledge Base folder/file to ingest", value="knowledge_base")
    uploaded = st.file_uploader("Optionally upload documents to ingest", type=["pdf","txt"], accept_multiple_files=True)
    if st.button("Ingest to KB"):
        import os
        os.makedirs("knowledge_base", exist_ok=True)
        if uploaded:
            for uf in uploaded:
                dst = os.path.join("knowledge_base", uf.name)
                with open(dst, "wb") as f: f.write(uf.read())
        rag = RAGAgent(lora_dir=("models/lora-distilbart" if use_lora else None))
        n = rag.ingest(kb_path)
        st.success(f"Ingested {n} chunks into FAISS index.")
    question = st.text_input("Your question")
    topk = st.slider("Top-K retrieval", 1, 10, 5)
    if st.button("Ask RAG"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            rag = RAGAgent(lora_dir=("models/lora-distilbart" if use_lora else None))
            with st.spinner("Retrieving and generating..."):
                out = rag.query(question, k=topk)
            st.success("Done!")
            st.markdown("**Answer:**")
            st.text_area("RAG Answer", out["answer"], height=220)
            with st.expander("View retrieved chunks"):
                for i, h in enumerate(out["hits"], 1):
                    st.markdown(f"**{i}. score={h.get('score',0):.3f}**\n\n{h['text']}")
