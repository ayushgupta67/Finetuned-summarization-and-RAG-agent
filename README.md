# 👋 Hi, I'm Ayush Gupta  

🎓 **B.Tech 3rd Year**  
🏫 **IIT Jodhpur**  
⚙️ **Mechanical Engineering**  

---
# 🧠 Summarization + RAG AI Agent

The project is  an **AI Summarization Agent with Retrieval-Augmented Generation (RAG)**. It uses a **local LLM (DistilBART)** for summarization,  **LoRA fine-tuned adapters** for improved style/reliability, and **FAISS-based retrieval** for dynamic document-grounded Q&A. Everything runs **locally** — no API key required.

---

## 📌 1. Need for the Project
- Traditional LLMs are large, expensive, and often require cloud APIs + API keys.  
- Students and researchers need **local, lightweight solutions** for:  
  - Summarizing essays, reports, and PDFs.  
  - Asking grounded questions from their own uploaded material.  
- This project solves the above by combining **summarization + RAG** in a single local agent, wrapped with a simple **Streamlit UI**.

---

## 📌 2. Why use Local LLM (DistilBART)?
- **What is DistilBART?**  
  - A distilled (lighter/faster) version of BART, trained for summarization.  
  - Model name: `sshleifer/distilbart-cnn-12-6`.  
  - Designed to generate concise summaries of long text.  

- **Why not an API-based LLM?**  
  - Cloud models (e.g., GPT, Claude) require **API keys** and internet access.  
  - They have privacy concerns (data leaves your machine).  
  - They are resource-intensive and costly for frequent summarization.  

- **Why DistilBART here?**  
  - Lightweight → can run locally on CPU or small GPU.  
  - Pretrained for summarization → requires minimal adaptation.  
  - No API key → usable offline, secure, and cost-free.

---

## 📌 3. Why Fine-Tuning? (LoRA)

- **Problem:** Base DistilBART produces summaries in a **news style** (since it was trained on CNN/DailyMail).  
- **Need:** Adapt to **academic/essay style** for student use cases.  

### What is LoRA?
- **Low-Rank Adaptation (LoRA):** A parameter-efficient fine-tuning method.  
- Instead of updating the whole LLM (billions of parameters), LoRA injects **small adapter matrices** into attention layers and only trains those.  

### Why LoRA?
- Memory-efficient → only MBs instead of GBs.  
- Fast to train locally.  
- Base model remains unchanged → you can toggle LoRA ON/OFF.  

### Which adapters are used?
- Targeted the **attention projection layers**:  
  - `q_proj`, `k_proj`, `v_proj`, `out_proj`.  
- These are responsible for how tokens attend to each other.  
- Training them helps the model adapt to new summarization styles without retraining everything.  

### Benefits
- Smaller storage size .  
- Reusable adapters → load/unload without retraining.  
- Improved summary **style, faithfulness, and readability**.

---

## 📌 4. What is RAG and Why Used?
- **RAG (Retrieval-Augmented Generation)** = Retrieval + LLM.  
- Process:  
  1. User ingests `.txt`/`.pdf` into **Knowledge Base**.  
  2. Text is **chunked** and converted into embeddings (MiniLM).  
  3. Stored in **FAISS index**.  
  4. When a question is asked → Top-K relevant chunks are retrieved and passed to the LLM for grounded answering.  

- **Why used here?**  
  - Allows the model to **answer from your own documents**.  
  - No need for retraining when new files are added → just ingest again.  
  - Prevents hallucinations → answers are grounded in actual retrieved text.  

---

## 📌 5. Evaluation Methodology & Results

### Methodology
- **Quantitative Metrics:**
  - **ROUGE-1, ROUGE-2, ROUGE-L** → measure overlap with reference summaries.  
  - **Compression Ratio** → input length ÷ output length (good summaries compress text ~3x).  
- **Qualitative Checks:**
  - Human inspection for **faithfulness, readability, style**.  
  - Spot-checking RAG answers to ensure they come from retrieved chunks.  

### Results
| Model              | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression |
|--------------------|---------|---------|---------|-------------|
| Base DistilBART    | 0.2189  | 0.0871  | 0.2189  | 3.05 |
| Fine-tuned (LoRA)  | **0.2466** | **0.1196** | **0.2262** | 2.67 |

**Observations:**
- LoRA fine-tuning improved ROUGE scores.  
- Summaries became more **informative and academic in tone**.  
- RAG ensured **factual correctness** when answering user questions.  

---


---

## 📌 AI Agent Architecture Document

### Components
- **UI:** Streamlit (3 tabs: Paragraph, PDF, RAG Q&A).  
- **Summarizer:** DistilBART + LoRA adapters.  
- **Retriever:** MiniLM embeddings + FAISS index.  
- **Knowledge Base:** `knowledge_base/` folder + `meta.jsonl`.  

### Interaction Flow
1. User inputs text / uploads PDF / asks question.  
2. Preprocessing: extract text, chunk if long.  
3. Summarizer path: agent chooses *single* or *hierarchical* summarization.  
4. RAG path: retriever searches FAISS → chunks injected into prompt → summarizer generates grounded answer.  
5. UI displays output + retrieved evidence.  

### Models Used
- **DistilBART:** local summarization backbone.  
- **LoRA adapters:** fine-tuned style adaptation.  
- **MiniLM + FAISS:** efficient semantic retrieval.  

### Why These Choices
- DistilBART is **lightweight & summarization-ready**.  
- LoRA enables **fast, efficient fine-tuning**.  
- RAG ensures **dynamic knowledge ingestion** without retraining.  
- Streamlit provides a **simple demo UI**.

---

## 📌 Data Science Report

### Fine-Tuning Setup
- **Data:** `sample_dataset.jsonl` (train/test split).  
- **Method:** LoRA fine-tuning on DistilBART, targeting attention projections.  
- **Params:** epochs=2, batch_size=2, lr=2e-4, r=8, alpha=16, dropout=0.05.  
- **Artifacts:** adapters saved in `models/lora-distilbart/`.

### Evaluation Outcomes
- Base model summaries were short and generic.  
- LoRA improved ROUGE (+12% on ROUGE-1, +37% on ROUGE-2).  
- Compression ratio closer to ideal (~2.67).  
- Qualitatively: summaries more **faithful and structured**, especially in academic contexts.  

---

## ✅ Conclusion
This project demonstrates a **local, efficient AI summarization and RAG pipeline**:
- **DistilBART** → local summarizer.  
- **LoRA** → efficient fine-tuning for domain/style adaptation.  
- **RAG** → dynamic, document-grounded answering.  
- **Evaluation** proves LoRA improved summary quality, and RAG grounded answers in ingested context.  

Everything runs **offline** and is easily extensible for more documents, larger datasets, or bigger models.

---
# 🚀 Quick Start Guide: AI Summarization + RAG Agent

# 1️⃣ Clone the Repository
git clone [https://github.com/you/ai-agent-prototype-rag.git](https://github.com/ayushgupta67/Finetuned-summarization-and-RAG-agent)
cd ai-agent-prototype-rag/ai-agent-prototype

# 2️⃣ Create and Activate Virtual Environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# 3️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣  Fine-tune LoRA
python training/finetune_lora.py --output_dir models/lora-distilbart --epochs 2 --batch_size 2 --lr 2e-4

# 5️⃣ Build Knowledge Base
python build_kb.py --path data/

# 6️⃣ Run Streamlit App
streamlit run run_app.py

