# Architecture

Reason–Plan–Execute:

1. **Loader**: Detects input type (text/PDF) and extracts text.
2. **Planner**: Chooses single-pass vs hierarchical multi-pass summarization.
3. **Executor**: Runs model per chunk and combines results; verifies length/compression.
4. **Model Wrapper**: DistilBART base, with optional LoRA adapters.
5. **UI**: Streamlit tabs for Text vs PDF.
