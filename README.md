# Graph RAG Project

A Graph RAG system using Neo4j and vLLM to answer questions about the UET Prospectus.

## Folder Structure
- `config/`: Configuration files (models, prompts)
- `src/`: Core source code
  - `llm/`: Neo4j and vLLM clients
  - `prompt_engineering/`: Prompt templates
- `data/`: Data storage (PDFs, cache)
- `examples/`: Run scripts

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place `UET Prospectus.pdf` in `data/`.
3. Run ingestion: `python examples/ingest_pdf.py`
4. Chat: `python examples/chat_session.py`
