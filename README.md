# Graph RAG System - UET Prospectus

A robust Retrieval-Augmented Generation (RAG) system answering questions about UET Lahore's prospectus using **Hybrid Search** (Graph Traversal + Vector Embeddings).

## 🚀 Features

-   **Hybrid Retrieval**: Combines Neo4j Graph traversal (for entities) and Vector Search (for semantic similarity).
-   **Multi-Provider LLM**: Support for **Gemini**, **Ollama**, and **vLLM** (switchable via config).
-   **Content Guardrail**: LLM-based filter to restrict answers to department/academic topics only.
-   **Modern Stack**:
    -   **Backend**: FastAPI (with lazy loading & pre-warming).
    -   **Frontend**: Streamlit (chat interface with source citations).
    -   **Database**: Neo4j (Graph + Vector Index).
    -   **Observability**: LangSmith integration.

## 📂 Project Structure

```
project-graph-rag/
├── api/                 # FastAPI Endpoints & Schemas
├── config/              # Configuration (Settings, Prompts)
├── frontend/            # Streamlit Application
├── src/                 # Core Logic (LLM Clients, Graph Manager, RAG)
├── scripts/             # Utility Scripts (Embeddings, Ingestion)
├── examples/            # Standalone usage examples
├── .env.example         # Environment variables template
├── main.py              # Backend Entry Point
└── pyproject.toml       # Python Dependencies
```

## 🛠️ Installation

1.  **Clone & Install Dependencies**:
    ```bash
    pip install -e .
    ```

2.  **Setup Environment**:
    Copy `.env.example` to `.env` and configure your keys:
    ```bash
    cp .env.example .env
    ```
    *Required keys*: `NEO4J_URI`, `NEO4J_PASSWORD`, and your chosen LLM provider's API key.

3.  **Populate Database**:
    -   Ingest PDF: `python examples/ingest_pdf.py`
    -   Update Embeddings (if needed): `python scripts/update_embeddings.py`

## 🚦 Running the Application

### 1. Start Backend (FastAPI)
```bash
python main.py
```
> API runs at: `http://localhost:8000`  
> Docs at: `http://localhost:8000/docs`

### 2. Start Frontend (Streamlit)
```bash
streamlit run frontend/app.py
```
> UI opens at: `http://localhost:8501`

## ⚙️ Configuration

Control behavior via `.env`:

| Variable | Description | Options |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | Active LLM Backend | `gemini`, `ollama`, `vllm` |
| `ENABLE_GUARDRAIL` | Enable topic filtering | `true`, `false` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith | `true` |

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```
