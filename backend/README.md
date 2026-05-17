# 🤖 RAG Technical Documentation Assistant

A **production-ready, self-corrective RAG pipeline** for answering questions from technical documentation. Built with LangGraph, LangChain, ChromaDB, and Groq's Llama 3.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                          │
│  POST /query  │  POST /ingest  │  GET /documents  │  /feedback │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  LangGraph StateGraph                           │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │  Query Analysis  │ ← Rewrite + detect type (Groq Llama 3)   │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │    Retrieval     │ ← ChromaDB similarity search              │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │ Document Grading │ ← LLM relevance filter                    │
│  └────────┬────────┘                                            │
│           │                                                     │
│    ┌──────┴──────┐                                              │
│    │  Relevant?  │                                              │
│    └──┬──────┬───┘                                              │
│       │YES   │NO (retry < max)                                  │
│       │      │                                                  │
│       │  ┌───▼────────────┐                                     │
│       │  │  Query Rewrite  │ ← Reformulate for better retrieval │
│       │  └───────┬────────┘                                     │
│       │          │ (loop back to retrieval)                     │
│  ┌────▼──────────┘                                              │
│  │   Generation    │ ← Grounded answer with citations           │
│  └────────┬────────┘                                            │
│           │                                                     │
│  ┌────────▼────────┐                                            │
│  │Hallucination Chk│ ← Verify answer vs. source docs            │
│  └────────┬────────┘                                            │
│           │                                                     │
│         [END]                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── router.py               # Central API router
│   │   └── routes/
│   │       ├── query.py            # POST /query (RAG pipeline)
│   │       ├── ingest.py           # POST /ingest
│   │       ├── documents.py        # GET /documents
│   │       ├── feedback.py         # POST /feedback
│   │       └── health.py           # GET /health
│   ├── graph/
│   │   ├── rag_graph.py            # LangGraph StateGraph assembly
│   │   └── nodes/
│   │       ├── query_analysis.py   # Query rewrite + type detection
│   │       ├── retrieval.py        # ChromaDB similarity search
│   │       ├── grading.py          # LLM document relevance filter
│   │       ├── query_rewrite.py    # Query reformulation
│   │       ├── generation.py       # Answer synthesis with citations
│   │       └── hallucination_check.py  # Grounding verification
│   ├── ingestion/
│   │   ├── document_loader.py      # File loaders (md, txt, html)
│   │   └── ingestion_pipeline.py  # Chunk → embed → store pipeline
│   ├── models/
│   │   └── schemas.py              # Pydantic models + RAGState TypedDict
│   ├── services/
│   │   ├── embedding_service.py    # HuggingFace sentence-transformers
│   │   ├── vector_store_service.py # ChromaDB wrapper
│   │   ├── llm_service.py          # Groq ChatGroq wrapper
│   │   └── feedback_service.py     # JSON-backed feedback store
│   ├── utils/
│   │   ├── logger.py               # Centralized logging
│   │   └── helpers.py              # Utility functions
│   └── main.py                     # FastAPI app + lifespan
├── data/
│   ├── chroma_db/                  # ChromaDB persistent storage
│   ├── sample_docs/                # Sample Markdown documentation
│   └── feedback.json               # Feedback storage
├── tests/
│   ├── test_api.py                 # FastAPI endpoint tests
│   ├── test_graph.py               # LangGraph node unit tests
│   └── test_ingestion.py           # Ingestion pipeline tests
├── .env.example                    # Environment variable template
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml              # Docker Compose config
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Test configuration
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Groq API key ([console.groq.com](https://console.groq.com/))
- 2GB+ RAM (for embedding model)

### 2. Clone and Setup

```bash
git clone <repo-url>
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

### 4. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at **http://localhost:8000**
- Interactive docs: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**
- Health check: **http://localhost:8000/api/v1/health**

---

## 🐳 Docker

### Build and Run

```bash
# Build image
docker build -t rag-assistant .

# Run with your API key
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data \
  rag-assistant
```

### Docker Compose

```bash
# Set your API key in .env file first
echo "GROQ_API_KEY=your_key_here" > .env

docker-compose up -d
docker-compose logs -f
```

---

## 📡 API Reference

### `POST /api/v1/query`

Submit a question to the RAG pipeline.

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does FastAPI handle async requests?",
    "session_id": "my-session-001",
    "top_k": 5,
    "max_retries": 2,
    "stream": false
  }'
```

**Response:**
```json
{
  "query": "How does FastAPI handle async requests?",
  "rewritten_query": "What is FastAPI's asynchronous request handling mechanism?",
  "answer": "FastAPI handles async requests using Python's asyncio...\n\n[Document 1]...",
  "citations": [
    {
      "source": "data/sample_docs/fastapi_guide.md",
      "chunk_id": "abc123",
      "relevance_score": 0.94,
      "excerpt": "FastAPI uses Python's asyncio under the hood...",
      "title": "FastAPI Guide"
    }
  ],
  "query_type": "conceptual",
  "retry_count": 0,
  "retrieved_doc_count": 5,
  "filtered_doc_count": 3,
  "is_hallucination": false,
  "hallucination_score": 0.04,
  "session_id": "my-session-001",
  "processing_time_ms": 2341.5
}
```

### `POST /api/v1/query` (Streaming)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain LangGraph StateGraph", "stream": true}'
```

Returns Server-Sent Events (SSE):
```
data: {"event": "start", "session_id": "xxx"}
data: {"event": "token", "content": "LangGraph "}
data: {"event": "token", "content": "uses "}
...
data: {"event": "end", "retry_count": 0}
```

### `POST /api/v1/ingest`

Load documents into ChromaDB.

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "source_path": "./data/my_docs",
    "collection_name": "technical_docs",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "recursive": true
  }'
```

**Supported file types:** `.md`, `.markdown`, `.txt`, `.html`, `.htm`

### `GET /api/v1/documents`

List ingested document chunks.

```bash
curl "http://localhost:8000/api/v1/documents?limit=20&offset=0"
```

### `POST /api/v1/feedback`

Submit feedback on an answer.

```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-001",
    "query": "How does FastAPI handle async?",
    "answer": "FastAPI uses asyncio...",
    "rating": 5,
    "is_helpful": true,
    "comment": "Very clear explanation!"
  }'
```

### `GET /api/v1/health`

```bash
curl "http://localhost:8000/api/v1/health"
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "chroma_connected": true,
  "llm_available": true,
  "embeddings_loaded": true,
  "collection_name": "technical_docs",
  "document_count": 148,
  "uptime_seconds": 3600.5
}
```

---

## 🗂️ Ingesting Your Own Documentation

Place your documentation files in `data/sample_docs/` or any directory, then call the ingest API:

```bash
# Via API
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -d '{"source_path": "/path/to/your/docs"}'

# Or use the Python pipeline directly
from app.ingestion.ingestion_pipeline import IngestionPipeline

pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=200)
result = pipeline.ingest("./my_docs/")
print(f"Ingested {result['chunks_created']} chunks from {result['files_processed']} files")
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_graph.py -v

# Run specific test class
pytest tests/test_api.py::TestQueryEndpoint -v
```

---

## 🔧 RAG Pipeline State

The LangGraph state contains:

| Field | Type | Description |
|---|---|---|
| `original_query` | str | Raw user input |
| `rewritten_query` | str | LLM-improved query |
| `retrieved_docs` | List[Dict] | Raw ChromaDB results |
| `filtered_docs` | List[Dict] | Relevance-filtered docs |
| `generated_answer` | str | Final LLM answer |
| `retry_count` | int | Current retry attempt |
| `max_retries` | int | Maximum retries allowed |
| `citations` | List[Dict] | Source references |
| `query_type` | str | factual/conceptual/procedural/comparison/troubleshooting |
| `is_hallucination` | bool | Hallucination detection result |
| `hallucination_score` | float | 0.0 (grounded) → 1.0 (hallucinated) |
| `conversation_history` | List[Dict] | Session memory turns |
| `session_id` | str | Session identifier |

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **Required** | Groq API key |
| `GROQ_MODEL` | `llama3-70b-8192` | Groq model name |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Max response tokens |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION` | `technical_docs` | Collection name |
| `RETRIEVAL_TOP_K` | `5` | Documents to retrieve |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| **Web Framework** | FastAPI 0.115 |
| **Graph Orchestration** | LangGraph 0.2 |
| **LLM Framework** | LangChain 0.3 |
| **LLM Provider** | Groq (Llama 3 70B) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Database** | ChromaDB 0.5 |
| **Validation** | Pydantic v2 |
| **Testing** | pytest + httpx |
| **Containerization** | Docker + Docker Compose |

---

## 📊 Performance Notes

- **Embedding model**: ~384-dimensional vectors, ~80ms per batch
- **Retrieval**: ChromaDB HNSW index, ~10ms for k=5
- **LLM latency**: Groq is extremely fast (~500 tokens/sec for Llama 3)
- **Total pipeline**: ~2-5 seconds per query (no retries)
- **Throughput**: 1 concurrent query per worker (scale horizontally)

For GPU acceleration: set `EMBEDDING_DEVICE=cuda` (requires CUDA PyTorch)

---

## 🔒 Security Considerations

- Never expose `GROQ_API_KEY` in client-side code
- The CORS policy allows all origins by default — restrict in production
- Input validation via Pydantic limits query length (max 2000 chars)
- ChromaDB data directory should be excluded from public access
- Run the Docker container as non-root user (already configured)

---

## 📝 License

MIT License — see LICENSE file for details.
