# RAG Based Technical Documentation Assistant

An AI-powered Retrieval-Augmented Generation (RAG) assistant designed for querying technical documentation using semantic search, LangGraph workflows, and LLM-based response generation.

The project combines FastAPI, LangChain, LangGraph, ChromaDB, and Groq-powered LLMs with a React + Vite frontend to provide scalable and intelligent documentation querying.

---

# Features

- Semantic document retrieval
- LangGraph-based RAG workflow
- Query analysis and rewriting
- Hallucination detection
- Document grading and filtering
- Citation-aware answer generation
- Vector embeddings with ChromaDB
- FastAPI backend APIs
- React + TypeScript frontend
- User feedback collection system

---

# Architecture

```text
User Query
   ↓
Query Analysis
   ↓
Document Retrieval
   ↓
Document Grading
   ↓
Query Rewrite (if needed)
   ↓
LLM Generation
   ↓
Hallucination Check
   ↓
Final Response
```

---

# Tech Stack

## Backend
- Python
- FastAPI
- LangChain
- LangGraph
- ChromaDB
- Groq API

## Frontend
- React
- TypeScript
- Vite
- TailwindCSS

---

# Project Structure

```bash
backend/
│
├── app/
│   ├── api/
│   ├── graph/
│   ├── ingestion/
│   ├── models/
│   ├── services/
│   └── utils/
│
├── data/sample_docs/
├── tests/
├── requirements.txt
└── Dockerfile

src/
public/
package.json
vite.config.ts
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/rdmis07/RAG-Based-Technical-Documentation-Assistant.git
cd RAG-Based-Technical-Documentation-Assistant
```

---

# Backend Setup

```bash
cd backend
python -m venv .venv
```

## Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Backend

```bash
uvicorn app.main:app --reload
```

---

# Frontend Setup

```bash
npm install
npm run dev
```
# Screenshots
---

# Environment Variables

Create a `.env` file inside backend directory.

```env
GROQ_API_KEY=your_api_key_here
```

---

# API Features

- Document ingestion
- Semantic retrieval
- Query rewriting
- Response generation
- Hallucination verification
- Vector storage
- Feedback APIs

---

# Future Improvements

- Authentication system
- Streaming LLM responses
- PDF upload interface
- Multi-user support
- Chat history persistence
- Cloud deployment support
