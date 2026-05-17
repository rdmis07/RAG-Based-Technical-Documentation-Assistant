# ChromaDB Guide

## What is ChromaDB?

ChromaDB is an AI-native open-source vector database. It makes it easy to build LLM apps by storing and retrieving embeddings and associated metadata.

## Installation

```bash
pip install chromadb
```

## Quick Start

```python
import chromadb

# In-memory client (ephemeral)
client = chromadb.EphemeralClient()

# Persistent client (data saved to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
collection = client.create_collection(name="my_docs")

# Add documents
collection.add(
    documents=["This is document 1", "This is document 2"],
    metadatas=[{"source": "file1.txt"}, {"source": "file2.txt"}],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_texts=["search query"],
    n_results=2
)
```

## PersistentClient

```python
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True,
    )
)
```

## Collections

```python
# Create or get collection
collection = client.get_or_create_collection(
    name="technical_docs",
    metadata={"hnsw:space": "cosine"}  # Distance metric
)

# List all collections
collections = client.list_collections()

# Delete a collection
client.delete_collection("old_collection")

# Get collection info
print(f"Count: {collection.count()}")
```

## Custom Embeddings

```python
from chromadb.utils import embedding_functions

# Use sentence-transformers
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="my_collection",
    embedding_function=sentence_transformer_ef
)
```

## Adding Documents

```python
# Add with embeddings pre-computed
collection.add(
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4]],
    documents=["document 1", "document 2"],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}],
    ids=["id1", "id2"]
)

# Upsert (add or update)
collection.upsert(
    documents=["Updated document"],
    ids=["id1"]
)
```

## Querying

```python
# Basic similarity search
results = collection.query(
    query_texts=["search query"],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

# Query with metadata filter
results = collection.query(
    query_texts=["machine learning"],
    n_results=3,
    where={"source": "textbook.pdf"},  # Metadata filter
    where_document={"$contains": "neural"}  # Content filter
)

# Get by ID
result = collection.get(
    ids=["id1", "id2"],
    include=["documents", "metadatas"]
)

# Get all documents
all_docs = collection.get(
    limit=100,
    offset=0,
    include=["documents", "metadatas"]
)
```

## Metadata Filtering

ChromaDB supports rich metadata filtering:

```python
# Equality
where={"category": "python"}

# Comparison
where={"year": {"$gt": 2020}}
where={"score": {"$gte": 0.8}}

# Logical operators
where={
    "$and": [
        {"category": "python"},
        {"year": {"$gt": 2020}}
    ]
}

where={
    "$or": [
        {"source": "book1.pdf"},
        {"source": "book2.pdf"}
    ]
}

# Not equal
where={"status": {"$ne": "draft"}}

# In list
where={"tag": {"$in": ["fastapi", "python"]}}
```

## Distance Metrics

ChromaDB supports multiple distance metrics:

```python
# Cosine similarity (default for most embedding models)
collection = client.create_collection(
    name="cosine_collection",
    metadata={"hnsw:space": "cosine"}
)

# L2 (Euclidean distance)
collection = client.create_collection(
    name="l2_collection",
    metadata={"hnsw:space": "l2"}
)

# Inner product
collection = client.create_collection(
    name="ip_collection",
    metadata={"hnsw:space": "ip"}
)
```

## LangChain Integration

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# Using LangChain's Chroma wrapper
embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# With persist directory
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_fn,
    collection_name="my_docs"
)

# With existing client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vectorstore = Chroma(
    client=chroma_client,
    collection_name="my_docs",
    embedding_function=embedding_fn
)

# Add documents
vectorstore.add_documents(documents)

# Similarity search
docs = vectorstore.similarity_search("query", k=5)

# Search with scores
docs_scores = vectorstore.similarity_search_with_relevance_scores("query", k=5)
for doc, score in docs_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}")
```

## Performance Tips

1. **Batch operations**: Add documents in batches of 100-1000 for better performance
2. **HNSW tuning**: Adjust `hnsw:M` and `hnsw:ef_construction` for speed/accuracy trade-off
3. **Use cosine similarity** for normalized embeddings (sentence-transformers output normalized vectors)
4. **Index size**: ChromaDB keeps indices in memory; plan for ~4KB per vector for all-MiniLM-L6-v2 (384 dims × 4 bytes × overhead)

```python
# HNSW performance tuning
collection = client.create_collection(
    name="optimized_collection",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 32,              # More connections, slower build, faster query
        "hnsw:ef_construction": 200,  # Higher = better index, slower build
        "hnsw:ef": 100,            # Higher = better recall, slower query
    }
)
```

## Resetting and Deleting

```python
# Reset entire database (delete all collections)
client.reset()

# Delete specific collection
client.delete_collection("my_collection")

# Delete specific documents by ID
collection.delete(ids=["id1", "id2"])

# Delete by metadata filter
collection.delete(where={"source": "old_file.txt"})
```
