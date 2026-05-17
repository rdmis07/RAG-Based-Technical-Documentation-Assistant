# LangChain Developer Guide

## What is LangChain?

LangChain is a framework for developing applications powered by language models. It enables:
- **Context-aware**: Connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
- **Reasoning**: Rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)

## Installation

```bash
pip install langchain
pip install langchain-openai  # For OpenAI models
pip install langchain-groq    # For Groq models
pip install langchain-community  # Community integrations
```

## Core Components

### LLMs and Chat Models

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.1,
    groq_api_key="your-api-key"
)

response = llm.invoke("What is the capital of France?")
print(response.content)
```

### Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | llm
response = chain.invoke({"question": "Tell me about Python"})
```

### Document Loaders

```python
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader

# Load a text file
loader = TextLoader("./my_document.txt")
docs = loader.load()

# Load a markdown file
md_loader = UnstructuredMarkdownLoader("./README.md")
md_docs = md_loader.load()
```

### Text Splitters

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")
```

### Embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Embed a query
query_embedding = embeddings.embed_query("What is machine learning?")

# Embed documents
doc_embeddings = embeddings.embed_documents(["Doc 1 text", "Doc 2 text"])
```

### Vector Stores

```python
from langchain_chroma import Chroma

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Similarity search
results = vectorstore.similarity_search("machine learning", k=3)

# Search with scores
results_with_scores = vectorstore.similarity_search_with_relevance_scores(
    "machine learning", k=3
)
```

## Chains

### Simple Chain (LCEL)

LangChain Expression Language (LCEL) uses the pipe `|` operator:

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is Python?"})
```

### RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the difference between async and sync?")
```

## Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

response1 = conversation.predict(input="Hi, my name is Alice")
response2 = conversation.predict(input="What's my name?")  # Remembers "Alice"
```

## Agents

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information."""
    # Implementation here
    return f"Wikipedia result for: {query}"

tools = [search_wikipedia]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "What year was Python created?"})
```

## Streaming

```python
# Stream tokens
for chunk in chain.stream({"question": "Tell me a story"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"question": "Tell me a story"}):
    print(chunk, end="", flush=True)
```

## Output Parsers

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

parser = JsonOutputParser(pydantic_object=MovieReview)
chain = prompt | llm | parser
result = chain.invoke({"movie": "Inception"})
# Returns: MovieReview(title='Inception', rating=9.0, summary='...')
```

## Callbacks

```python
from langchain.callbacks import StdOutCallbackHandler

handler = StdOutCallbackHandler()
llm_with_callbacks = llm.with_config(callbacks=[handler])
```

## Best Practices

1. **Use LCEL** for composing chains — it's more readable and supports streaming natively
2. **Async first** — use `ainvoke`, `astream` for production workloads
3. **Set temperature=0** for factual/deterministic tasks
4. **Use structured output** with Pydantic models when you need specific formats
5. **Cache embeddings** — embeddings are expensive, cache them in a vector store
6. **Chunk strategically** — smaller chunks (500-1000 chars) work better for precision, larger chunks for context
