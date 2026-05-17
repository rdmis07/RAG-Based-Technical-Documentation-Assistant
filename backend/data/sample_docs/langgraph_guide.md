# LangGraph Developer Guide

## What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.

Key benefits:
- **Stateful**: Maintain state across multiple LLM calls
- **Cyclical**: Support loops and branching (unlike simple chains)
- **Controllable**: Fine-grained control over agent behavior
- **Persistent**: Built-in checkpointing for long-running workflows

## Installation

```bash
pip install langgraph
pip install langchain-core
```

## Core Concepts

### State

The state is a TypedDict that is shared across all nodes:

```python
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    messages: List[str]
    current_step: str
    result: Optional[str]
    retry_count: int
```

### Nodes

Nodes are Python functions that take the state and return updates:

```python
def my_node(state: AgentState) -> dict:
    # Read from state
    messages = state["messages"]
    
    # Do some processing
    result = process(messages)
    
    # Return ONLY the fields that changed
    return {"result": result, "current_step": "done"}
```

### Edges

Edges connect nodes:
- **Normal edges**: Always go from node A to node B
- **Conditional edges**: Route based on state values

```python
from langgraph.graph import StateGraph, END, START

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("step1", step1_fn)
workflow.add_node("step2", step2_fn)
workflow.add_node("step3", step3_fn)

# Normal edge: step1 always goes to step2
workflow.add_edge("step1", "step2")

# Conditional edge
def route(state: AgentState) -> str:
    if state["result"] is not None:
        return "step3"
    return "step1"  # Retry

workflow.add_conditional_edges(
    "step2",   # From node
    route,     # Decision function
    {          # Mapping: return value → node name
        "step3": "step3",
        "step1": "step1",
    }
)

# Entry and exit points
workflow.add_edge(START, "step1")
workflow.add_edge("step3", END)

# Compile
graph = workflow.compile()
```

## Building a RAG Graph

```python
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Dict, Any, Optional

class RAGState(TypedDict):
    original_query: str
    rewritten_query: str
    retrieved_docs: List[Dict[str, Any]]
    filtered_docs: List[Dict[str, Any]]
    generated_answer: str
    retry_count: int
    max_retries: int
    citations: List[Dict[str, Any]]

def query_analysis(state: RAGState) -> dict:
    query = state["original_query"]
    # Rewrite query for better retrieval
    rewritten = rewrite_query(query)
    return {"rewritten_query": rewritten, "retry_count": 0}

def retrieve(state: RAGState) -> dict:
    query = state["rewritten_query"]
    docs = vector_store.similarity_search(query, k=5)
    return {"retrieved_docs": [{"content": d.page_content} for d in docs]}

def grade_documents(state: RAGState) -> dict:
    docs = state["retrieved_docs"]
    query = state["rewritten_query"]
    filtered = [d for d in docs if is_relevant(query, d["content"])]
    return {"filtered_docs": filtered}

def should_retrieve_again(state: RAGState) -> str:
    if state["filtered_docs"]:
        return "generate"
    if state["retry_count"] < state["max_retries"]:
        return "rewrite"
    return "generate"  # Fallback

def rewrite_query(state: RAGState) -> dict:
    new_query = improve_query(state["rewritten_query"])
    return {
        "rewritten_query": new_query,
        "retry_count": state["retry_count"] + 1
    }

def generate(state: RAGState) -> dict:
    docs = state["filtered_docs"]
    answer = llm.generate(state["original_query"], docs)
    return {"generated_answer": answer}

# Build the graph
workflow = StateGraph(RAGState)
workflow.add_node("analysis", query_analysis)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate)

workflow.add_edge(START, "analysis")
workflow.add_edge("analysis", "retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    should_retrieve_again,
    {"generate": "generate", "rewrite": "rewrite"}
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

graph = workflow.compile()

# Run it
result = graph.invoke({
    "original_query": "How does FastAPI handle async?",
    "max_retries": 2,
    "retry_count": 0,
    "retrieved_docs": [],
    "filtered_docs": [],
    "generated_answer": "",
    "citations": [],
    "rewritten_query": "",
})
```

## Conditional Edges Patterns

### Simple Binary Routing

```python
def route(state) -> str:
    return "yes_node" if state["condition"] else "no_node"

workflow.add_conditional_edges("decision", route, {
    "yes_node": "yes_node",
    "no_node": "no_node"
})
```

### Multiple Paths

```python
def complex_route(state) -> str:
    if state["score"] > 0.9:
        return "high_quality"
    elif state["score"] > 0.5:
        return "medium_quality"
    else:
        return "low_quality"

workflow.add_conditional_edges(
    "grader",
    complex_route,
    {
        "high_quality": "fast_path",
        "medium_quality": "standard_path",
        "low_quality": "slow_path",
    }
)
```

## Checkpointing (Persistence)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# Run with thread ID for persistence
config = {"configurable": {"thread_id": "session-123"}}
result = graph.invoke(initial_state, config=config)

# Resume from checkpoint
result2 = graph.invoke({"input": "continue"}, config=config)
```

## Streaming

```python
# Stream events
for event in graph.stream(initial_state):
    node_name = list(event.keys())[0]
    state_update = event[node_name]
    print(f"Node '{node_name}' produced: {state_update}")

# Stream token-level output
async for token in graph.astream_events(initial_state, version="v1"):
    if token["event"] == "on_chat_model_stream":
        print(token["data"]["chunk"].content, end="")
```

## Human-in-the-Loop

```python
from langgraph.types import interrupt

def human_approval(state):
    # This pauses the graph and waits for human input
    answer = interrupt("Please approve this action: " + state["plan"])
    if answer == "yes":
        return {"approved": True}
    return {"approved": False}
```

## Parallel Node Execution

```python
# Fan-out: analysis → [retrieve_web, retrieve_db] → merge
workflow.add_edge("analysis", "retrieve_web")
workflow.add_edge("analysis", "retrieve_db")
# LangGraph executes both in parallel automatically
workflow.add_edge("retrieve_web", "merge")
workflow.add_edge("retrieve_db", "merge")
```

## Debugging

```python
# Visualize graph structure
print(graph.get_graph().draw_mermaid())

# Step-by-step execution with full state
for step in graph.stream(initial_state, stream_mode="values"):
    print("Current state:", step)
```
