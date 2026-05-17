# FastAPI Complete Guide

## Overview

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.

**Key features:**
- **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic)
- **Fast to code**: Increase the speed to develop features by about 200% to 300%
- **Fewer bugs**: Reduce about 40% of human (developer) induced errors
- **Intuitive**: Great editor support. Completion everywhere. Less time debugging
- **Easy**: Designed to be easy to use and learn. Less time reading docs
- **Short**: Minimize code duplication. Multiple features from each parameter declaration
- **Robust**: Get production-ready code. With automatic interactive documentation
- **Standards-based**: Based on (and fully compatible with) the open standards for APIs: OpenAPI and JSON Schema

## Installation

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

## Creating Your First App

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run the server:
```bash
uvicorn main:app --reload
```

## Path Parameters

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

Path parameters with types are validated automatically. If you pass a string where an int is expected, FastAPI returns a 422 Unprocessable Entity error.

## Query Parameters

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

Query parameters are function parameters that are not part of the path parameters.

## Request Body with Pydantic

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

## Async Support

FastAPI supports both synchronous and asynchronous route handlers.

```python
# Async route (recommended for I/O-bound operations)
@app.get("/async-items/{item_id}")
async def read_item_async(item_id: str):
    result = await some_async_database_call(item_id)
    return result

# Sync route (use for CPU-bound or blocking operations)
@app.get("/sync-items/{item_id}")
def read_item_sync(item_id: str):
    result = some_blocking_call(item_id)
    return result
```

FastAPI uses Python's asyncio under the hood. When you use `async def`, FastAPI runs the handler in the event loop. For sync functions, FastAPI runs them in a thread pool to avoid blocking the event loop.

## Dependency Injection

FastAPI has a powerful dependency injection system:

```python
from fastapi import Depends

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/{user_id}")
async def read_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()
```

## Error Handling

```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}
```

## Background Tasks

```python
from fastapi import BackgroundTasks

def send_email_notification(email: str, message: str):
    # This runs in the background after the response is sent
    send_email(email, message)

@app.post("/send-notification/")
async def send_notification(
    background_tasks: BackgroundTasks,
    email: str,
):
    background_tasks.add_task(send_email_notification, email, "Hello!")
    return {"message": "Notification sent in background"}
```

## Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Response Models

```python
class UserOut(BaseModel):
    id: int
    username: str
    email: str

@app.get("/users/{user_id}", response_model=UserOut)
async def get_user(user_id: int):
    # Password is excluded from response automatically
    return get_user_from_db(user_id)
```

## Status Codes

```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return item
```

## Router Organization

For larger applications, use APIRouter to organize routes:

```python
# routers/items.py
from fastapi import APIRouter

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/")
async def list_items():
    return []

# main.py
from routers import items
app.include_router(items.router)
```

## Testing

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
```

## Deployment

### With Uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Gunicorn + Uvicorn workers
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
