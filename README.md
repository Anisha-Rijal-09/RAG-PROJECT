# RAG Backend — Document Ingestion & Conversational RAG API

A FastAPI backend for document-based conversational AI using RAG.

## Features

- Upload PDF/TXT documents
- Two chunking strategies: `recursive` and `fixed`
- Generate embeddings using Sentence Transformers
- Store vectors in Qdrant
- Store metadata and bookings in SQLite
- Redis-based multi-turn conversation memory
- Groq LLM for response generation and interview booking extraction
- Swagger API documentation
---
## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| Vector Database | Qdrant |
| Chat Memory | Redis |
| Metadata Database | SQLite |
| ORM | SQLAlchemy |
| Embeddings | Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2` |
| LLM | Groq |
| PDF Parsing | pypdf |
| Containerization | Docker |

---

## Project Structure

```text
app/
├── api/
│   ├── chat.py
│   └── ingestion.py
├── db/
│   ├── models.py
│   └── session.py
├── services/
│   ├── booking.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── memory.py
│   └── vectorstore.py
├── init_db.py
└── main.py

requirements.txt
README.md
.gitignore
```

## Architecture

```text
                    ┌──────────────────┐
                    │     FastAPI      │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
      Document Ingestion              Conversational Chat
              │                             │
              ▼                             ▼
       Text Extraction                 Query Embedding
              │                             │
              ▼                             ▼
          Chunking                     Qdrant Search
              │                             │
              ▼                             ▼
        Embeddings                    Retrieved Chunks
              │                             │
              ▼                             ▼
           Qdrant                     Prompt + Memory
                                            │
                              ┌─────────────┴─────────────┐
                              │                           │
                              ▼                           ▼
                           Groq LLM                Redis Memory
                              │
                              ▼
                           Response
                              │
                              ▼
                       Booking Detection
                              │
                              ▼
                           SQLite
```

---


## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Anisha-Rijal-09/RAG-PROJECT.git
cd RAG-PROJECT

```

### 2. Create virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Qdrant and Redis

```bash
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 6379:6379 redis
```

### 5. Configure `.env`

```env
GROQ_API_KEY=your_groq_api_key
```

### 6. Initialize database

```bash
python -m app.init_db
```

### 7. Run the API

```bash
uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## APIs

### Upload Document

```text
POST /documents/upload
```

Supports:

- PDF
- TXT
- `recursive` chunking
- `fixed` chunking

### Chat

```text
POST /chat
```

Example:

```json
{
  "session_id": "user-123",
  "message": "What does the document say about pricing?"
}
```

The API retrieves relevant document chunks, uses Redis conversation history, and generates a response with the Groq LLM.

## Interview Booking

The chat API can extract:

- Name
- Email
- Date
- Time

When all required information is provided, the booking is saved in SQLite.

## Testing

Run:

```bash
python -m compileall app
```

Then test the APIs through Swagger:

```text
http://127.0.0.1:8000/docs
```

## Notes

- `.env`, `.venv`, `uploads`, `__pycache__`, and database files are excluded using `.gitignore`.
- RAG retrieval and generation are implemented explicitly without LangChain chain abstractions.
