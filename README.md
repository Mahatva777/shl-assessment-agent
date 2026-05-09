# SHL Assessment Recommendation Agent

An AI-powered conversational agent that recommends SHL assessments based on
job descriptions, hiring requirements, and natural-language queries.

---

## Quick start

### 1. Clone & install

```bash
git clone <repo-url> && cd shl-assessment-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY (required for the LLM)
```

### 3. Run the dev server

```bash
uvicorn app.main:app --reload
```

The API will be available at **http://127.0.0.1:8000**.

- Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: `GET /health`
- Chat endpoint: `POST /chat`

### 4. Example request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need a test for a junior Python developer"}
    ]
  }'
```

---

## Running tests

```bash
python -m pytest app/tests/ -v
```

---

## Project structure

```
shl-assessment-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry-point (/health, /chat)
│   ├── schemas.py            # Pydantic request/response models
│   ├── config.py             # Environment-based settings
│   ├── catalog_loader.py     # Load & normalise SHL product catalog
│   ├── scoring.py            # Seniority mapping & scoring helpers
│   ├── retrieval.py          # Embedding index & candidate retrieval
│   ├── state_extraction.py   # Extract structured constraints from chat
│   ├── policy.py             # Conversation policy / decision logic
│   ├── llm_client.py         # Thin wrapper over Google Gemini LLM
│   ├── agent.py              # Top-level agent orchestrator
│   └── tests/
│       ├── test_scoring.py   # Unit tests for scoring utilities
│       └── test_schema.py    # Schema validation tests
├── data/
│   └── shl_product_catalog.json
├── .env.example
├── requirements.txt
└── README.md
```

---

## Environment variables

| Variable           | Default                       | Description                       |
|--------------------|-------------------------------|-----------------------------------|
| `MODEL_NAME`       | `gemini-2.0-flash`            | LLM model identifier              |
| `GOOGLE_API_KEY`   | *(required)*                  | Google AI API key                  |
| `CATALOG_PATH`     | `data/shl_product_catalog.json` | Path to the SHL product catalog  |
| `EMBEDDING_MODEL`  | `models/text-embedding-004`   | Embedding model for retrieval      |
| `TOP_K`            | `10`                          | Number of candidates to retrieve   |
| `HOST`             | `0.0.0.0`                     | Server bind address                |
| `PORT`             | `8000`                        | Server port                        |
| `DEBUG`            | `false`                       | Enable debug mode                  |
