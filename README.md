# SHL Conversational Assessment Recommender

A conversational recommendation system for grounded SHL assessment selection using semantic retrieval and LLM-guided consultative reasoning.

The system is designed to behave like an SHL consultant:

* consultative instead of robotic
* grounded instead of hallucinating
* concise instead of overly verbose
* flexible without brittle hardcoded flows

---

# Architecture

The system intentionally combines:

```text
Deterministic Retrieval + Grounding + LLM Conversational Reasoning
```

---

## High-Level Retrieval Flow

```text
Conversation
    ↓
Lightweight State Extraction
    ↓
Query Construction & Safety / Compare / Refine Gates
    ↓
Embedding Similarity Search (sentence-transformers)
    ↓
Diversification (family-capping via battery.py)
    ↓
Rule-based Anchor Injections (OPQ32r / HIPAA constraints)
    ↓
LLM Conversational Reasoning (Claude 3.5 Sonnet)
    ↓
Structured API Response
```

---

# Core Design

## Deterministic Infrastructure

Responsible for:

* Semantic retrieval using `sentence-transformers` (`all-MiniLM-L6-v2`) and NumPy cosine similarity
* Pre-computed embedding management (loaded in FastAPI lifespan to prevent cold-starts)
* Catalog grounding and validation
* Safety filtering and boundary enforcement
* Compare/refine detection
* Diversification (capping product families to max 3)

---

## LLM Conversational Layer (Claude 3.5 Sonnet)

Responsible for:

* 6 strict modes: clarify, consult, recommend, refine, compare, refuse
* Conversational reasoning and tone
* Clarification quality and directional guidance (via consult mode)
* Recommendation timing and shortlist explanation
* Conversation completion and lifecycle management

---

# Stateless Conversation Model

The backend is fully stateless.

Every request must include the full conversation history.

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring senior leadership"
    },
    {
      "role": "assistant",
      "content": "Is this for screening or executive benchmarking?"
    },
    {
      "role": "user",
      "content": "Executive benchmarking"
    }
  ]
}
```

Without full history:

* refinement breaks
* consultative continuity breaks
* recommendation staging breaks

---

# Conversation Modes

The LLM must choose exactly one mode:

| Mode      | Purpose                                      |
| --------- | -------------------------------------------- |
| clarify   | Ask one high-value question                  |
| consult   | Directional guidance / flag catalog gaps without final shortlist |
| recommend | Return grounded recommendations              |
| refine    | Modify prior recommendations (requires prior history) |
| compare   | Compare grounded products                    |
| refuse    | Reject unsupported/off-topic requests        |

---

# Consultative Recommendation Staging

The system supports:

```text
clarify
→ consult
→ recommend
→ refine
```

instead of binary:

```text
clarify vs recommend
```

---

# Grounding & Hallucination Prevention

The LLM NEVER directly emits recommendation objects.

Instead:

```text
LLM chosen_names
    ↓
Deterministic validation
    ↓
Grounded recommendation objects
```

If the LLM outputs invalid products:

* recommendations are discarded safely
* retrieval-ranked fallback recommendations are used

---

# Evaluation Approach

The agent's retrieval quality and orchestrator logic were evaluated rigorously:
* **Custom Test Harness:** Measures `Mean Recall@10` against 10 official benchmark traces.
* **Recall Progression:** Baseline was **35.8%** before prompt tuning and heuristic gating. Following diversification (`max_per_family=3`), temperature pinning (0.2), and anchor injection, final performance reached **75.7%**.
* **Regression Suite:** **108 automated unit tests** protecting orchestrator behaviors, schema compliance, and battery duplicate suppression.

---


## Project structure

```
shl-assessment-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry-point (/health, /chat, lifespan hooks)
│   ├── schemas.py            # Pydantic request/response models
│   ├── config.py             # Environment-based settings
│   ├── catalog_loader.py     # Load & normalise SHL product catalog
│   ├── scoring.py            # Seniority mapping & scoring helpers
│   ├── retrieval.py          # Embedding index & candidate retrieval (sentence-transformers)
│   ├── state_extraction.py   # Extract structured constraints from chat
│   ├── policy.py             # Conversation policy / decision logic (heuristic gates)
│   ├── battery.py            # Diversification and catalog limitation detection
│   ├── llm_client.py         # Anthropic Claude 3.5 Sonnet integration
│   ├── agent.py              # Top-level agent orchestrator
│   └── tests/
│       ├── test_scoring.py   # Unit tests for scoring utilities
│       ├── test_schema.py    # Schema validation tests
│       ├── test_battery.py   # Duplicate family suppression tests
│       └── test_official_traces.py # Recall@10 evaluation harness
├── data/
│   ├── shl_product_catalog.json
│   ├── catalog_embeddings.npy  # Pre-computed vectors
│   └── catalog_metadata.json
├── railway.toml                # Railway deployment config
├── evaluation_results.md       # Trace eval score logs
├── .env.example
├── requirements.txt
└── README.md
```


---

# Important Code Contracts

## LLM Output Contract

The LLM must always return strict JSON:

```json
{
  "mode": "clarify|consult|recommend|refine|compare|refuse",
  "reply": "...",
  "chosen_names": [...]
}
```

---

## Recommendation Grounding

Only validated catalog items can become recommendations.

Example:

```python
recs = build_recommendations_from_names(
    chosen_names,
    candidates,
    top_n=10,
)
```

---

## Forced Deterministic Modes

The policy layer only forces:

* refuse
* compare
* refine

Example:

```python
forced_mode = get_forced_mode(state, last_user)
```

Otherwise:

```text
LLM decides conversational behavior
```

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

*(Note: Environment variables for LLM API keys must be set. Copy `.env.example` to `.env` and fill in your keys).*

---

## Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

---

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hiring graduate software engineers"
      }
    ]
  }'
```

---

# Example Behavior

## Clarify

User:

```text
We're assessing senior leadership candidates.
```

Agent:

```text
Is this for early-stage leadership screening or executive benchmarking?
```

---

## Consult

User:

```text
Hiring a senior Rust engineer for networking infrastructure.
```

Agent:

```text
SHL doesn't currently include a Rust-specific assessment, but live coding and systems-oriented technical evaluation would likely be the closest fit.
```

---

## Recommend

User:

```text
Need a cognitive test under 30 minutes for graduate hiring.
```

Agent:

```text
<grounded shortlist>
```

---

# Engineering Principles

The architecture intentionally prioritizes:

* grounded recommendations
* consultative behavior
* lightweight infrastructure
* minimal hardcoding
* production simplicity
* maintainability

while avoiding:

* giant regex trees
* brittle routing logic
* hallucinated recommendations
* overengineered agent frameworks
