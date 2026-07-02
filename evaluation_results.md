# SHL Agent — Official Trace Evaluation Results

**Model**: `claude-sonnet-4-5` (Anthropic paid)  
**Traces**: 10 × `GenAI_SampleConversations/C*.md`  
**Run time**: ~60 seconds (no throttle needed)  
**Constraint**: Measurement only — `agent.py` not modified

---

## Per-Trace Results

| Trace | Turns | Recall@10 | Recommended (got) | Expected | Notes |
|---|---|---|---|---|---|
| **C1** | 2 | 0.33 | OPQ Leadership Report ✓; Verify Numerical ✗ | OPQ32r; UCF Report 2.0; OPQ Leadership | Clarified purpose correctly; returned wrong cluster |
| **C2** | 2 | 0.00 | Verify G+ only | Smart Interview Live Coding; Linux; Networking; Verify G+; OPQ32r | Recommended just 1 item; Rust gap not resolved to full battery |
| **C3** | 1 | **0.75** | Contact Center Call Sim ✓; Entry Level CS ✓; SVAR English ✗ | SVAR Spoken English (US); Contact Center Sim; Entry Level CS | Missing SVAR Spoken English specifically |
| **C4** | 1 | 0.20 | Verify Numerical Ability ✗; Financial Accounting ✓ | Verify Interactive Numerical; Financial Accounting; Basic Stats; Graduate Scenarios; OPQ32r | Wrong numerical variant (legacy vs. Interactive/New) |
| **C5** | 2 | 0.40 | Sales Transform IC ✓; OPQ MQ Sales ✓; wrong manager variants ✗ | GSA; GSA Dev Report; OPQ32r; OPQ MQ Sales; Sales Transform IC | Missing Global Skills Assessment + GSA Dev Report; manager variants hallucinated |
| **C6** | 1 | **1.00** | Safety & Dependability 8.0 ✓; WHS ✓ | Safety & Dependability 8.0; WHS | Perfect ✅ |
| **C7** | 1 | 0.00 | SVAR Spanish; Written Spanish; Verify Technical | HIPAA Security; Medical Terminology; MS Word 365 Essentials; DSI; OPQ32r | Misread intent — focused on language assessment, missed HIPAA + medical knowledge tests |
| **C8** | 1 | 0.40 | MS Excel ✓; MS Word ✓; Verify General Ability Screen ✗ | MS Excel 365; MS Word 365; MS Excel; MS Word; OPQ32r | Missing the 365 simulation variants; no OPQ32r |
| **C9** | 1 | 0.00 | Java 8 ✗; Java Frameworks ✗; Verify Inductive ✗; OPQ Manager ✗ | Java Advanced (New); Spring; SQL; AWS; Docker; Verify G+; OPQ32r | Wrong Java variants (legacy Java 8 vs. Core Java Advanced Level New); missing SQL/AWS/Docker |
| **C10** | 1 | 0.50 | Verify G+ ✓; Graduate Scenarios ✓; MQ MQM5 ✗ | Verify G+; Graduate Scenarios | Correct on 2/2 expected; MQM5 is noise |

---

## Summary

```
Mean Recall@10:  0.3583  (35.8%)
No schema violations detected
```

| Score band | Traces |
|---|---|
| ≥ 0.75 (good) | C3 (0.75), C6 (1.00) |
| 0.4–0.74 (partial) | C1, C5, C8, C10 |
| < 0.4 (failing) | C2, C4, C7, C9 |

---

## Root Cause Analysis

### 🔴 Wrong Assessment Variant (C4, C9)
The retrieval layer returns **legacy product names** instead of the `(New)` catalog variants:
- `Verify - Numerical Ability` → should be `SHL Verify Interactive – Numerical Reasoning`
- `Java 8 (New)` / `Java Frameworks (New)` → should be `Core Java (Advanced Level) (New)` / `Spring (New)`
- `Verify - Inductive Reasoning (2014)` → should be `SHL Verify Interactive G+`

**Cause**: Embeddings for old and new variants are similar; retrieval doesn't prefer the "(New)" versions.

### 🔴 Misread Primary Intent (C7)
The agent latched onto "Spanish" and "bilingual" instead of the core ask (HIPAA knowledge + medical terminology + MS Word). Recommended language assessments when the trace expected knowledge tests.

### 🟡 Missing Depth — Too Few Items (C2)
For the Rust engineer trace, the agent returned only 1 item (`Verify G+`) instead of the 5-item battery expected (Smart Interview Live Coding + Linux + Networking + Verify G+ + OPQ32r). The agent correctly identified the catalog gap for Rust but didn't fill the shortlist with adjacent applicable tests.

### 🟡 Missing Skill Development Tools (C5)
`Global Skills Assessment` + `Global Skills Development Report` were completely absent. The re-skill / talent-audit framing doesn't surface these via semantic search on the current embeddings.

### 🟡 Simulation Variants Not Preferred (C8)
When "simulation" is not explicitly requested, the agent returns knowledge-only (`MS Excel (New)`) instead of the 365 simulation variants (`Microsoft Excel 365 (New)`). C8's user *does* explicitly request simulations in Turn 2 — but since the agent returned a recommendation on Turn 1, the evaluation stops there.

### 🟡 OPQ32r Consistently Missing
`Occupational Personality Questionnaire OPQ32r` is expected as a default add in most traces (C1, C4, C5, C7, C8, C9) but the agent rarely surfaces it unless the user explicitly asks for personality. The system prompt doesn't instruct the agent to include OPQ32r as a baseline personality measure.

---

## Recommended Fixes (Measurement Only — Not Applied Yet)

| Fix | Expected impact | Traces |
|---|---|---|
| Boost `(New)` / `Interactive` variants in retrieval scoring | +0.2–0.4 recall | C4, C9 |
| Add "include OPQ32r by default for selection" rule to system prompt | +0.1 recall | C1, C4, C5, C7, C8, C9 |
| Expand `_build_document_text` to weight `(New)` in embeddings | Reduces legacy variant surfacing | C4, C9 |
| Add "when re-skilling, recommend GSA + dev report" rule | +0.2 recall | C5 |
| Multi-turn stopping condition: wait for simulation confirmation | +0.4 recall | C8 |
| Fix C7 intent detection: "HIPAA" → knowledge test, not language | +1.0 recall | C7 |
