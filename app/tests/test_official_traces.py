"""
app/tests/test_official_traces.py
==================================
Evaluate the SHL Assessment Recommendation Agent against the 10 official
conversation traces located in GenAI_SampleConversations/.

Trace format (discovered by reading each .md file)
---------------------------------------------------
Each trace is a Markdown file with the following structure:

  ## Conversation

  ### Turn N
  **User**
  > <user message text>

  **Agent**
  <agent reply text>

  | # | Name | Test Type | Keys | Duration | Languages | URL |
  |---|------|-----------|------|----------|-----------|-----|
  | 1 | <name> | <type> | <keys> | <dur> | <langs> | <url> |

  _`end_of_conversation`: **true/false**_

Key observations:
- User messages are inside blockquotes (lines starting with ">")
- Agent recommendations are in markdown tables (| # | Name | ... | URL |)
- The FINAL recommendations in the trace define the expected shortlist
- No separate "persona" / "facts" YAML block — the conversation itself IS the script
- Expected items: names extracted from the last recommendation table in the trace

Simulation strategy
-------------------
1. Parse each trace to extract: (a) ordered user turns, (b) expected item names
   from the last recommendation table.
2. Replay turns sequentially against POST /chat:
   - Use the scripted user turns verbatim where they match the turn number.
   - If the agent asks a clarifying question NOT covered by remaining scripted
     turns, reply with "I don't have a preference on that."
3. Stop when: end_of_conversation=True OR recommendations non-empty OR 8 turns.
4. Compute Recall@10 = |expected ∩ recommended| / |expected|
5. Print per-trace table + mean Recall@10.
6. For traces scoring < 0.5, print the full turn-by-turn transcript.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
TRACES_DIR = REPO_ROOT / "GenAI_SampleConversations"
CATALOG_PATH = REPO_ROOT / "data" / "catalog_metadata.json"

BASE_URL = "http://localhost:8000"
TURN_CAP = 8

# ---------------------------------------------------------------------------
# Load catalog metadata for URL validation
# ---------------------------------------------------------------------------

def _load_catalog_urls() -> set[str]:
    """Return the set of all URLs present in catalog_metadata.json."""
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return {item["url"].rstrip("/") for item in data}
    except Exception:
        return set()

CATALOG_URLS: set[str] = _load_catalog_urls()

# ---------------------------------------------------------------------------
# Trace parser
# ---------------------------------------------------------------------------

@dataclass
class OfficialTrace:
    trace_id: str                        # e.g. "C1"
    scripted_user_turns: list[str]       # ordered user messages from the trace
    expected_names: list[str]            # names from the final recommendation table
    expected_urls: list[str]             # URLs from the final recommendation table


def _extract_user_turns(text: str) -> list[str]:
    """
    Extract user messages in order.
    User messages appear as blockquote lines under **User** headings.
    """
    turns: list[str] = []
    # Split on turn headings
    turn_blocks = re.split(r"###\s+Turn\s+\d+", text)
    for block in turn_blocks[1:]:  # skip preamble before Turn 1
        # Check if this block has a **User** section
        user_match = re.search(r"\*\*User\*\*\s*(.*?)\*\*Agent\*\*", block, re.DOTALL)
        if not user_match:
            continue
        user_section = user_match.group(1)
        # Extract blockquote lines
        bq_lines = re.findall(r"^>\s?(.*)$", user_section, re.MULTILINE)
        # Join multi-line blockquotes; strip empty lines
        combined = " ".join(line.strip() for line in bq_lines if line.strip())
        if combined:
            turns.append(combined)
    return turns


def _extract_last_recommendation_table(text: str) -> tuple[list[str], list[str]]:
    """
    Find the LAST markdown table in the trace that has a URL column
    and return (names, urls).
    """
    # Find all table blocks (consecutive | lines)
    table_pattern = re.compile(
        r"(\|[^\n]+\|\n)+"  # one or more pipe-delimited rows
    )
    all_tables = list(table_pattern.finditer(text))

    for match in reversed(all_tables):
        table_text = match.group(0)
        rows = [r.strip() for r in table_text.strip().splitlines() if r.strip()]
        # Skip separator rows (only dashes/pipes/spaces)
        data_rows = [
            r for r in rows
            if not re.match(r"^[\|\-\s]+$", r)
        ]
        if not data_rows:
            continue
        # Header row
        header_row = data_rows[0]
        headers = [h.strip().lower() for h in header_row.split("|") if h.strip()]
        if "url" not in headers or "name" not in headers:
            continue

        name_idx = headers.index("name")
        url_idx = headers.index("url")

        names: list[str] = []
        urls: list[str] = []

        for row in data_rows[1:]:  # skip header
            cols = [c.strip() for c in row.split("|")]
            # Remove empty leading/trailing from split
            cols = [c for c in cols if c or True]  # keep all
            # split produces ['', col1, col2, ..., ''] for | col | col |
            # remove leading/trailing empties
            while cols and not cols[0]:
                cols.pop(0)
            while cols and not cols[-1]:
                cols.pop()

            if len(cols) <= max(name_idx, url_idx):
                continue

            name = cols[name_idx].strip()
            raw_url = cols[url_idx].strip()
            # Strip markdown link syntax <url> or [text](url)
            url = re.sub(r"^<(.+)>$", r"\1", raw_url)
            url = re.sub(r"\[.*?\]\((.+?)\)", r"\1", url)
            url = url.strip().rstrip("/")

            if name and url.startswith("http"):
                names.append(name)
                urls.append(url)

        if names:
            return names, urls

    return [], []


def _parse_trace_file(path: Path) -> OfficialTrace:
    text = path.read_text(encoding="utf-8")
    trace_id = path.stem  # "C1", "C2", ...
    user_turns = _extract_user_turns(text)
    expected_names, expected_urls = _extract_last_recommendation_table(text)
    return OfficialTrace(
        trace_id=trace_id,
        scripted_user_turns=user_turns,
        expected_names=expected_names,
        expected_urls=expected_urls,
    )


def load_all_traces() -> list[OfficialTrace]:
    """Load and parse all .md files in GenAI_SampleConversations/."""
    traces = []
    for path in sorted(TRACES_DIR.glob("*.md")):
        try:
            trace = _parse_trace_file(path)
            traces.append(trace)
        except Exception as exc:
            print(f"[WARN] Failed to parse {path.name}: {exc}", file=sys.stderr)
    return traces


# ---------------------------------------------------------------------------
# LLM-based simulated user (falls back to scripted turns, then default reply)
# ---------------------------------------------------------------------------

def _simulated_user_reply(
    agent_question: str,
    remaining_turns: list[str],
    turn_idx: int,
) -> str:
    """
    Return what the simulated user would say in response to the agent.

    Priority:
    1. If there are scripted turns left, use the next one.
    2. Otherwise reply with a no-preference default.
    """
    if remaining_turns:
        return remaining_turns.pop(0)
    return "I don't have a preference on that."


# ---------------------------------------------------------------------------
# Recall@10 computation
# ---------------------------------------------------------------------------

def _recall_at_10(
    recommended: list[dict],
    expected_names: list[str],
) -> float:
    """
    Compute Recall@10.

    Match is case-insensitive substring: expected name appears anywhere in
    recommended name, or vice versa (handles minor wording differences).
    """
    if not expected_names:
        return 0.0

    rec_names = [
        (r.get("name") or r.get("product_name") or "").lower()
        for r in recommended[:10]
    ]

    hits = 0
    for exp in expected_names:
        exp_l = exp.lower()
        matched = any(
            exp_l in rec_n or rec_n in exp_l or _token_overlap(exp_l, rec_n) >= 0.6
            for rec_n in rec_names
        )
        if matched:
            hits += 1

    return hits / len(expected_names)


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings."""
    ta = set(re.split(r"\W+", a)) - {"", "the", "a", "an", "and", "or"}
    tb = set(re.split(r"\W+", b)) - {"", "the", "a", "an", "and", "or"}
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Schema violation check
# ---------------------------------------------------------------------------

def _check_schema(body: dict, catalog_urls: set[str]) -> list[str]:
    """Return list of schema violation strings (empty if clean)."""
    violations: list[str] = []

    recs = body.get("recommendations", [])
    if len(recs) > 10:
        violations.append(f"recommendations length {len(recs)} > 10")

    for i, rec in enumerate(recs):
        if not isinstance(rec, dict):
            violations.append(f"rec[{i}] not a dict")
            continue
        name = rec.get("name") or rec.get("product_name") or ""
        if not name:
            violations.append(f"rec[{i}] missing name")
        url = rec.get("url", "")
        if not url.startswith("http"):
            violations.append(f"rec[{i}] bad url: {url!r}")
        elif catalog_urls and url.rstrip("/") not in catalog_urls:
            violations.append(f"rec[{i}] url not in catalog: {url!r}")
        if "test_type" in rec and not rec["test_type"]:
            violations.append(f"rec[{i}] empty test_type")

    if not isinstance(body.get("reply"), str) or not body["reply"]:
        violations.append("reply is empty or not a string")
    if not isinstance(body.get("end_of_conversation"), bool):
        violations.append("end_of_conversation is not bool")

    return violations


# ---------------------------------------------------------------------------
# Core replay engine
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn_num: int
    user_msg: str
    agent_reply: str
    has_recs: bool
    rec_count: int
    end_of_conv: bool
    latency_ms: float


@dataclass
class TraceResult:
    trace_id: str
    turns_taken: int
    recommended_names: list[str]
    expected_names: list[str]
    recall: float
    schema_violations: list[str]
    transcript: list[TurnRecord]
    final_recs: list[dict]
    error: str = ""


def replay_trace_against_server(trace: OfficialTrace) -> TraceResult:
    """
    Replay an OfficialTrace against the live server at BASE_URL.
    Returns a TraceResult with recall, schema issues, and transcript.
    """
    messages: list[dict] = []
    final_body: dict = {}
    transcript: list[TurnRecord] = []

    scripted = list(trace.scripted_user_turns)  # mutable copy
    turn_num = 0
    violations: list[str] = []
    # Throttle delay: 13s between requests to stay within 5 RPM (= 1 req/12s)
    # Can be overridden via THROTTLE_DELAY_S env var (set to 0 to disable)
    _throttle = float(os.environ.get("THROTTLE_DELAY_S", "13"))
    _last_req_time: float = 0.0

    while turn_num < TURN_CAP:
        # Determine the user message for this turn
        if turn_num == 0:
            # First turn: always use the first scripted message
            if not scripted:
                break
            user_msg = scripted.pop(0)
        else:
            # Subsequent turns: use the next scripted turn if available,
            # otherwise a no-preference reply
            user_msg = scripted.pop(0) if scripted else "I don't have a preference on that."

        messages.append({"role": "user", "content": user_msg})

        # Rate-limit: ensure at least _throttle seconds between API calls
        if _throttle > 0:
            _elapsed_since_last = time.monotonic() - _last_req_time
            _sleep_needed = _throttle - _elapsed_since_last
            if _sleep_needed > 0 and _last_req_time > 0:
                time.sleep(_sleep_needed)
        _last_req_time = time.monotonic()

        t0 = time.monotonic()
        try:
            resp = requests.post(
                f"{BASE_URL}/chat",
                json={"messages": messages},
                timeout=200,   # 200s: accommodates 30+60+90s Gemini 429 retry
            )
            latency_ms = (time.monotonic() - t0) * 1000
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            return TraceResult(
                trace_id=trace.trace_id,
                turns_taken=turn_num + 1,
                recommended_names=[],
                expected_names=trace.expected_names,
                recall=0.0,
                schema_violations=[f"HTTP error: {exc}"],
                transcript=transcript,
                final_recs=[],
                error=str(exc),
            )

        final_body = body
        agent_reply = body.get("reply", "")
        recs = body.get("recommendations", [])
        eoc = bool(body.get("end_of_conversation", False))

        # Schema check per turn
        turn_violations = _check_schema(body, CATALOG_URLS)
        violations.extend(turn_violations)

        transcript.append(TurnRecord(
            turn_num=turn_num + 1,
            user_msg=user_msg,
            agent_reply=agent_reply,
            has_recs=bool(recs),
            rec_count=len(recs),
            end_of_conv=eoc,
            latency_ms=latency_ms,
        ))

        messages.append({"role": "assistant", "content": agent_reply})
        turn_num += 1

        # Stop conditions
        if eoc or recs:
            break

        if turn_num >= TURN_CAP:
            break

    final_recs = final_body.get("recommendations", [])
    recommended_names = [
        r.get("name") or r.get("product_name") or "" for r in final_recs
    ]
    recall = _recall_at_10(final_recs, trace.expected_names)

    return TraceResult(
        trace_id=trace.trace_id,
        turns_taken=turn_num,
        recommended_names=recommended_names,
        expected_names=trace.expected_names,
        recall=recall,
        schema_violations=list(dict.fromkeys(violations)),  # dedup
        transcript=transcript,
        final_recs=final_recs,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_names(names: list[str], max_len: int = 60) -> str:
    if not names:
        return "(none)"
    joined = "; ".join(names)
    return joined[:max_len] + "…" if len(joined) > max_len else joined


def _print_transcript(result: TraceResult) -> None:
    print(f"\n{'─'*70}")
    print(f"  TRANSCRIPT — {result.trace_id}  (Recall={result.recall:.2f})")
    print(f"{'─'*70}")
    for t in result.transcript:
        print(f"\n  [Turn {t.turn_num}] USER: {t.user_msg}")
        # Truncate long agent replies for readability
        reply_lines = t.agent_reply.splitlines()
        short_reply = "\n           ".join(reply_lines[:6])
        if len(reply_lines) > 6:
            short_reply += f"\n           ... ({len(reply_lines) - 6} more lines)"
        print(f"           AGENT: {short_reply}")
        tags = []
        if t.has_recs:
            tags.append(f"recs={t.rec_count}")
        if t.end_of_conv:
            tags.append("END")
        tags.append(f"{t.latency_ms:.0f}ms")
        print(f"           [{', '.join(tags)}]")
    print()
    print(f"  Expected : {'; '.join(result.expected_names)}")
    print(f"  Got      : {'; '.join(result.recommended_names)}")
    if result.schema_violations:
        print(f"  Violations: {'; '.join(result.schema_violations)}")
    print()


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    """Run all 10 traces and print the evaluation report."""
    traces = load_all_traces()
    if not traces:
        print("ERROR: No traces found in", TRACES_DIR)
        return

    print(f"\n{'='*80}")
    print(f"  SHL Agent — Official Trace Evaluation ({len(traces)} traces)")
    print(f"  Server: {BASE_URL}")
    print(f"{'='*80}\n")

    # Check server health first
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        assert health.json().get("status") == "ok"
        print("  [✓] Server health OK\n")
    except Exception as exc:
        print(f"  [✗] Server not reachable at {BASE_URL}: {exc}")
        print("  Start with: uvicorn app.main:app --reload")
        return

    results: list[TraceResult] = []
    low_recall_results: list[TraceResult] = []

    for trace in traces:
        print(f"  Running {trace.trace_id}… ", end="", flush=True)
        result = replay_trace_against_server(trace)
        results.append(result)
        status = "✓" if result.recall >= 0.5 else "✗"
        print(
            f"{status}  recall={result.recall:.2f}  turns={result.turns_taken}  "
            f"recs={len(result.recommended_names)}"
        )
        if result.recall < 0.5:
            low_recall_results.append(result)

    # ── Per-trace table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  PER-TRACE RESULTS")
    print(f"{'='*80}")
    col_w = [6, 6, 8, 48, 48, 8]
    header = (
        f"{'Trace':6s} {'Turns':6s} {'Recall':8s} "
        f"{'Recommended (top 3)':48s} {'Expected (top 3)':48s} {'Issues':8s}"
    )
    print(header)
    print("─" * sum(col_w + [2] * len(col_w)))

    for r in results:
        rec_short = _fmt_names(r.recommended_names[:3], 46)
        exp_short = _fmt_names(r.expected_names[:3], 46)
        issues = "NONE" if not r.schema_violations else str(len(r.schema_violations))
        print(
            f"{r.trace_id:6s} {r.turns_taken:6d} {r.recall:8.2f} "
            f"{rec_short:48s} {exp_short:48s} {issues:8s}"
        )
        if r.error:
            print(f"       ERROR: {r.error}")

    # ── Mean Recall@10 ───────────────────────────────────────────────────────
    mean_recall = sum(r.recall for r in results) / len(results) if results else 0.0
    print(f"\n{'─'*80}")
    print(f"  Mean Recall@10 across {len(results)} traces: {mean_recall:.4f}  ({mean_recall*100:.1f}%)")
    print(f"{'─'*80}")

    # ── Schema violations summary ────────────────────────────────────────────
    all_violations = [
        (r.trace_id, v)
        for r in results
        for v in r.schema_violations
    ]
    if all_violations:
        print(f"\n  SCHEMA VIOLATIONS ({len(all_violations)} total):")
        for trace_id, v in all_violations:
            print(f"    [{trace_id}] {v}")
    else:
        print("\n  No schema violations detected.")

    # ── Full transcript for low-recall traces ────────────────────────────────
    if low_recall_results:
        print(f"\n{'='*80}")
        print(f"  DETAILED TRANSCRIPTS — traces scoring < 0.5 ({len(low_recall_results)} traces)")
        print(f"{'='*80}")
        for r in low_recall_results:
            _print_transcript(r)
    else:
        print("\n  All traces scored >= 0.5 — no detailed transcripts needed.")

    print(f"\n{'='*80}")
    print("  Evaluation complete.")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# pytest integration — one test per trace + overall mean guard
# ---------------------------------------------------------------------------

def _all_traces_cached() -> list[OfficialTrace]:
    """Cached loader so we only parse once during a pytest run."""
    if not hasattr(_all_traces_cached, "_cache"):
        _all_traces_cached._cache = load_all_traces()
    return _all_traces_cached._cache


@pytest.fixture(scope="session")
def server_url() -> str:
    """Check the server is reachable and return its URL."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        assert r.json().get("status") == "ok"
        return BASE_URL
    except Exception as exc:
        pytest.skip(f"Server not reachable at {BASE_URL}: {exc}")


@pytest.mark.parametrize(
    "trace",
    _all_traces_cached(),
    ids=[t.trace_id for t in _all_traces_cached()],
)
def test_official_trace(trace: OfficialTrace, server_url: str) -> None:
    """
    Replay one official trace and assert Recall@10 > 0 (any hit is a pass).
    The full recall numbers are printed by run_evaluation() for analysis.
    """
    result = replay_trace_against_server(trace)

    # Print compact summary
    print(
        f"\n[{trace.trace_id}] recall={result.recall:.2f}  "
        f"turns={result.turns_taken}  recs={result.recommended_names[:3]}"
    )

    if result.recall < 0.5:
        _print_transcript(result)

    # Soft assertion: at least one expected item found
    assert result.recall > 0.0 or not trace.expected_names, (
        f"Trace {trace.trace_id}: Recall@10=0 — none of the expected items found.\n"
        f"Expected: {trace.expected_names}\n"
        f"Got     : {result.recommended_names}"
    )


def test_mean_recall(server_url: str) -> None:
    """Assert that mean Recall@10 across all official traces >= 0.3."""
    traces = _all_traces_cached()
    results = [replay_trace_against_server(t) for t in traces]
    mean = sum(r.recall for r in results) / len(results) if results else 0.0
    print(f"\nMean Recall@10: {mean:.4f}")
    # Diagnostic threshold — adjust as the agent improves
    assert mean >= 0.3, f"Mean Recall@10={mean:.4f} is below 0.3 threshold"


# ---------------------------------------------------------------------------
# Script entry point — run evaluation directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load .env if dotenv is available
    try:
        from dotenv import load_dotenv
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass

    run_evaluation()
