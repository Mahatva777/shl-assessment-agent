"""
Top-level agent orchestrator for the SHL Assessment Agent.

Responsibilities:
    - Receive the conversation history from the ``/chat`` endpoint.
    - Orchestrate the pipeline: state extraction → retrieval → scoring →
      policy → LLM reply generation.
    - Return a ``ChatResponse`` to the caller.

TODO: implement run()
"""

from __future__ import annotations
