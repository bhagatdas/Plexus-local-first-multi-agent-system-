"""Settings (from config.yml) and the LLM factory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_ollama import ChatOllama


_CONFIG_PATH = Path(__file__).parent / "config.yml"


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    temperature: float
    max_revisions: int    # max verifier → planner loops
    max_steps: int        # graph recursion limit


def _load() -> Settings:
    data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    ollama = data.get("ollama", {}) or {}
    agent = data.get("agent", {}) or {}
    return Settings(
        base_url=ollama.get("base_url", "http://localhost:11434"),
        model=ollama.get("model", "llama3.1"),
        temperature=float(ollama.get("temperature", 0.2)),
        max_revisions=int(agent.get("max_revisions", 3)),
        max_steps=int(agent.get("max_steps", 15)),
    )


SETTINGS = _load()


def make_llm(temperature: float | None = None, model_name: str | None = None) -> ChatOllama:
    """Build a ChatOllama client.

    Args:
        temperature: Per-agent override (verifier uses 0.0, others use the default).
        model_name: Optional per-request override (the UI dropdown passes this).
    """
    return ChatOllama(
        model=model_name or SETTINGS.model,
        base_url=SETTINGS.base_url,
        temperature=SETTINGS.temperature if temperature is None else temperature,
    )
