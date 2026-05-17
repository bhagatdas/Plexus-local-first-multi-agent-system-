from __future__ import annotations

import operator
import re
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from config import SETTINGS, make_llm
from prompts import EXECUTOR_PROMPT, PLANNER_PROMPT, REVISION_BLOCK, ROUTER_PROMPT, VERIFIER_PROMPT
from tools import ALL_TOOLS


# ---------------------------------------------------------------------------
# State — every node reads and writes this dict
# ---------------------------------------------------------------------------

class State(TypedDict):
    """Shared graph state.

    Two fields use LangGraph reducers and must NOT be naively overwritten:
    - ``messages``   appended via ``add_messages``
    - ``past_steps`` appended via ``operator.add`` (executor adds one per step)

    All other fields are plain overwrites. ``token_usage`` is a plain dict, so
    agents must merge prior totals before returning (see _merge_tokens).
    """

    input: str                                 # original user question
    plan: list[str]                            # ordered steps to execute
    current_step: int                          # index of next step
    past_steps: Annotated[list[tuple[str, str]], operator.add]  # (step, result) pairs
    response: str                              # final answer (set by verifier when approved)
    verification_result: str                   # "" | "APPROVED" | "REVISE: ..."
    revision_count: int                        # how many replan loops have run
    token_usage: dict                          # {agent_name: {"input": N, "output": M, "total": ...}}
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Token-count helpers (used by agents and seeded with the user message in main.py)
# ---------------------------------------------------------------------------

_TIKTOKEN_ENC = None


def _enc():
    """Return a cached cl100k_base tokenizer.

    cl100k_base is GPT-4's encoding. For non-OpenAI models the count is
    usually within ~10% — close enough for a UI counter, and far better
    than zero when the model doesn't report usage metadata.
    """
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        import tiktoken
        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENC


def _count(text: str) -> int:
    return len(_enc().encode(text or ""))


def _tokens_from(response_or_msg, prompt_text: str = "") -> tuple[int, int]:
    """Return (input_tokens, output_tokens) for one LLM call.

    Order of preference:
      1. ``usage_metadata`` (modern LangChain).
      2. Ollama's ``response_metadata`` (``prompt_eval_count`` / ``eval_count``).
      3. Fall back to counting the prompt + response content with tiktoken —
         this guarantees non-zero numbers for cloud-hosted Ollama models that
         don't report token usage.
    """
    usage = getattr(response_or_msg, "usage_metadata", None)
    if usage and (usage.get("input_tokens") or usage.get("output_tokens")):
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    meta = getattr(response_or_msg, "response_metadata", None) or {}
    if meta.get("prompt_eval_count") or meta.get("eval_count"):
        return int(meta.get("prompt_eval_count", 0)), int(meta.get("eval_count", 0))

    content = getattr(response_or_msg, "content", "") or ""
    return _count(prompt_text), _count(content if isinstance(content, str) else str(content))


def _merge_tokens(state: State, agent: str, in_tokens: int, out_tokens: int) -> dict:
    """Return a fresh token_usage dict with this agent's counts added to prior totals."""
    prior = dict(state.get("token_usage") or {})
    existing = prior.get(agent, {"input": 0, "output": 0, "total": 0})
    prior[agent] = {
        "input": existing["input"] + in_tokens,
        "output": existing["output"] + out_tokens,
        "total": existing["input"] + existing["output"] + in_tokens + out_tokens,
    }
    return prior


def _model(config: RunnableConfig, *, temperature: float | None = None):
    return make_llm(
        temperature=temperature,
        model_name=config.get("configurable", {}).get("model_name"),
    )


def _retry(fn, *, retries: int = 2, base_delay: float = 1.5):
    """Run ``fn()`` with exponential backoff on transient errors.

    Re-raises GraphInterrupt immediately (HITL must not retry). All other
    exceptions are retried up to ``retries`` extra times — useful for
    flaky cloud LLM hiccups like ``ResponseError: Internal Server Error``.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except GraphInterrupt:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(base_delay * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _parse_plan(text: str) -> list[str]:
    """Pull numbered steps out of an LLM response."""
    steps: list[str] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for n in range(1, 4):
            if line[:n].isdigit() and len(line) > n and line[n] in ".:)":
                line = line[n + 1:].strip()
                break
        steps.append(line)
    return steps


# ---------------------------------------------------------------------------
# Router — fast-path trivial messages, otherwise hand off to the planner
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|hola|howdy|sup|good\s+(morning|afternoon|evening|night)|"
    r"thanks?|thank\s+you|ty|tysm|bye|goodbye|cya|see\s+ya)"
    r"([\s,!.?]|$)",
    re.IGNORECASE,
)
_INTRO_RE = re.compile(
    r"\b(my\s+name\s+is|i\s+am|i'?m|this\s+is)\s+[A-Za-z][\w'\-\. ]{0,40}\s*[.!?]?\s*$",
    re.IGNORECASE,
)


def _is_trivial_greeting(text: str) -> bool:
    """Detect obvious greetings/introductions deterministically.

    Avoids depending on the router LLM to follow the 'output PLAN or answer'
    contract for the easy cases. main.py prepends prior conversation and
    long-term facts, marking the new message with 'CURRENT QUESTION:' —
    we strip that prefix so we only judge the new turn.
    """
    body = text or ""
    m = re.search(r"CURRENT QUESTION:\s*(.+?)\s*$", body, flags=re.IGNORECASE | re.DOTALL)
    latest = m.group(1).strip() if m else body.strip()
    if not latest or len(latest) > 120:
        return False
    if _GREETING_RE.search(latest):
        return True
    if _INTRO_RE.search(latest) and len(latest.split()) <= 8:
        return True
    return False


def router_node(state: State, config: RunnableConfig) -> dict:
    """Single LLM call that either answers directly OR returns 'PLAN'.

    Skips the planner/executor/verifier loop for greetings, small talk, and
    questions the model can answer confidently in one shot — saving ~10-15s
    for those cases. Anything needing tools, math, or research routes to
    the full pipeline.

    For ultra-obvious greetings/introductions we short-circuit with a
    deterministic regex check, then still ask the model for the *content*
    of the reply — but force the path so it can't accidentally trigger
    the planner and an ask_user loop.
    """
    start = time.time()
    user_text = state.get("input", "")
    force_direct = _is_trivial_greeting(user_text)
    prompt = ROUTER_PROMPT.format(input=user_text)
    response = _retry(lambda: _model(config, temperature=0.3).invoke([HumanMessage(content=prompt)]))
    raw = (response.content or "").strip()
    in_tok, out_tok = _tokens_from(response, prompt)
    elapsed = time.time() - start
    tokens = _merge_tokens(state, "router", in_tok, out_tok)

    # Deterministic override: greetings/introductions always answer directly.
    if force_direct and raw.upper().strip().startswith("PLAN") and len(raw) < 20:
        raw = "Hi! Nice to meet you. What would you like help with today?"

    # The model outputs the literal token 'PLAN' (case-insensitive) when
    # it wants to defer to the full workflow. Anything else IS the answer.
    if raw.upper().strip().startswith("PLAN") and len(raw) < 20:
        return {
            "messages": [AIMessage(content=f"[router] ({elapsed:.1f}s) needs full plan", name="router")],
            "token_usage": tokens,
        }

    return {
        "messages": [
            AIMessage(content=f"[router] ({elapsed:.1f}s) answered directly", name="router"),
            AIMessage(content=raw, name="answer"),
        ],
        "response": raw,
        "token_usage": tokens,
    }


# ---------------------------------------------------------------------------
# Planner — handles both initial planning and revision
# ---------------------------------------------------------------------------

def planner_node(state: State, config: RunnableConfig) -> dict:
    start = time.time()
    verification = state.get("verification_result", "")
    revision_count = state.get("revision_count", 0)
    is_revision = bool(verification) and not verification.upper().startswith("APPROVED")

    if is_revision:
        plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state.get("plan", [])))
        past = "\n".join(
            f"Step {i+1}: {s}\n  Result: {r}"
            for i, (s, r) in enumerate(state.get("past_steps", []))
        ) or "(none)"
        revision_block = REVISION_BLOCK.format(
            plan=plan_text, past_steps=past, verification_result=verification
        )
    else:
        revision_block = ""

    prompt = PLANNER_PROMPT.format(input=state.get("input", ""), revision_block=revision_block)
    response = _retry(lambda: _model(config, temperature=0.0).invoke([HumanMessage(content=prompt)]))
    steps = _parse_plan(response.content or "") or ["Answer the user's question directly"]
    in_tok, out_tok = _tokens_from(response, prompt)

    elapsed = time.time() - start
    label = f"Revision #{revision_count + 1}" if is_revision else "Created plan"
    body = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
    audit = f"[planner] {label} ({len(steps)} steps, {elapsed:.1f}s):\n{body}"

    return {
        "messages": [AIMessage(content=audit, name="planner")],
        "plan": steps,
        "current_step": 0,
        # past_steps is append-only via reducer; clearing requires a fresh
        # graph thread. The verifier re-evaluates everything, so leftover
        # entries from a failed attempt are fine — they show up as context.
        "revision_count": revision_count + (1 if is_revision else 0),
        "verification_result": "",
        "token_usage": _merge_tokens(state, "planner", in_tok, out_tok),
    }


# ---------------------------------------------------------------------------
# Executor — runs ONE plan step with the ReAct + tools loop
# ---------------------------------------------------------------------------

def executor_node(state: State, config: RunnableConfig) -> dict:
    start = time.time()
    plan = state.get("plan", [])
    idx = state.get("current_step", 0)

    if idx >= len(plan):
        return {"messages": [AIMessage(content="[executor] All steps done.", name="executor")]}

    step = plan[idx]
    past = state.get("past_steps", [])
    past_text = (
        "\n".join(f"Step {i+1}: {s}\n  Result: {r}" for i, (s, r) in enumerate(past))
        or "No previous steps."
    )
    prompt = EXECUTOR_PROMPT.format(
        current_step=f"Step {idx + 1}: {step}", past_results=past_text
    )

    react = create_react_agent(model=_model(config), tools=ALL_TOOLS, prompt=prompt)
    # _retry handles transient LLM errors (cloud hiccups); GraphInterrupt
    # is re-raised inside _retry so HITL still pauses cleanly.
    result = _retry(lambda: react.invoke(
        {"messages": [HumanMessage(content=step)]},
        config={"recursion_limit": 40},
    ))

    # The ReAct sub-agent makes many internal calls; we don't see the
    # individual prompts, so the tiktoken fallback for input tokens is
    # approximated as the system prompt size — close enough for a counter.
    in_tok = out_tok = 0
    result_text = ""
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            i, o = _tokens_from(msg, prompt)
            in_tok += i
            out_tok += o
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and (msg.content or "").strip():
            result_text = msg.content.strip()
            break
    if not result_text:
        result_text = f"[executor] No result for: {step}"

    elapsed = time.time() - start
    audit = (
        f"[executor] Step {idx + 1}/{len(plan)} ({elapsed:.1f}s): {step}\n"
        f"  Result: {result_text}"
    )

    return {
        "messages": [AIMessage(content=audit, name="executor")],
        "past_steps": [(step, result_text)],
        "current_step": idx + 1,
        "token_usage": _merge_tokens(state, "executor", in_tok, out_tok),
    }


# ---------------------------------------------------------------------------
# Verifier — checks the work AND writes the final answer when approved
# ---------------------------------------------------------------------------

def verifier_node(state: State, config: RunnableConfig) -> dict:
    start = time.time()
    past_text = "\n".join(
        f"Step {i+1}: {s}\n  Result: {r}"
        for i, (s, r) in enumerate(state.get("past_steps", []))
    )
    prompt = VERIFIER_PROMPT.format(input=state.get("input", ""), past_steps=past_text)
    response = _retry(lambda: _model(config, temperature=0.0).invoke([HumanMessage(content=prompt)]))
    raw = (response.content or "").strip()
    in_tok, out_tok = _tokens_from(response, prompt)
    elapsed = time.time() - start

    # Format: first line is APPROVED / REVISE: ...; if APPROVED, the rest
    # is the final user-facing answer.
    first_line, _, rest = raw.partition("\n")
    upper = first_line.strip().upper()
    tokens = _merge_tokens(state, "verifier", in_tok, out_tok)

    if upper.startswith("APPROVED"):
        final_answer = rest.strip() or "(no answer body produced)"
        return {
            "messages": [
                AIMessage(content=f"[verifier] ({elapsed:.1f}s) APPROVED", name="verifier"),
                AIMessage(content=final_answer, name="answer"),
            ],
            "verification_result": "APPROVED",
            "response": final_answer,
            "token_usage": tokens,
        }

    feedback = raw if upper.startswith("REVISE") else f"REVISE: {raw}"
    return {
        "messages": [AIMessage(content=f"[verifier] ({elapsed:.1f}s) {feedback[:200]}", name="verifier")],
        "verification_result": feedback,
        "token_usage": tokens,
    }


# ---------------------------------------------------------------------------
# Routing + build
# ---------------------------------------------------------------------------

def _router_router(state: State) -> Literal["planner", "__end__"]:
    """If the router already produced a direct answer, skip the rest of the graph."""
    return "__end__" if state.get("response") else "planner"


def _executor_router(state: State) -> Literal["executor", "verifier"]:
    return "executor" if state.get("current_step", 0) < len(state.get("plan", [])) else "verifier"


def _verifier_router(state: State) -> Literal["planner", "__end__"]:
    if state.get("verification_result", "").upper().startswith("APPROVED"):
        return "__end__"
    if state.get("revision_count", 0) >= SETTINGS.max_revisions:
        return "__end__"  # cap hit — give up; whatever's in state['response'] is the best we got
    return "planner"


def build_graph():
    g = StateGraph(State)
    g.add_node("router", router_node)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("verifier", verifier_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _router_router)
    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", _executor_router)
    g.add_conditional_edges("verifier", _verifier_router)

    compiled = g.compile(checkpointer=MemorySaver())
    compiled.step_timeout = None
    return compiled
