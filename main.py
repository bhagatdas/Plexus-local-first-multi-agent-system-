"""Plexus — local-first multi-agent system. Single entry point.

Usage:
    python main.py                # Web UI on :5000 + terminal REPL
    python main.py "question"     # One-shot CLI (no server)
    python main.py --no-server    # REPL only, no web UI
    python main.py --no-cli       # Web UI only, no terminal

The web and CLI share the same compiled graph and memory. Each
conversation (UI tab or CLI session) gets its own thread_id so the
LangGraph checkpointer keeps them isolated.

Human-in-the-loop: if any agent calls ``ask_user``, the graph pauses.
The web UI shows an amber input card; the CLI prompts on the terminal.
"""

from __future__ import annotations

import warnings

try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass

import json
import logging
import sys
import threading
import time
import traceback
import urllib.request
import uuid

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from flask import Flask, Response, jsonify, request, send_from_directory
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from config import SETTINGS, make_llm
from graph import _count, build_graph
from memory import ConversationMemory, LongTermMemory, list_conversations


# ---------------------------------------------------------------------------
# Shared state — built once, used by both the web routes and the CLI
# ---------------------------------------------------------------------------

_graph = None
long_term_memory = LongTermMemory()
active_conversations: dict[str, ConversationMemory] = {}

# How many tokens of prior conversation to pack into each new turn's prompt.
# Trims newest-first; older turns roll off automatically once we hit the cap.
# Bump if your model has a big context window; lower for tight 8k models.
HISTORY_TOKEN_BUDGET = 2000
# Maps a conversation_id to the LangGraph thread_id used for its most recent
# turn. Each turn gets a fresh thread so the checkpointer state doesn't leak
# (past_steps especially); /api/resume looks this up to continue the SAME
# thread when the user replies to an ask_user prompt.
latest_thread: dict[str, str] = {}


def get_graph():
    """Compile the graph on first call and cache it."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _trim_history(messages: list[dict], max_tokens: int) -> list[dict]:
    """Return the most recent user/assistant messages that fit within max_tokens.

    Walks newest-first, accumulating until the budget is exhausted, then
    reverses for chronological order. Caller controls the budget.
    """
    kept: list[dict] = []
    used = 0
    for m in reversed(messages):
        if m.get("role") not in ("user", "assistant"):
            continue
        cost = _count(f"{m['role']}: {m.get('content', '')}")
        if used + cost > max_tokens:
            break
        kept.append(m)
        used += cost
    kept.reverse()
    return kept


def _generate_title(question: str, answer: str) -> str | None:
    """Ask the model for a 3-5 word title. Used after the first turn so the
    sidebar shows a meaningful name instead of the raw user message.
    Returns None on any failure — caller keeps the existing title.
    """
    prompt = (
        "Write a short, descriptive title (3-5 words) for the chat below. "
        "Output ONLY the title — no quotes, no preface, no punctuation at the end.\n\n"
        f"User: {question[:300]}\n"
        f"Assistant: {answer[:400]}\n"
        "Title:"
    )
    try:
        response = make_llm(temperature=0.3).invoke([HumanMessage(content=prompt)])
        title = (response.content or "").strip().strip('"\'').strip()
        title = title.splitlines()[0].strip() if title else ""
        return title[:60] if title else None
    except Exception:
        return None


def _make_initial_state(user_message: str) -> dict:
    """Build the initial state for a fresh graph run."""
    tok = _count(user_message)
    return {
        "messages": [HumanMessage(content=user_message)],
        "input": user_message,
        "plan": [],
        "current_step": 0,
        "past_steps": [],
        "response": "",
        "verification_result": "",
        "revision_count": 0,
        "token_usage": {"user": {"input": tok, "output": 0, "total": tok}},
    }


# ---------------------------------------------------------------------------
# Web server (Flask)
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static")


def _stream_sse(g, stream_input, cfg, conv, user_text, start_time):
    """Yield SSE events as the graph runs; surface human-input pauses.

    Also collects every state snapshot so we can persist them with the
    final assistant message — that lets the UI replay the workflow
    after a page refresh or when an old conversation is reloaded.
    """
    seen = 0
    final_state = None
    workflow_snapshots: list[dict] = []  # persisted with the final message

    # Multi-mode stream:
    #   "values"   → per-node state snapshots (existing behavior)
    #   "messages" → per-LLM-chunk token streaming for live typing feel
    for mode, payload in g.stream(
        stream_input, config=cfg, stream_mode=["values", "messages"]
    ):
        if mode == "messages":
            chunk, meta = payload
            text = getattr(chunk, "content", "") or ""
            if not isinstance(text, str) or not text:
                continue
            node = (meta or {}).get("langgraph_node") or "assistant"
            yield f"data: {json.dumps({'type': 'token', 'agent': node, 'content': text})}\n\n"
            continue

        # mode == "values"
        event = payload
        final_state = event
        msgs = event.get("messages", [])
        for msg in msgs[seen:]:
            if isinstance(msg, AIMessage) and (msg.content or "").strip():
                yield f"data: {json.dumps({'type': 'agent_step', 'agent': msg.name or 'assistant', 'content': msg.content})}\n\n"
        seen = len(msgs)

        last_node = "start"
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.name:
                last_node = m.name
                break
        snapshot = {
            "node": last_node,
            "ts": round(time.time() - start_time, 2),
            "plan": event.get("plan", []),
            "current_step": event.get("current_step", 0),
            "past_steps": [
                {"step": s[0], "result": s[1]}
                for s in event.get("past_steps", [])
                if isinstance(s, (list, tuple)) and len(s) >= 2
            ],
            "verification_result": event.get("verification_result", ""),
            "revision_count": event.get("revision_count", 0),
            "token_usage": event.get("token_usage", {}),
            "message_count": len(msgs),
        }
        workflow_snapshots.append(snapshot)
        yield f"data: {json.dumps({'type': 'state', **snapshot}, default=str)}\n\n"

    runtime = g.get_state(cfg)
    pending = [iv for task in (runtime.tasks or []) for iv in (getattr(task, "interrupts", None) or [])]
    elapsed = time.time() - start_time

    if pending:
        payload = pending[0].value if hasattr(pending[0], "value") else pending[0]
        question = payload.get("question") if isinstance(payload, dict) else str(payload)
        yield f"data: {json.dumps({'type': 'human_input_needed', 'question': question, 'elapsed': round(elapsed, 1)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    final_answer = (final_state or {}).get("response", "")
    yield f"data: {json.dumps({'type': 'final_answer', 'content': final_answer, 'elapsed': round(elapsed, 1)})}\n\n"

    if conv is not None and final_answer:
        conv.add_message(
            "assistant", final_answer, agent="verifier", workflow=workflow_snapshots
        )
        conv.save()
        # First turn? Upgrade the auto-set title to an LLM-summarized one.
        # ``add_message("user", ...)`` set a fallback title from the raw
        # question; after the assistant's first reply we have enough to do
        # something better. Adds ~1-2s for an extra small LLM call.
        if len(conv.messages) == 2:
            better = _generate_title(user_text, final_answer)
            if better:
                conv.title = better
                conv.save()
        if len(final_answer) > 50:
            long_term_memory.add_fact(f"Q: {user_text[:100]} → A: {final_answer[:200]}")

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    raw = data.get("message", "")
    user_message = "".join(c for c in raw if not (0xD800 <= ord(c) <= 0xDFFF)).strip()
    conversation_id = data.get("conversation_id", "")
    selected_model = data.get("model", "")
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if conversation_id and conversation_id in active_conversations:
        conv = active_conversations[conversation_id]
    else:
        conv = ConversationMemory()
        conversation_id = conv.conversation_id
        active_conversations[conversation_id] = conv
    conv.add_message("user", user_message)

    def generate():
        try:
            g = get_graph()

            # 1. Build conversation context from prior turns in THIS
            #    conversation. Exclude the user message we just appended.
            #    Token-budgeted (newest-first); older turns roll off when
            #    the budget is exhausted.
            prior = _trim_history(conv.get_messages()[:-1], HISTORY_TOKEN_BUDGET)
            context_parts = []
            if prior:
                turns = "\n".join(
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in prior
                )
                context_parts.append(f"PREVIOUS CONVERSATION:\n{turns}")
            memory_context = long_term_memory.get_context_string()
            if memory_context:
                context_parts.append(memory_context)

            if context_parts:
                contextualized = (
                    "\n\n".join(context_parts) + f"\n\nCURRENT QUESTION: {user_message}"
                )
            else:
                contextualized = user_message

            initial = _make_initial_state(user_message)
            initial["input"] = contextualized
            initial["messages"] = [HumanMessage(content=contextualized)]

            # 2. Fresh thread_id per turn so the checkpointer doesn't carry
            #    past_steps (or any other state) from earlier turns.
            thread_id = f"t-{uuid.uuid4().hex[:12]}"
            latest_thread[conversation_id] = thread_id
            configurable = {"thread_id": thread_id}
            if selected_model:
                configurable["model_name"] = selected_model
            cfg = {"recursion_limit": SETTINGS.max_steps, "configurable": configurable}

            yield f"data: {json.dumps({'type': 'conversation_id', 'id': conversation_id})}\n\n"
            yield from _stream_sse(g, initial, cfg, conv, user_message, time.time())
        except Exception as exc:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {exc}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/resume", methods=["POST"])
def resume():
    data = request.json or {}
    conversation_id = (data.get("conversation_id") or "").strip()
    answer = "".join(c for c in data.get("answer", "") if not (0xD800 <= ord(c) <= 0xDFFF)).strip()
    selected_model = data.get("model", "")
    if not conversation_id:
        return jsonify({"error": "Missing conversation_id"}), 400

    conv = active_conversations.get(conversation_id) or ConversationMemory.load(conversation_id)
    active_conversations[conversation_id] = conv
    conv.add_message("user", answer)

    def generate():
        try:
            g = get_graph()
            # Resume the SAME thread the original /api/chat call started —
            # that's where the interrupt is checkpointed.
            thread_id = latest_thread.get(conversation_id, conversation_id)
            configurable = {"thread_id": thread_id}
            if selected_model:
                configurable["model_name"] = selected_model
            cfg = {"recursion_limit": SETTINGS.max_steps, "configurable": configurable}
            yield f"data: {json.dumps({'type': 'conversation_id', 'id': conversation_id})}\n\n"
            yield from _stream_sse(g, Command(resume=answer), cfg, conv, answer, time.time())
        except Exception as exc:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {exc}'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/conversations")
def get_conversations():
    return jsonify(list_conversations())


@app.route("/api/conversations/<conv_id>")
def get_conversation(conv_id):
    conv = ConversationMemory.load(conv_id)
    active_conversations[conv_id] = conv
    return jsonify({"id": conv.conversation_id, "title": conv.title, "messages": conv.get_messages()})


@app.route("/api/memory", methods=["GET"])
def get_memory():
    return jsonify({"facts": long_term_memory.get_facts(), "preferences": long_term_memory.get_preferences()})


@app.route("/api/memory/fact", methods=["POST"])
def add_fact():
    fact = (request.json or {}).get("fact", "").strip()
    if not fact:
        return jsonify({"error": "Empty fact"}), 400
    long_term_memory.add_fact(fact)
    return jsonify({"status": "ok"})


@app.route("/api/memory", methods=["DELETE"])
def clear_memory():
    long_term_memory.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/models")
def get_models():
    try:
        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return jsonify({"models": [m["name"] for m in json.loads(req.read()).get("models", [])]})
    except Exception as e:
        return jsonify({"error": str(e), "models": []}), 500


def start_server_in_thread() -> threading.Thread:
    """Run Flask in a daemon thread so the main thread can host the REPL."""
    # Silence per-request access logs — keeps the REPL output clean.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Terminal REPL
# ---------------------------------------------------------------------------

def _print_msg(msg) -> None:
    speaker = msg.name if isinstance(msg, AIMessage) else "user"
    print(f"\n=== {speaker} ===\n{msg.content}")


def run_once(graph, user_request: str) -> None:
    """Stream one question through the graph; prompt the human if it pauses."""
    print(f"\n>>> Request: {user_request}\n")

    initial = _make_initial_state(user_request)
    cfg = {
        "recursion_limit": SETTINGS.max_steps,
        "configurable": {"thread_id": f"cli-{uuid.uuid4().hex[:8]}"},
    }

    seen = [0]
    final = [None]
    start = time.time()

    def stream(inp):
        for event in graph.stream(inp, config=cfg, stream_mode="values"):
            final[0] = event
            msgs = event.get("messages", [])
            for msg in msgs[seen[0]:]:
                _print_msg(msg)
            seen[0] = len(msgs)

    stream(initial)
    while True:
        state = graph.get_state(cfg)
        pending = [iv for t in (state.tasks or []) for iv in (getattr(t, "interrupts", None) or [])]
        if not pending:
            break
        payload = pending[0].value if hasattr(pending[0], "value") else pending[0]
        question = payload.get("question") if isinstance(payload, dict) else str(payload)
        print(f"\n{'-' * 60}\n  Agent asks: {question}\n{'-' * 60}")
        answer = input(">>> Your answer: ").strip()
        stream(Command(resume=answer))

    print(f"\n{'=' * 60}\n  RUN COMPLETE ({time.time() - start:.1f}s)\n{'=' * 60}")
    if final[0] and final[0].get("response"):
        print(f"\n{final[0]['response']}\n")


def repl(graph) -> None:
    print("\nREPL ready. Type a question (blank line to quit).")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not line:
            return
        run_once(graph, line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _print_banner(server: bool, cli: bool) -> None:
    print("\n" + "=" * 60)
    print("  Plexus — local-first multi-agent system")
    print(f"  Model: {SETTINGS.model}  ({SETTINGS.base_url})")
    if server:
        print(f"  Web:   http://localhost:5000")
    if cli:
        print(f"  CLI:   terminal REPL")
    print("=" * 60)


def main(argv: list[str]) -> int:
    argv = argv[1:]

    # One-shot: any positional argument that isn't a flag.
    positional = [a for a in argv if not a.startswith("--")]
    if positional:
        run_once(get_graph(), " ".join(positional))
        return 0

    with_server = "--no-server" not in argv
    with_cli = "--no-cli" not in argv

    # Build graph eagerly so the first request isn't slow.
    get_graph()
    _print_banner(server=with_server, cli=with_cli)

    if with_server:
        start_server_in_thread()

    if with_cli:
        repl(get_graph())
    elif with_server:
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
