# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project is **Plexus** — a local-first multi-agent system (LangGraph + Ollama + Flask + SQLite). No tests, no linter, no build step.

## Commands

```bash
python main.py                            # Web UI on :5000 + terminal REPL together
python main.py "What is 15% of 847?"      # One-shot CLI (no server)
python main.py --no-server                # REPL only
python main.py --no-cli                   # Web UI only
```

[main.py](main.py) hosts both surfaces in one process: Flask runs in a daemon thread (`start_server_in_thread`), the REPL on the main thread. Both share the same compiled `get_graph()`, the same `LongTermMemory`, and the same `active_conversations` dict. Werkzeug's per-request access logger is silenced so it doesn't garble the REPL.

Config (model, base URL, agent limits) lives in [config.yml](config.yml) and is read once at import time into `SETTINGS` in [config.py](config.py). Restart after editing.

## Architecture

Four nodes. The router fast-paths trivial questions in a single LLM call; complex queries go through plan → execute → verify with a revision loop.

```
START → router ──┬── direct answer ─────────────────────────────────────→ END
                 │                                              REVISE (≤ max_revisions)
                 └── planner → executor (self-loops) → verifier ───────→ planner
                                                            └── APPROVED → END
```

The whole brain lives in [graph.py](graph.py): the `State` TypedDict at the top, four node functions, then `build_graph()` at the bottom. Read top-down.

- **router** — one LLM call. Outputs the literal token `PLAN` to defer to the full pipeline, otherwise outputs the final answer directly. Saves ~10-15s for greetings, small talk, simple Q&A.
- **planner** — handles both initial planning *and* revision. When `state["verification_result"]` is non-empty (verifier said REVISE), it includes the failed plan + verifier feedback as context to produce a corrected plan.
- **executor** — runs ONE plan step using a fresh `create_react_agent` bound with `ALL_TOOLS`. Self-loops via `_executor_router` until `current_step >= len(plan)`, then routes to verifier.
- **verifier** — anti-hallucination check AND final-answer writer (the old `synthesizer` was merged in). On `APPROVED <answer>` it sets `state["response"]`. On `REVISE: …` it loops back to planner up to `SETTINGS.max_revisions` times.

### State contract

Two fields use LangGraph reducers and must NOT be naively overwritten:
- `messages` — `add_messages` reducer (LangGraph built-in).
- `past_steps` — `operator.add` (each executor return appends one `(step, result)` tuple). **Returning `[]` does NOT clear it; it appends nothing.** Past_steps therefore accumulates across revisions — intentional, it gives the verifier full history. To truly reset, start a fresh `thread_id`.

All other fields are plain overwrites. **`token_usage` is a plain dict**, so agents must merge prior totals via `_merge_tokens` before returning, or they wipe earlier counts.

### Human-in-the-loop

The `ask_user` tool in [tools.py](tools.py) calls `langgraph.types.interrupt({"question": ...})`. This is **not a node in the graph** — it pauses execution inside the executor. The graph's `MemorySaver` checkpointer (in `build_graph()`) lets the run resume per `thread_id` with `Command(resume=<answer>)`.

**Critical invariant:** `GraphInterrupt` must propagate out of the executor. Earlier code had a broad `except Exception` that swallowed it and turned the pause into a fake `"Error: ..."` string. The current executor uses `_retry()` which explicitly re-raises `GraphInterrupt` — don't reintroduce a catch-all that hides it.

In [main.py](main.py), `_stream_sse` detects pending interrupts via `g.get_state(cfg).tasks` after the stream loop exits and emits a `human_input_needed` SSE event. `/api/resume` continues the same thread with `Command(resume=answer)`. The CLI's `run_once` does the same with `input()`.

### Per-turn thread_id

Each `/api/chat` call generates a fresh `thread_id = f"t-{uuid.uuid4().hex[:12]}"` and stores it in `latest_thread[conversation_id]`. This is intentional — sharing the conversation_id as thread_id would make the checkpointer carry `past_steps` and other state from earlier turns into the new run. `/api/resume` looks up `latest_thread` to continue the same paused thread.

### Transient-error retries

Every LLM call goes through `_retry(fn, retries=2)` in [graph.py](graph.py). Cloud-hosted Ollama models (`*-cloud`) occasionally return `ResponseError: Internal Server Error (ref: ...)` — those auto-retry with exponential backoff (1.5s → 3s). `GraphInterrupt` is re-raised immediately so HITL still pauses cleanly.

### LLM factory

Single factory `make_llm()` in [config.py](config.py) always returns `ChatOllama` pointed at `SETTINGS.base_url`. Anthropic support was removed; Ollama only. Every agent calls it through `_model(config, ...)` in [graph.py](graph.py) which forwards the optional per-request `model_name` from the UI's model dropdown — so users can switch local models per turn without restarting.

### Conversation context

Each new `/api/chat` prepends prior turns to the planner's `state["input"]` via `_trim_history(messages, HISTORY_TOKEN_BUDGET=2000)` in [main.py](main.py). Token-budgeted (newest-first) — tight enough not to blow context, generous enough that follow-ups like "now multiply that by 2" still work. Bump the budget for big-context models, drop it for 8k models.

### Streaming UX

The Flask SSE stream uses `g.stream(stream_input, config=cfg, stream_mode=["values", "messages"])` to get both per-node state snapshots AND per-LLM-chunk tokens. The UI ([static/index.html](static/index.html)) builds a vertical stack of color-coded agent "trace cards" — each starts in `running` state with a live token stream + animated progress bar + elapsed-time ticker, then flips to `done` with the final summary when its `agent_step` event arrives.

### Memory ([memory.py](memory.py))

Conversations live in **SQLite** at `memory/conversations.db` (`conversations` + `messages` tables). Schema is created on first `_connect()`; any legacy `memory/conversations/*.json` files are auto-imported once. Long-term facts stay in `memory/long_term.json` (small, human-editable).

The `messages.workflow` column stores the JSON array of state-trace snapshots captured during each agent run — that's what re-hydrates the 🔬 View workflow button when a past conversation is reloaded.

Conversation titles are first set from the user's opening message, then **upgraded by an LLM** after the first assistant reply (`_generate_title` in [main.py](main.py)) — adds one small LLM call per new chat. If the call fails, the fallback title sticks.

**Known issue:** every answer over 50 chars is still auto-saved as a long-term "fact" (in `_stream_sse`). This grows unbounded and pollutes future prompts. If you touch the memory flow, consider gating it with an extraction step (e.g. have the verifier emit a `facts_to_remember` field) rather than dumping every Q→A.

### Suppressing the langgraph warning

[main.py](main.py) starts by filtering `LangChainPendingDeprecationWarning` by class. Message-pattern filters and `catch_warnings()` both fail to suppress this one because langchain registers its own filter that overrides standard ones. If you upgrade langchain/langgraph and the warning returns, re-check that the class import path still resolves.

### `run_python` sandbox

The `run_python` tool execs agent-generated code in a restricted namespace with a whitelisted `_ALLOWED_BUILTINS` dict. The `__import__` builtin is allowed (otherwise `import math` would fail). `math` and `statistics` are pre-imported at **module level** (not inside `run_python`) to avoid an edge case where Python's import machinery sees the tainted sandbox builtins. Local-use only — do NOT expose this tool over a public endpoint without tightening the importer.

## File layout

Six Python files total. Each has a single, top-down purpose:

```
config.py    settings (config.yml loader) + make_llm()
graph.py     State + 4 agent nodes + token helpers + build_graph()
tools.py     7 tools: solve_math, run_python, calculator, web_search,
                       write_file, read_file, ask_user
prompts.py   prompt templates for the agents
memory.py    SQLite-backed ConversationMemory + JSON LongTermMemory
main.py      Flask web UI + CLI REPL (one entry point, one process)
```

Plus `static/index.html` (single-file frontend), `config.yml`, `requirements.txt`, `memory/` (SQLite + JSON), `output/` (any saved artifacts).
