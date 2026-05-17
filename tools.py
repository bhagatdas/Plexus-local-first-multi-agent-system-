"""All tools the executor can use.

- solve_math:  SymPy-based math (the anti-hallucination tool)
- run_python:  Sandboxed Python execution
- calculator:  Quick arithmetic
- web_search:  DuckDuckGo lookup
- write_file:  Save content to disk
- read_file:   Read a file
- ask_user:    Pause and ask the human (LangGraph interrupt)
"""

from __future__ import annotations

import contextlib
import io
import math
import statistics
from pathlib import Path

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def solve_math(expression: str) -> str:
    """Solve a math expression with SymPy.

    Use for ANY arithmetic, algebra, or calculus. Examples:
    "0.15 * 847", "sqrt(144)", "solve(2*x + 5 - 15, x)", "diff(x**2, x)".
    """
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    locals_ = {
        "sqrt": sympy.sqrt, "log": sympy.log, "ln": sympy.ln,
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "pi": sympy.pi, "e": sympy.E, "oo": sympy.oo,
        "solve": sympy.solve, "diff": sympy.diff,
        "integrate": sympy.integrate, "limit": sympy.limit,
        "factorial": sympy.factorial, "Abs": sympy.Abs,
        "x": sympy.Symbol("x"), "y": sympy.Symbol("y"), "z": sympy.Symbol("z"),
    }
    parsed = parse_expr(expression, transformations=transformations, local_dict=locals_)
    result = sympy.simplify(parsed)
    if result.is_number:
        val = float(result.evalf())
        return str(int(val)) if val == int(val) else str(val)
    return str(result)


_ALLOWED_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "int": int, "isinstance": isinstance, "len": len, "list": list,
    "map": map, "max": max, "min": min, "pow": pow, "print": print,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type,
    "zip": zip,
    # Allow `import X` inside agent-generated code. This is meant for local
    # use only; do NOT expose this tool over a public endpoint without
    # tightening the importer (e.g. a module allowlist).
    "__import__": __import__,
}


@tool
def run_python(code: str) -> str:
    """Run a Python snippet in a restricted sandbox and return printed output.

    Use for loops, conditionals, or computations the math solver can't do.
    Builtins are restricted; `math` and `statistics` are available.
    Use print() for output.
    """
    namespace = {"__builtins__": _ALLOWED_BUILTINS, "math": math, "statistics": statistics}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, namespace)  # noqa: S102
    except Exception as exc:
        # Return the error as a normal tool result so the ReAct loop can
        # see it and retry. Otherwise it propagates up and kills the run.
        return f"Error: {type(exc).__name__}: {exc}"
    out = buf.getvalue().strip()
    return out or "Code executed (no output)."


@tool
def calculator(expression: str) -> str:
    """Evaluate basic arithmetic like '2 * (3 + 4)'. For complex math use solve_math."""
    allowed = set("0123456789+-*/().% ")
    if not all(ch in allowed for ch in expression):
        return "Error: disallowed characters."
    result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    if isinstance(result, float) and result == int(result):
        return str(int(result))
    return str(result)


@tool
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo). Returns top 5 results with title, snippet, URL."""
    from ddgs import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(
                f"- **{r.get('title', '')}**\n  {r.get('body', '')}\n  Source: {r.get('href', '')}"
            )
    return f"Search results for '{query}':\n\n" + "\n\n".join(results) if results else f"No results for: {query}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Parent dirs are created. Overwrites if exists."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"File written: {path.resolve()} ({path.stat().st_size} bytes)"


@tool
def read_file(file_path: str) -> str:
    """Read a file. Max 50KB."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: file '{file_path}' does not exist."
    if path.stat().st_size > 50_000:
        return f"Error: file too large ({path.stat().st_size} bytes)."
    return path.read_text(encoding="utf-8")


@tool
def ask_user(question: str) -> str:
    """Pause the workflow and ask the human a clarifying question.

    Use ONLY when the step instructs you to ask the user, or when a required
    value is genuinely missing. Pass the full question; the workflow resumes
    with the user's answer as this tool's return value.
    """
    answer = interrupt({"question": question})
    return str(answer) if answer is not None else ""


ALL_TOOLS = [solve_math, run_python, calculator, web_search, write_file, read_file, ask_user]
