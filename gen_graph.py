"""Render the compiled LangGraph workflow to a Mermaid file and a PNG.

Run:  python gen_graph.py

Outputs:
  output/workflow.mmd   — Mermaid source (paste into any Mermaid renderer)
  output/workflow.png   — Rendered PNG (uses mermaid.ink; needs internet)
"""
from __future__ import annotations

import os
import sys

from graph import build_graph


def main() -> int:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    print("Building graph...")
    g = build_graph().get_graph()

    mmd_path = os.path.join(out_dir, "workflow.mmd")
    png_path = os.path.join(out_dir, "workflow.png")
    ascii_path = os.path.join(out_dir, "workflow.txt")

    mermaid = g.draw_mermaid()
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid)
    print(f"  Mermaid -> {mmd_path}")

    try:
        ascii_art = g.draw_ascii()
        with open(ascii_path, "w", encoding="utf-8") as f:
            f.write(ascii_art)
        print(f"  ASCII   -> {ascii_path}")
    except Exception as exc:
        print(f"  ASCII   -> skipped ({type(exc).__name__}: {exc})")

    try:
        png_bytes = g.draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"  PNG     -> {png_path}")
    except Exception as exc:
        print(f"  PNG     -> skipped ({type(exc).__name__}: {exc})")
        print("           (PNG rendering needs internet — Mermaid source still saved.)")

    print("\n--- Mermaid source ---\n")
    print(mermaid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
