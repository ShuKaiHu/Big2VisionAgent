#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from big2_vision_agent.decision_review import build_report, format_step, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", help="Path to artifacts/<timestamp>/autoplay_agent")
    parser.add_argument("--limit", type=int, default=20, help="Max steps to print")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    parser.add_argument("--stdout", action="store_true", help="Print human-readable summary to stdout")
    parser.add_argument(
        "--output",
        default="decision_review.md",
        help="Markdown output filename relative to artifact_dir, or absolute path",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    report = build_report(artifact_dir)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = artifact_dir / output_path
    output_path.write_text(render_markdown(report, limit=args.limit), encoding="utf-8")
    print(f"Saved decision review to: {output_path}")
    if args.stdout:
        summary = report["summary"]
        print(
            "Summary: "
            f"decisions={summary['total_agent_decisions']} "
            f"model_debug={summary['model_debug_rows']} "
            f"matched={summary['matching_decisions']} "
            f"failures={summary['action_failures']} "
            f"mismatches={summary['decision_mismatches']}"
        )
        print("Steps:")
        for step in report["steps"][: args.limit]:
            print(format_step(step))


if __name__ == "__main__":
    main()
