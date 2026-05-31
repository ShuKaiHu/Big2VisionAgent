from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def decision_key(decision: dict[str, Any] | None) -> tuple:
    decision = decision or {}
    return (
        decision.get("action"),
        tuple(decision.get("card_codes", []) or []),
        decision.get("combo_type"),
    )


def observation_key(observation: dict[str, Any] | None) -> tuple:
    observation = observation or {}
    constraint = observation.get("constraint", {}) or {}
    return (
        observation.get("game_index"),
        observation.get("trick_index"),
        observation.get("source_seq"),
        observation.get("turn"),
        tuple(card.get("code") for card in observation.get("self_hand", []) if isinstance(card, dict)),
        constraint.get("required_combo_type"),
        tuple(card.get("code") for card in constraint.get("last_played_cards", []) if isinstance(card, dict)),
        constraint.get("last_played_by"),
        constraint.get("passes_since_last_play"),
        tuple(
            (opp.get("seat"), opp.get("remaining_count"))
            for opp in observation.get("opponents", [])
            if isinstance(opp, dict)
        ),
    )


def summarize_step(
    index: int,
    action_row: dict[str, Any],
    model_row: dict[str, Any] | None,
) -> dict[str, Any]:
    observation = action_row.get("observation", {}) or {}
    decision = action_row.get("decision", {}) or {}
    result = action_row.get("result", {}) or {}

    summary = {
        "index": index,
        "turn": observation.get("turn"),
        "hand_count": observation.get("hand_count"),
        "required_combo_type": (observation.get("constraint", {}) or {}).get("required_combo_type"),
        "right_remaining": next(
            (
                opp.get("remaining_count")
                for opp in observation.get("opponents", [])
                if isinstance(opp, dict) and opp.get("seat") == "right"
            ),
            None,
        ),
        "decision": decision,
        "result_ok": result.get("ok"),
        "result_reason": result.get("reason"),
        "ws_message": result.get("ws_message"),
        "model_debug_present": model_row is not None,
        "model_decision": model_row.get("decision") if model_row else None,
        "decision_matches_model": model_row is not None and decision_key(decision) == decision_key(model_row.get("decision")),
        "model_note": (model_row or {}).get("decision", {}).get("note"),
        "candidate_scores": (model_row or {}).get("candidate_scores", []),
    }
    return summary


def build_report(artifact_dir: Path) -> dict[str, Any]:
    action_log_path = artifact_dir / "action_log.json"
    if not action_log_path.exists():
        raise FileNotFoundError(f"Missing action log: {action_log_path}")

    action_rows = load_json(action_log_path)
    model_rows = load_jsonl(artifact_dir / "model_debug.jsonl")

    action_steps = [row for row in action_rows if row.get("step") == "agent_decision"]
    model_rows_by_observation: dict[tuple, deque[dict[str, Any]]] = defaultdict(deque)
    fallback_model_rows: deque[dict[str, Any]] = deque()
    for model_row in model_rows:
        raw_observation = model_row.get("raw_observation")
        if raw_observation:
            model_rows_by_observation[observation_key(raw_observation)].append(model_row)
        else:
            fallback_model_rows.append(model_row)

    summaries = []
    for index, action_row in enumerate(action_steps):
        obs_key = observation_key(action_row.get("observation"))
        candidates = model_rows_by_observation.get(obs_key, deque())
        model_row = None
        if candidates:
            expected_decision_key = decision_key(action_row.get("decision"))
            matched_index = next(
                (
                    candidate_index
                    for candidate_index, candidate in enumerate(candidates)
                    if decision_key(candidate.get("decision")) == expected_decision_key
                ),
                None,
            )
            if matched_index is None:
                model_row = candidates.popleft()
            else:
                model_row = candidates[matched_index]
                del candidates[matched_index]
        elif fallback_model_rows:
            model_row = fallback_model_rows.popleft()
        summaries.append(summarize_step(index, action_row, model_row))

    matched = sum(1 for row in summaries if row["decision_matches_model"])
    with_model = sum(1 for row in summaries if row["model_debug_present"])
    failures = [row for row in summaries if row["result_reason"]]
    mismatches = [row for row in summaries if row["model_debug_present"] and not row["decision_matches_model"]]

    return {
        "artifact_dir": str(artifact_dir),
        "steps": summaries,
        "summary": {
            "total_agent_decisions": len(summaries),
            "model_debug_rows": len(model_rows),
            "steps_with_model_debug": with_model,
            "matching_decisions": matched,
            "action_failures": len(failures),
            "decision_mismatches": len(mismatches),
        },
    }


def format_step(step: dict[str, Any]) -> str:
    parts = [
        f"#{step['index']}",
        f"turn={step['turn']}",
        f"hand={step['hand_count']}",
        f"req={step['required_combo_type']}",
        f"right={step['right_remaining']}",
        f"decision={step['decision'].get('action')}:{step['decision'].get('card_codes', [])}",
        f"ok={step['result_ok']}",
    ]
    if step["result_reason"]:
        parts.append(f"reason={step['result_reason']}")
    if step["model_debug_present"]:
        parts.append(f"match={step['decision_matches_model']}")
        if step["model_note"]:
            parts.append(f"note={step['model_note']}")
        top = step["candidate_scores"][:3]
        if top:
            top_text = ", ".join(
                f"{item.get('card_codes', [])}/{item.get('combo_type')}={item.get('score'):.3f}"
                for item in top
                if isinstance(item.get("score"), (int, float))
            )
            parts.append(f"top3=[{top_text}]")
    return " | ".join(parts)


def render_markdown(report: dict[str, Any], limit: int | None = None) -> str:
    summary = report["summary"]
    steps = report["steps"] if limit is None else report["steps"][:limit]
    lines = [
        "# Model Decision Review",
        "",
        f"- Artifact: `{report['artifact_dir']}`",
        f"- Agent decisions: `{summary['total_agent_decisions']}`",
        f"- Model debug rows: `{summary['model_debug_rows']}`",
        f"- Steps with model debug: `{summary['steps_with_model_debug']}`",
        f"- Matching decisions: `{summary['matching_decisions']}`",
        f"- Action failures: `{summary['action_failures']}`",
        f"- Decision mismatches: `{summary['decision_mismatches']}`",
        "",
        "## Steps",
        "",
    ]
    for step in steps:
        lines.extend(
            [
                f"### Step {step['index']}",
                "",
                f"- Turn: `{step['turn']}`",
                f"- Hand count: `{step['hand_count']}`",
                f"- Required combo: `{step['required_combo_type']}`",
                f"- Right remaining: `{step['right_remaining']}`",
                f"- Decision: `{step['decision'].get('action')}` `{step['decision'].get('card_codes', [])}`",
                f"- Result: `ok={step['result_ok']}` `reason={step['result_reason']}`",
                f"- WS message: `{step['ws_message']}`",
            ]
        )
        if step["model_debug_present"]:
            lines.extend(
                [
                    f"- Decision matches model debug: `{step['decision_matches_model']}`",
                    f"- Model note: `{step['model_note']}`",
                ]
            )
            top = step["candidate_scores"][:5]
            if top:
                lines.append("- Top candidates:")
                for item in top:
                    score = item.get("score")
                    score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                    lines.append(
                        f"  - `{item.get('card_codes', [])}` `{item.get('combo_type')}` score=`{score_text}`"
                    )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(
    artifact_dir: Path,
    output_name: str = "decision_review.md",
    limit: int = 20,
) -> Path:
    report = build_report(artifact_dir)
    output_path = Path(output_name)
    if not output_path.is_absolute():
        output_path = artifact_dir / output_path
    output_path.write_text(render_markdown(report, limit=limit), encoding="utf-8")
    return output_path
