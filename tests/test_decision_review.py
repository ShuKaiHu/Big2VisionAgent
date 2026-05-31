import json
from pathlib import Path

from big2_vision_agent.decision_review import build_report, render_markdown, write_markdown_report


def test_build_report_matches_action_log_and_model_debug(tmp_path: Path):
    artifact_dir = tmp_path / "autoplay_agent"
    artifact_dir.mkdir()

    action_log = [
        {
            "step": "agent_decision",
            "observation": {
                "turn": "self",
                "hand_count": 5,
                "constraint": {"required_combo_type": "single"},
                "opponents": [{"seat": "right", "remaining_count": 1}],
            },
            "decision": {
                "action": "play",
                "card_codes": ["22"],
                "combo_type": "single",
                "note": "override:forced_max_single_right_one_left",
            },
            "result": {
                "ok": True,
                "reason": None,
                "ws_message": "send 9 22",
            },
        }
    ]
    (artifact_dir / "action_log.json").write_text(json.dumps(action_log), encoding="utf-8")
    (artifact_dir / "model_debug.jsonl").write_text(
        json.dumps(
            {
                "decision": {
                    "action": "play",
                    "card_codes": ["22"],
                    "combo_type": "single",
                    "note": "override:forced_max_single_right_one_left",
                },
                "candidate_scores": [
                    {"card_codes": ["22"], "combo_type": "single", "score": 0.5},
                    {"card_codes": ["1K"], "combo_type": "single", "score": 0.9},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_report(artifact_dir)

    assert report["summary"] == {
        "total_agent_decisions": 1,
        "model_debug_rows": 1,
        "steps_with_model_debug": 1,
        "matching_decisions": 1,
        "action_failures": 0,
        "decision_mismatches": 0,
    }
    assert report["steps"][0]["decision_matches_model"] is True
    assert report["steps"][0]["ws_message"] == "send 9 22"


def test_render_markdown_contains_summary_and_step(tmp_path: Path):
    artifact_dir = tmp_path / "autoplay_agent"
    artifact_dir.mkdir()
    report = {
        "artifact_dir": str(artifact_dir),
        "summary": {
            "total_agent_decisions": 1,
            "model_debug_rows": 1,
            "steps_with_model_debug": 1,
            "matching_decisions": 1,
            "action_failures": 0,
            "decision_mismatches": 0,
        },
        "steps": [
            {
                "index": 0,
                "turn": "self",
                "hand_count": 5,
                "required_combo_type": "single",
                "right_remaining": 1,
                "decision": {"action": "play", "card_codes": ["22"]},
                "result_ok": True,
                "result_reason": None,
                "ws_message": "send 9 22",
                "model_debug_present": True,
                "decision_matches_model": True,
                "model_note": "override:test",
                "candidate_scores": [
                    {"card_codes": ["22"], "combo_type": "single", "score": 0.5},
                ],
            }
        ],
    }

    markdown = render_markdown(report)

    assert "# Model Decision Review" in markdown
    assert "### Step 0" in markdown
    assert "send 9 22" in markdown
    assert "override:test" in markdown


def test_write_markdown_report_creates_default_file(tmp_path: Path):
    artifact_dir = tmp_path / "autoplay_agent"
    artifact_dir.mkdir()
    (artifact_dir / "action_log.json").write_text(
        json.dumps(
            [
                {
                    "step": "agent_decision",
                    "observation": {"turn": "self", "hand_count": 3, "constraint": {}, "opponents": []},
                    "decision": {"action": "pass", "card_codes": [], "combo_type": None},
                    "result": {"ok": True, "reason": None, "ws_message": None},
                }
            ]
        ),
        encoding="utf-8",
    )

    output_path = write_markdown_report(artifact_dir)

    assert output_path == artifact_dir / "decision_review.md"
    assert output_path.exists()


def test_build_report_matches_model_rows_by_observation_instead_of_index(tmp_path: Path):
    artifact_dir = tmp_path / "autoplay_agent"
    artifact_dir.mkdir()

    action_log = [
        {
            "step": "agent_decision",
            "observation": {
                "game_index": 1,
                "trick_index": 1,
                "source_seq": 10,
                "turn": "self",
                "self_hand": [{"code": "22"}],
                "constraint": {"required_combo_type": "single", "last_played_cards": [], "last_played_by": "left", "passes_since_last_play": 0},
                "opponents": [{"seat": "right", "remaining_count": 1}],
            },
            "decision": {"action": "play", "card_codes": ["22"], "combo_type": "single"},
            "result": {"ok": True, "reason": None, "ws_message": "send 9 22"},
        }
    ]
    (artifact_dir / "action_log.json").write_text(json.dumps(action_log), encoding="utf-8")
    (artifact_dir / "model_debug.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "raw_observation": {
                            "game_index": 9,
                            "trick_index": 9,
                            "source_seq": 999,
                            "turn": "self",
                            "self_hand": [{"code": "11"}],
                            "constraint": {"required_combo_type": "single", "last_played_cards": [], "last_played_by": "left", "passes_since_last_play": 0},
                            "opponents": [{"seat": "right", "remaining_count": 1}],
                        },
                        "decision": {"action": "play", "card_codes": ["11"], "combo_type": "single"},
                        "candidate_scores": [],
                    }
                ),
                json.dumps(
                    {
                        "raw_observation": action_log[0]["observation"],
                        "decision": {"action": "play", "card_codes": ["22"], "combo_type": "single"},
                        "candidate_scores": [{"card_codes": ["22"], "combo_type": "single", "score": 0.5}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_report(artifact_dir)

    assert report["summary"]["model_debug_rows"] == 2
    assert report["summary"]["matching_decisions"] == 1
    assert report["summary"]["decision_mismatches"] == 0
    assert report["steps"][0]["decision_matches_model"] is True
