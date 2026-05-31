import json
from pathlib import Path

from big2_vision_agent.training_export import (
    append_live_training_corpus,
    build_training_rows,
    render_summary,
    write_training_export,
)


def test_training_export_attaches_round_score_and_tags(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts" / "20260528-010203" / "autoplay_agent"
    artifact_dir.mkdir(parents=True)
    action_log = [
        {
            "step": "agent_decision",
            "observation": {
                "game_index": 1,
                "trick_index": 1,
                "source_seq": 10,
                "turn": "self",
                "self_hand": [
                    {"code": "24"},
                    {"code": "25"},
                    {"code": "26"},
                    {"code": "17"},
                    {"code": "48"},
                    {"code": "38"},
                ],
                "hand_count": 6,
                "constraint": {
                    "required_combo_type": "single",
                    "last_played_cards": [{"code": "13"}],
                    "last_played_by": "left",
                    "passes_since_last_play": 0,
                },
                "opponents": [{"seat": "right", "remaining_count": 8}],
            },
            "decision": {"action": "play", "card_codes": ["26"], "combo_type": "single"},
            "result": {"ok": True, "reason": None},
        }
    ]
    model_row = {
        "ckpt_path": "/tmp/model.pt",
        "agent_type": "model",
        "device": "cpu",
        "raw_observation": action_log[0]["observation"],
        "observation_key": {
            "game_index": 1,
            "trick_index": 1,
            "source_seq": 10,
            "turn": "self",
            "self_hand_codes": ["24", "25", "26", "17", "48", "38"],
            "required_combo_type": "single",
            "last_played_codes": ["13"],
            "last_played_by": "left",
            "passes_since_last_play": 0,
            "opponents": [{"seat": "right", "remaining_count": 8}],
        },
        "ml_public_state": {"my_hand": [7, 8, 9, 19, 20, 21]},
        "candidate_scores": [
            {"action": "play", "combo_type": "single", "card_codes": ["26"], "score": 4.0},
            {"action": "play", "combo_type": "single", "card_codes": ["48"], "score": 3.0},
        ],
        "decision": {"action": "play", "card_codes": ["26"], "combo_type": "single"},
    }
    timeline = [
        {"event": "room_snapshot", "room_id": "abc"},
        {"event": "round_result", "actor": "top", "score": 6, "remaining_cards": []},
        {"event": "round_result", "actor": "left", "score": -2, "remaining_cards": ["11", "12"]},
        {"event": "round_result", "actor": "self", "score": -3, "remaining_cards": ["18", "19", "1T"]},
        {"event": "round_result", "actor": "right", "score": -1, "remaining_cards": ["21"]},
    ]
    (artifact_dir / "action_log.json").write_text(json.dumps(action_log), encoding="utf-8")
    (artifact_dir / "model_debug.jsonl").write_text(json.dumps(model_row) + "\n", encoding="utf-8")
    (artifact_dir / "game_timeline.json").write_text(json.dumps(timeline), encoding="utf-8")

    rows = build_training_rows(artifact_dir)

    assert rows[0]["artifact_id"] == "20260528-010203"
    assert rows[0]["decision_id"] == "20260528-010203:0"
    assert rows[0]["round_self_score"] == -3
    assert rows[0]["round_remaining_cards_by_actor"] == {
        "top": [],
        "left": ["11", "12"],
        "self": ["18", "19", "1T"],
        "right": ["21"],
    }
    assert rows[0]["decision_matches_model"] is True
    assert "single_breaks_straight" in rows[0]["tags"]

    dataset_path, summary_path = write_training_export(artifact_dir)
    assert dataset_path.exists()
    assert summary_path.exists()
    assert "`single_breaks_straight`: `1`" in render_summary(rows)

    corpus_path = tmp_path / "data" / "live_training_corpus.jsonl"
    append_live_training_corpus(artifact_dir, rows, corpus_path=corpus_path)
    append_live_training_corpus(artifact_dir, rows, corpus_path=corpus_path)
    assert len(corpus_path.read_text(encoding="utf-8").splitlines()) == 1


def test_training_export_does_not_tag_forced_max_single_as_lowest_error(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts" / "20260528-010204" / "autoplay_agent"
    artifact_dir.mkdir(parents=True)
    observation = {
        "game_index": 1,
        "trick_index": 1,
        "source_seq": 10,
        "turn": "self",
        "self_hand": [{"code": "48"}, {"code": "27"}],
        "hand_count": 2,
        "constraint": {
            "required_combo_type": None,
            "last_played_cards": [],
            "last_played_by": None,
            "passes_since_last_play": 3,
        },
        "opponents": [{"seat": "right", "remaining_count": 1}],
    }
    decision = {
        "action": "play",
        "card_codes": ["48"],
        "combo_type": "single",
        "note": "override:forced_max_single_right_one_left",
    }
    model_row = {
        "raw_observation": observation,
        "observation_key": {
            "game_index": 1,
            "trick_index": 1,
            "source_seq": 10,
            "turn": "self",
            "self_hand_codes": ["48", "27"],
            "required_combo_type": None,
            "last_played_codes": [],
            "last_played_by": None,
            "passes_since_last_play": 3,
            "opponents": [{"seat": "right", "remaining_count": 1}],
        },
        "candidate_scores": [
            {"action": "play", "combo_type": "single", "card_codes": ["48"], "score": 0.5},
            {"action": "play", "combo_type": "single", "card_codes": ["27"], "score": 0.1},
        ],
        "decision": decision,
    }
    (artifact_dir / "action_log.json").write_text(
        json.dumps(
            [{"step": "agent_decision", "observation": observation, "decision": decision, "result": {"ok": True}}],
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model_debug.jsonl").write_text(json.dumps(model_row) + "\n", encoding="utf-8")
    (artifact_dir / "game_timeline.json").write_text(json.dumps([]), encoding="utf-8")

    rows = build_training_rows(artifact_dir)

    assert "endgame_single_not_lowest" not in rows[0]["tags"]


def test_training_export_tags_right_one_single_not_max(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts" / "20260528-010205" / "autoplay_agent"
    artifact_dir.mkdir(parents=True)
    observation = {
        "game_index": 1,
        "trick_index": 1,
        "source_seq": 10,
        "turn": "self",
        "self_hand": [{"code": "43"}, {"code": "22"}],
        "hand_count": 2,
        "constraint": {
            "required_combo_type": None,
            "last_played_cards": [],
            "last_played_by": None,
            "passes_since_last_play": 3,
        },
        "opponents": [{"seat": "right", "remaining_count": 1}],
    }
    decision = {"action": "play", "card_codes": ["43"], "combo_type": "single"}
    model_row = {
        "raw_observation": observation,
        "observation_key": {
            "game_index": 1,
            "trick_index": 1,
            "source_seq": 10,
            "turn": "self",
            "self_hand_codes": ["43", "22"],
            "required_combo_type": None,
            "last_played_codes": [],
            "last_played_by": None,
            "passes_since_last_play": 3,
            "opponents": [{"seat": "right", "remaining_count": 1}],
        },
        "candidate_scores": [
            {"action": "play", "combo_type": "single", "card_codes": ["43"], "score": 2.0},
            {"action": "play", "combo_type": "single", "card_codes": ["22"], "score": 0.1},
        ],
        "decision": decision,
    }
    (artifact_dir / "action_log.json").write_text(
        json.dumps(
            [{"step": "agent_decision", "observation": observation, "decision": decision, "result": {"ok": True}}],
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model_debug.jsonl").write_text(json.dumps(model_row) + "\n", encoding="utf-8")
    (artifact_dir / "game_timeline.json").write_text(json.dumps([]), encoding="utf-8")

    rows = build_training_rows(artifact_dir)

    assert "right_one_single_not_max" in rows[0]["tags"]


def test_training_export_tags_inference_error_fallback(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts" / "20260528-010206" / "autoplay_agent"
    artifact_dir.mkdir(parents=True)
    observation = {
        "game_index": 1,
        "trick_index": 1,
        "source_seq": 10,
        "turn": "self",
        "self_hand": [{"code": "43"}],
        "hand_count": 1,
        "constraint": {
            "required_combo_type": None,
            "last_played_cards": [],
            "last_played_by": None,
            "passes_since_last_play": 0,
        },
        "opponents": [],
    }
    decision = {
        "action": "pass",
        "card_codes": [],
        "combo_type": None,
        "note": "fallback:inference_error",
    }
    model_row = {
        "raw_observation": observation,
        "observation_key": {
            "game_index": 1,
            "trick_index": 1,
            "source_seq": 10,
            "turn": "self",
            "self_hand_codes": ["43"],
            "required_combo_type": None,
            "last_played_codes": [],
            "last_played_by": None,
            "passes_since_last_play": 0,
            "opponents": [],
        },
        "candidate_scores": [],
        "decision": decision,
    }
    (artifact_dir / "action_log.json").write_text(
        json.dumps(
            [{"step": "agent_decision", "observation": observation, "decision": decision, "result": {"ok": True}}],
        ),
        encoding="utf-8",
    )
    (artifact_dir / "model_debug.jsonl").write_text(json.dumps(model_row) + "\n", encoding="utf-8")
    (artifact_dir / "game_timeline.json").write_text(json.dumps([]), encoding="utf-8")

    rows = build_training_rows(artifact_dir)

    assert "fallback_inference_error" in rows[0]["tags"]
