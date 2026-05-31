import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "alpha_big2_wrapper.py"
MODULE_SPEC = importlib.util.spec_from_file_location("alpha_big2_wrapper", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules["alpha_big2_wrapper"] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)

PublicStateTracker = MODULE.PublicStateTracker
card_code_to_alpha_id = MODULE.card_code_to_alpha_id
choose_legal_decision = MODULE.choose_legal_decision
choose_forced_max_single = MODULE.choose_forced_max_single
filter_right_one_single_rule = MODULE.filter_right_one_single_rule
align_ml_observation_to_runtime_turn = MODULE.align_ml_observation_to_runtime_turn


def test_card_code_to_alpha_id_matches_alpha_big2_order():
    assert card_code_to_alpha_id("43") == 1
    assert card_code_to_alpha_id("11") == 48
    assert card_code_to_alpha_id("1K") == 44
    assert card_code_to_alpha_id("22") == 51


def test_public_state_tracker_uses_self_right_top_left_mapping():
    tracker = PublicStateTracker()
    tracker.update(
        {
            "game_index": 7,
            "turn": "self",
            "self_hand": [{"code": "43"}, {"code": "11"}],
            "opponents": [
                {"seat": "right", "remaining_count": 11},
                {"seat": "top", "remaining_count": 9},
                {"seat": "left", "remaining_count": 8},
            ],
            "constraint": {
                "last_played_by": "right",
                "last_played_cards": [{"code": "43"}],
                "passes_since_last_play": 2,
            },
            "legal_actions": [],
        }
    )

    state = tracker.to_ml_observation()
    assert state["opponent_counts"] == {"2": 11, "3": 9, "4": 8}
    assert state["last_player"] == 2
    assert state["last_hand"] == [1]
    assert state["passed"] == {"1": False, "2": False, "3": True, "4": True}
    assert state["action_history"] == [
        {
            "player": 2,
            "hand": [1],
            "pass": False,
            "forced_skip": False,
            "control_break": False,
            "passed_snapshot": [False, False, False, False],
        },
        {
            "player": 3,
            "hand": None,
            "pass": True,
            "forced_skip": False,
            "control_break": False,
            "passed_snapshot": [False, False, False, False],
        },
        {
            "player": 4,
            "hand": None,
            "pass": True,
            "forced_skip": False,
            "control_break": False,
            "passed_snapshot": [False, False, True, False],
        },
    ]


def test_choose_legal_decision_prefers_best_supported_action():
    logits = [0.0] * 10
    logits[9] = 0.2
    logits[3] = 0.8
    logits[7] = 1.5

    action_index = {
        (1,): 3,
        (1, 5): 7,
    }

    decision = choose_legal_decision(
        legal_actions=[
            {"action": "pass"},
            {"action": "play", "cards": [{"code": "43"}], "combo_type": "single"},
            {
                "action": "play",
                "cards": [{"code": "43"}, {"code": "44"}],
                "combo_type": "pair",
            },
            {
                "action": "play",
                "cards": [{"code": "43"}] * 13,
                "combo_type": "dragon",
            },
        ],
        logits=logits,
        pass_index=9,
        action_index_from_cards=lambda cards: action_index[tuple(cards)],
    )

    assert decision == {
        "action": "play",
        "card_codes": ["43", "44"],
        "combo_type": "pair",
        "note": "model_choice",
    }


def test_choose_forced_max_single_when_right_has_one_card_left():
    decision = choose_forced_max_single(
        legal_actions=[
            {"action": "pass"},
            {"action": "play", "cards": [{"code": "43"}], "combo_type": "single"},
            {"action": "play", "cards": [{"code": "1K"}], "combo_type": "single"},
            {"action": "play", "cards": [{"code": "22"}], "combo_type": "single"},
        ],
        opponents=[
            {"seat": "left", "remaining_count": 5},
            {"seat": "top", "remaining_count": 7},
            {"seat": "right", "remaining_count": 1},
        ],
    )

    assert decision == {
        "action": "play",
        "card_codes": ["22"],
        "combo_type": "single",
        "note": "override:forced_max_single_right_one_left",
    }


def test_choose_legal_decision_uses_forced_max_single_override():
    logits = [0.0] * 20
    logits[3] = 2.0
    logits[4] = 0.5
    logits[5] = 1.0

    decision = choose_legal_decision(
        legal_actions=[
            {"action": "pass"},
            {"action": "play", "cards": [{"code": "43"}], "combo_type": "single"},
            {"action": "play", "cards": [{"code": "1K"}], "combo_type": "single"},
            {"action": "play", "cards": [{"code": "22"}], "combo_type": "single"},
        ],
        logits=logits,
        pass_index=19,
        action_index_from_cards=lambda cards: {
            (1,): 3,
            (44,): 4,
            (51,): 5,
        }[tuple(cards)],
        opponents=[
            {"seat": "right", "remaining_count": 1},
        ],
    )

    assert decision == {
        "action": "play",
        "card_codes": ["22"],
        "combo_type": "single",
        "note": "override:forced_max_single_right_one_left",
    }


def test_filter_right_one_single_rule_keeps_multi_card_actions():
    filtered = filter_right_one_single_rule(
        legal_actions=[
            {"action": "pass"},
            {"action": "play", "cards": [{"code": "43"}], "combo_type": "single"},
            {"action": "play", "cards": [{"code": "22"}], "combo_type": "single"},
            {
                "action": "play",
                "cards": [{"code": "35"}, {"code": "25"}],
                "combo_type": "pair",
            },
        ],
        opponents=[{"seat": "right", "remaining_count": 1}],
    )

    assert filtered == [
        {"action": "pass"},
        {"action": "play", "cards": [{"code": "22"}], "combo_type": "single"},
        {
            "action": "play",
            "cards": [{"code": "35"}, {"code": "25"}],
            "combo_type": "pair",
        },
    ]


def test_align_ml_observation_to_runtime_turn_prefers_runtime_self_turn():
    aligned = align_ml_observation_to_runtime_turn(
        {"current_player": 2, "my_hand": [1]},
        {"turn": "self"},
    )

    assert aligned["current_player"] == 1


def test_choose_legal_decision_filters_low_single_but_can_choose_pair():
    logits = [0.0] * 20
    logits[3] = 5.0
    logits[5] = 1.0
    logits[8] = 4.0

    decision = choose_legal_decision(
        legal_actions=[
            {"action": "pass"},
            {"action": "play", "cards": [{"code": "43"}], "combo_type": "single"},
            {"action": "play", "cards": [{"code": "22"}], "combo_type": "single"},
            {
                "action": "play",
                "cards": [{"code": "35"}, {"code": "25"}],
                "combo_type": "pair",
            },
        ],
        logits=logits,
        pass_index=19,
        action_index_from_cards=lambda cards: {
            (1,): 3,
            (51,): 5,
            (10, 11): 8,
        }[tuple(cards)],
        opponents=[{"seat": "right", "remaining_count": 1}],
    )

    assert decision == {
        "action": "play",
        "card_codes": ["35", "25"],
        "combo_type": "pair",
        "note": "model_choice",
    }
