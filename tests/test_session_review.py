from big2_vision_agent.session_review import build_rounds, build_session_summary


def test_build_rounds_groups_four_round_results():
    timeline = [
        {"event": "room_snapshot", "room_id": "room-a", "players": [{"actor": "self", "gmoney": 100}]},
        {"event": "round_result", "actor": "self", "score": 6, "remaining_cards": []},
        {"event": "round_result", "actor": "right", "score": -1, "remaining_cards": ["11"]},
        {"event": "round_result", "actor": "top", "score": -2, "remaining_cards": ["12", "13"]},
        {"event": "round_result", "actor": "left", "score": -3, "remaining_cards": ["14", "15", "16"]},
    ]

    rounds = build_rounds(timeline)

    assert len(rounds) == 1
    assert rounds[0]["room_id"] == "room-a"
    assert rounds[0]["self_score"] == 6
    assert rounds[0]["self_won"] is True
    assert rounds[0]["winner_actor"] == "self"


def test_build_session_summary_tracks_bankroll_delta_from_next_room():
    timeline = [
        {
            "event": "room_snapshot",
            "room_id": "room-a",
            "players": [{"actor": "self", "userid": "me", "nickname": "Me", "gmoney": 100}],
        },
        {"event": "round_result", "actor": "self", "score": 6, "remaining_cards": []},
        {"event": "round_result", "actor": "right", "score": -1, "remaining_cards": ["11"]},
        {"event": "round_result", "actor": "top", "score": -2, "remaining_cards": ["12", "13"]},
        {"event": "round_result", "actor": "left", "score": -3, "remaining_cards": ["14", "15", "16"]},
        {
            "event": "room_snapshot",
            "room_id": "room-b",
            "players": [{"actor": "self", "userid": "me", "nickname": "Me", "gmoney": 132}],
        },
        {"event": "round_result", "actor": "self", "score": -2, "remaining_cards": ["31", "32"]},
        {"event": "round_result", "actor": "right", "score": 5, "remaining_cards": []},
        {"event": "round_result", "actor": "top", "score": -1, "remaining_cards": ["12"]},
        {"event": "round_result", "actor": "left", "score": -2, "remaining_cards": ["14", "15"]},
    ]

    summary = build_session_summary(timeline)

    assert summary["session_count"] == 2
    assert summary["round_count"] == 2
    assert summary["total_self_score"] == 4
    assert summary["total_round_wins"] == 1
    assert summary["sessions"][0]["delta_gmoney"] == 32
    assert summary["sessions"][1]["delta_gmoney"] is None


def test_build_session_summary_includes_rounds_before_room_snapshot():
    timeline = [
        {"event": "round_result", "actor": "self", "score": -1, "remaining_cards": ["11"]},
        {"event": "round_result", "actor": "right", "score": 3, "remaining_cards": []},
        {"event": "round_result", "actor": "top", "score": -1, "remaining_cards": ["12"]},
        {"event": "round_result", "actor": "left", "score": -1, "remaining_cards": ["13"]},
        {
            "event": "room_snapshot",
            "room_id": "room-a",
            "players": [{"actor": "self", "gmoney": 100}],
        },
        {"event": "round_result", "actor": "self", "score": 4, "remaining_cards": []},
        {"event": "round_result", "actor": "right", "score": -1, "remaining_cards": ["21"]},
        {"event": "round_result", "actor": "top", "score": -1, "remaining_cards": ["22"]},
        {"event": "round_result", "actor": "left", "score": -2, "remaining_cards": ["23", "24"]},
    ]

    summary = build_session_summary(timeline)

    assert summary["session_count"] == 2
    assert summary["round_count"] == 2
    assert summary["total_self_score"] == 3
    assert summary["sessions"][0]["room_id"] is None
    assert summary["sessions"][0]["round_count"] == 1
