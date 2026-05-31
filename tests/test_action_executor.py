from unittest.mock import AsyncMock

import pytest

from big2_vision_agent.action_executor import _card_click_points, execute_agent_decision, execute_packet_decision
from big2_vision_agent.agent_schema import AgentDecision


@pytest.mark.asyncio
async def test_execute_agent_decision_detects_unconfirmed_play(monkeypatch):
    state_before = {
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [{"sprite_frame": "c43", "center": {"x": 1, "y": 2}}],
        "action_buttons": {"play": {"active": True, "center": {"x": 10, "y": 20}}},
    }
    state_after_select = {
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [{"sprite_frame": "c43", "selected": True, "center": {"x": 1, "y": 2}}],
        "action_buttons": {"play": {"active": True, "center": {"x": 10, "y": 20}}},
    }
    state_after_play = {
        "my_hand_count": 5,
        "turn": "self",
        "system_messages": {
            "card_type_error": True,
            "no_bigger_card": False,
            "cant_lock": False,
        },
    }

    click_mock = AsyncMock()
    read_mock = AsyncMock(side_effect=[state_after_select, state_after_play, state_after_play])

    monkeypatch.setattr("big2_vision_agent.action_executor.click_design_point", click_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr(
        "big2_vision_agent.action_executor.toggle_my_card_by_sprite",
        AsyncMock(return_value={"invoked": False, "reason": "unsupported_test_page"}),
    )

    result = await execute_agent_decision(
        page=_FakePage(),
        state=state_before,
        decision=AgentDecision(action="play", card_codes=["43"], combo_type="single"),
    )

    assert result["ok"] is False
    assert result["reason"] == "play_not_confirmed"


def test_card_click_points_bias_to_upper_visible_region():
    points = _card_click_points(
        [
            {
                "center": {"x": 100, "y": 200},
                "box": {"left": 60, "right": 140, "top": 120, "width": 80, "height": 160},
            },
            {
                "center": {"x": 160, "y": 200},
                "box": {"left": 120, "right": 200, "top": 120, "width": 80, "height": 160},
            }
        ],
        0,
    )

    assert points
    assert points[0]["x"] >= 100
    assert 120 < points[0]["y"] < 180


@pytest.mark.asyncio
async def test_execute_agent_decision_clears_stale_selection_via_cancel(monkeypatch):
    state_before = {
        "my_selected_count": 1,
        "my_cards": [{"sprite_frame": "c43", "center": {"x": 100, "y": 200}}],
        "action_buttons": {
            "cancel": {"active": True, "center": {"x": 10, "y": 20}},
            "pass": {"active": True, "center": {"x": 30, "y": 40}},
        },
    }
    state_after_cancel = {
        "my_selected_count": 0,
        "my_cards": [{"sprite_frame": "c43", "center": {"x": 100, "y": 200}}],
        "action_buttons": {
            "cancel": {"active": False, "center": {"x": 10, "y": 20}},
            "pass": {"active": True, "center": {"x": 30, "y": 40}},
        },
    }
    state_after_pass = {
        "my_selected_count": 0,
        "action_buttons": {"pass": {"active": True, "center": {"x": 30, "y": 40}}},
    }

    click_mock = AsyncMock()
    read_mock = AsyncMock(side_effect=[state_after_cancel, state_after_pass])

    monkeypatch.setattr("big2_vision_agent.action_executor.click_design_point", click_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr(
        "big2_vision_agent.action_executor.toggle_my_card_by_sprite",
        AsyncMock(return_value={"invoked": False, "reason": "unsupported_test_page"}),
    )

    result = await execute_agent_decision(
        page=object(),
        state=state_before,
        decision=AgentDecision(action="pass"),
    )

    assert result["ok"] is True
    assert click_mock.await_args_list[0].args[1:] == (10, 20)
    assert click_mock.await_args_list[1].args[1:] == (30, 40)


@pytest.mark.asyncio
async def test_execute_agent_decision_rejects_selection_mismatch(monkeypatch):
    state_before = {
        "my_selected_count": 0,
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [
            {"sprite_frame": "c43", "center": {"x": 100, "y": 200}},
            {"sprite_frame": "c44", "center": {"x": 120, "y": 200}},
        ],
        "action_buttons": {
            "play": {"active": True, "center": {"x": 10, "y": 20}},
            "cancel": {"active": True, "center": {"x": 30, "y": 40}},
        },
    }
    state_after_select = {
        "my_selected_count": 1,
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [
            {"sprite_frame": "c43", "selected": False, "center": {"x": 100, "y": 200}},
            {"sprite_frame": "c44", "selected": True, "center": {"x": 120, "y": 200}},
        ],
        "action_buttons": {
            "play": {"active": True, "center": {"x": 10, "y": 20}},
            "cancel": {"active": True, "center": {"x": 30, "y": 40}},
        },
    }
    state_after_clear = {
        "my_selected_count": 0,
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [
            {"sprite_frame": "c43", "selected": False, "center": {"x": 100, "y": 200}},
            {"sprite_frame": "c44", "selected": False, "center": {"x": 120, "y": 200}},
        ],
        "action_buttons": {
            "play": {"active": True, "center": {"x": 10, "y": 20}},
            "cancel": {"active": False, "center": {"x": 30, "y": 40}},
        },
    }

    click_mock = AsyncMock()
    read_mock = AsyncMock(side_effect=[state_after_select, state_after_clear])

    monkeypatch.setattr("big2_vision_agent.action_executor.click_design_point", click_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr(
        "big2_vision_agent.action_executor.toggle_my_card_by_sprite",
        AsyncMock(return_value={"invoked": False, "reason": "unsupported_test_page"}),
    )

    result = await execute_agent_decision(
        page=object(),
        state=state_before,
        decision=AgentDecision(action="play", card_codes=["43"], combo_type="single"),
    )

    assert result["ok"] is False
    assert result["reason"] == "target_click_failed"
    assert result["card_code"] == "43"
    assert len(click_mock.await_args_list) == 1


class _FakePage:
    async def wait_for_timeout(self, _ms):
        return None


@pytest.mark.asyncio
async def test_execute_packet_decision_accepts_success_when_turn_moves_away(monkeypatch):
    state_before = {
        "turn": "self",
        "my_clock_active": True,
        "my_hand_count": 5,
        "my_cards": [{"sprite_frame": "c43"}, {"sprite_frame": "c44"}],
        "action_buttons": {
            "pass": {"active": True},
            "play": {"active": True},
        },
    }
    state_after = {
        "turn": "right",
        "my_clock_active": False,
        "my_hand_count": 5,
        "my_cards": [{"sprite_frame": "c43"}, {"sprite_frame": "c44"}],
        "action_buttons": {
            "pass": {"active": True},
            "play": {"active": True},
        },
        "system_messages": {
            "card_type_error": True,
            "no_bigger_card": True,
            "cant_lock": True,
        },
    }

    read_mock = AsyncMock(side_effect=[state_before, state_after])
    send_mock = AsyncMock(return_value=True)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.ws_send_raw", send_mock)

    result = await execute_packet_decision(
        page=_FakePage(),
        state=state_before,
        decision=AgentDecision(action="play", card_codes=["43"], combo_type="single"),
    )

    assert result["ok"] is True
    assert result["reason"] is None
    assert result["state_before"]["turn"] == "self"
    assert result["confirmation_states"][-1]["turn"] == "right"


@pytest.mark.asyncio
async def test_execute_packet_decision_accepts_delayed_success(monkeypatch):
    state_before = {
        "turn": "self",
        "my_clock_active": True,
        "my_hand_count": 5,
        "my_cards": [{"sprite_frame": "c43"}, {"sprite_frame": "c44"}],
        "action_buttons": {
            "pass": {"active": True},
            "play": {"active": True},
        },
    }
    state_still_pending = {
        **state_before,
        "system_messages": {
            "card_type_error": False,
            "no_bigger_card": False,
            "cant_lock": False,
        },
    }
    state_after = {
        **state_before,
        "turn": "right",
        "my_clock_active": False,
        "my_hand_count": 4,
        "my_cards": [{"sprite_frame": "c44"}],
    }

    read_mock = AsyncMock(side_effect=[state_before, state_still_pending, state_still_pending, state_after])
    send_mock = AsyncMock(return_value=True)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.ws_send_raw", send_mock)

    result = await execute_packet_decision(
        page=_FakePage(),
        state=state_before,
        decision=AgentDecision(action="play", card_codes=["43"], combo_type="single"),
    )

    assert result["ok"] is True
    assert result["reason"] is None
    assert len(result["confirmation_states"]) == 4


@pytest.mark.asyncio
async def test_execute_packet_decision_reports_server_rejection(monkeypatch):
    state_before = {
        "turn": "self",
        "my_clock_active": True,
        "my_hand_count": 5,
        "my_cards": [{"sprite_frame": "c43"}, {"sprite_frame": "c44"}],
        "action_buttons": {
            "pass": {"active": True},
            "play": {"active": True},
        },
        "system_messages": {
            "card_type_error": False,
            "no_bigger_card": False,
            "cant_lock": False,
        },
    }
    state_rejected = {
        **state_before,
        "system_messages": {
            "card_type_error": True,
            "no_bigger_card": False,
            "cant_lock": False,
        },
    }

    read_mock = AsyncMock(side_effect=[state_before, state_rejected])
    send_mock = AsyncMock(return_value=True)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.ws_send_raw", send_mock)

    result = await execute_packet_decision(
        page=_FakePage(),
        state=state_before,
        decision=AgentDecision(action="play", card_codes=["43"], combo_type="single"),
    )

    assert result["ok"] is False
    assert result["reason"] == "play_rejected_card_type_error"
    assert result["confirmation_states"][-1]["system_messages"]["card_type_error"] is True


@pytest.mark.asyncio
async def test_execute_packet_decision_ignores_stale_rejection_message(monkeypatch):
    state_before = {
        "turn": "self",
        "my_clock_active": True,
        "my_hand_count": 5,
        "my_cards": [{"sprite_frame": "c43"}, {"sprite_frame": "c44"}],
        "action_buttons": {
            "pass": {"active": True},
            "play": {"active": True},
        },
        "system_messages": {
            "card_type_error": True,
            "no_bigger_card": False,
            "cant_lock": False,
        },
    }
    state_pending = {
        **state_before,
    }
    state_after = {
        **state_before,
        "turn": "right",
        "my_clock_active": False,
        "my_hand_count": 4,
        "my_cards": [{"sprite_frame": "c44"}],
    }

    read_mock = AsyncMock(side_effect=[state_before, state_pending, state_after])
    send_mock = AsyncMock(return_value=True)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.ws_send_raw", send_mock)

    result = await execute_packet_decision(
        page=_FakePage(),
        state=state_before,
        decision=AgentDecision(action="play", card_codes=["43"], combo_type="single"),
    )

    assert result["ok"] is True
    assert result["reason"] is None


@pytest.mark.asyncio
async def test_execute_packet_decision_aborts_if_state_changed_before_send(monkeypatch):
    stale_state = {
        "turn": "top",
        "my_clock_active": False,
        "my_hand_count": 5,
        "my_cards": [{"sprite_frame": "c43"}],
        "action_buttons": {
            "pass": {"active": False},
            "play": {"active": False},
        },
    }
    read_mock = AsyncMock(return_value=stale_state)
    send_mock = AsyncMock(return_value=True)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.ws_send_raw", send_mock)

    result = await execute_packet_decision(
        page=_FakePage(),
        state={"turn": "self", "my_clock_active": True, "action_buttons": {"play": {"active": True}}},
        decision=AgentDecision(action="play", card_codes=["43"], combo_type="single"),
    )

    assert result["ok"] is False
    assert result["reason"] == "state_changed_before_send"
    send_mock.assert_not_awaited()
