from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rounds(timeline: list[dict[str, object]]) -> list[dict[str, object]]:
    rounds: list[dict[str, object]] = []
    current_room_id: str | None = None
    bucket: list[dict[str, object]] = []

    for item in timeline:
        event = item.get("event")
        if event == "room_snapshot":
            room_id = item.get("room_id")
            if isinstance(room_id, str):
                current_room_id = room_id
            continue

        if event != "round_result":
            continue

        bucket.append(
            {
                "seq": item.get("seq"),
                "ts": item.get("ts"),
                "actor": item.get("actor"),
                "score": item.get("score"),
                "remaining_count": len(item.get("remaining_cards") or []),
                "remaining_cards": list(item.get("remaining_cards") or []),
                "remaining_decoded_cards": list(item.get("remaining_decoded_cards") or []),
                "room_id": current_room_id,
            }
        )
        actors = {entry.get("actor") for entry in bucket}
        if len(bucket) >= 4 or actors >= {"self", "right", "top", "left"}:
            ordered = sorted(
                bucket,
                key=lambda entry: (
                    entry.get("score") if isinstance(entry.get("score"), int) else -10**9,
                    -(entry.get("remaining_count") or 0),
                ),
                reverse=True,
            )
            self_entry = next((entry for entry in bucket if entry.get("actor") == "self"), None)
            rounds.append(
                {
                    "room_id": current_room_id,
                    "seq_end": bucket[-1].get("seq"),
                    "entries": list(bucket),
                    "winner_actor": ordered[0].get("actor") if ordered else None,
                    "winner_score": ordered[0].get("score") if ordered else None,
                    "self_score": self_entry.get("score") if self_entry else None,
                    "self_remaining_count": self_entry.get("remaining_count") if self_entry else None,
                    "self_won": bool(self_entry and (self_entry.get("remaining_count") == 0)),
                }
            )
            bucket = []

    return rounds


def build_session_summary(timeline: list[dict[str, object]]) -> dict[str, object]:
    snapshots: list[dict[str, object]] = [
        item for item in timeline if item.get("event") == "room_snapshot"
    ]
    rounds = build_rounds(timeline)

    session_order: list[str] = []
    session_starts: dict[str, dict[str, object]] = {}
    for snapshot in snapshots:
        room_id = snapshot.get("room_id")
        if not isinstance(room_id, str):
            continue
        if room_id not in session_order:
            session_order.append(room_id)
            session_starts[room_id] = snapshot

    sessions: list[dict[str, object]] = []
    total_self_score = 0
    total_round_wins = 0
    total_rounds = 0

    session_keys: list[str | None] = []
    if any(item.get("room_id") is None for item in rounds):
        session_keys.append(None)
    session_keys.extend(session_order)

    for index, room_id in enumerate(session_keys):
        session_rounds = [item for item in rounds if item.get("room_id") == room_id]
        if not session_rounds:
            continue

        start_snapshot = session_starts.get(room_id) or {}
        start_self = _self_player(start_snapshot.get("players"))
        next_key = session_keys[index + 1] if index + 1 < len(session_keys) else None
        next_start_snapshot = session_starts.get(next_key) if next_key is not None else None
        end_self = _self_player((next_start_snapshot or {}).get("players"))

        self_scores = [
            entry.get("self_score")
            for entry in session_rounds
            if isinstance(entry.get("self_score"), int)
        ]
        self_score_sum = sum(self_scores)
        self_round_wins = sum(1 for entry in session_rounds if entry.get("self_won"))
        total_self_score += self_score_sum
        total_round_wins += self_round_wins
        total_rounds += len(session_rounds)

        start_gmoney = start_self.get("gmoney") if isinstance(start_self.get("gmoney"), int) else None
        end_gmoney = end_self.get("gmoney") if isinstance(end_self.get("gmoney"), int) else None
        delta_gmoney = (
            end_gmoney - start_gmoney
            if isinstance(start_gmoney, int) and isinstance(end_gmoney, int)
            else None
        )

        sessions.append(
            {
                "session_index": len(sessions) + 1,
                "room_id": room_id,
                "self_userid": start_self.get("userid"),
                "self_nickname": start_self.get("nickname"),
                "round_count": len(session_rounds),
                "round_wins": self_round_wins,
                "round_losses": len(session_rounds) - self_round_wins,
                "self_score_sum": self_score_sum,
                "start_gmoney": start_gmoney,
                "end_gmoney": end_gmoney,
                "delta_gmoney": delta_gmoney,
                "rounds": session_rounds,
            }
        )

    winning_sessions = sum(1 for session in sessions if (session.get("delta_gmoney") or 0) > 0)
    losing_sessions = sum(1 for session in sessions if (session.get("delta_gmoney") or 0) < 0)

    return {
        "sessions": sessions,
        "session_count": len(sessions),
        "round_count": total_rounds,
        "total_self_score": total_self_score,
        "total_round_wins": total_round_wins,
        "total_round_losses": total_rounds - total_round_wins,
        "winning_sessions": winning_sessions,
        "losing_sessions": losing_sessions,
        "known_gmoney_sessions": sum(1 for session in sessions if session.get("delta_gmoney") is not None),
        "total_gmoney_delta": sum(
            session.get("delta_gmoney") or 0
            for session in sessions
            if session.get("delta_gmoney") is not None
        ),
    }


def render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Session Review",
        "",
        f"- Sessions: `{summary.get('session_count')}`",
        f"- Total rounds: `{summary.get('round_count')}`",
        f"- Self round wins: `{summary.get('total_round_wins')}`",
        f"- Self round losses: `{summary.get('total_round_losses')}`",
        f"- Self cumulative score: `{summary.get('total_self_score')}`",
        f"- Sessions with known bankroll delta: `{summary.get('known_gmoney_sessions')}`",
        f"- Total bankroll delta: `{summary.get('total_gmoney_delta')}`",
        "",
        "## Sessions",
        "",
    ]

    for session in summary.get("sessions", []):
        lines.extend(
            [
                f"### Session {session.get('session_index')}",
                "",
                f"- Room: `{session.get('room_id')}`",
                f"- Self: `{session.get('self_nickname') or session.get('self_userid')}`",
                f"- Rounds: `{session.get('round_count')}`",
                f"- Round wins: `{session.get('round_wins')}`",
                f"- Round losses: `{session.get('round_losses')}`",
                f"- Cumulative score: `{session.get('self_score_sum')}`",
                f"- Start bankroll: `{session.get('start_gmoney')}`",
                f"- End bankroll: `{session.get('end_gmoney')}`",
                f"- Bankroll delta: `{session.get('delta_gmoney')}`",
                "",
                "| Round | Self score | Self won | Remaining | Winner |",
                "| --- | ---: | --- | ---: | --- |",
            ]
        )
        for round_index, round_item in enumerate(session.get("rounds", []), start=1):
            lines.append(
                "| "
                f"{round_index} | "
                f"{round_item.get('self_score')} | "
                f"{'yes' if round_item.get('self_won') else 'no'} | "
                f"{round_item.get('self_remaining_count')} | "
                f"{round_item.get('winner_actor')} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(artifact_dir: str | Path, output_name: str = "session_review.md") -> Path:
    artifact_dir = Path(artifact_dir)
    timeline = load_json(artifact_dir / "game_timeline.json")
    summary = build_session_summary(timeline)
    output_path = artifact_dir / output_name
    output_path.write_text(render_markdown(summary), encoding="utf-8")
    return output_path


def _self_player(players: object) -> dict[str, object]:
    if not isinstance(players, list):
        return {}
    for player in players:
        if isinstance(player, dict) and player.get("actor") == "self":
            return player
    for player in players:
        if isinstance(player, dict) and player.get("seat_index") == 0:
            return player
    if players and isinstance(players[0], dict):
        return players[0]
    return {}
