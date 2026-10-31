"""Data access repository for Football Intelligence.
Provides clean, structured functions to store and retrieve matches, predictions,
and actual match outcomes using Python's built-in sqlite3.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional


def insert_match(conn: sqlite3.Connection, match: Dict[str, Any]) -> str:
    """Insert or replace a match record. Returns the match_id."""
    sql = """
    INSERT INTO matches (
        match_id, season, division, match_date, match_time, has_reliable_time,
        home_team, away_team, raw_home_team, raw_away_team, source_file, status
    ) VALUES (
        :match_id, :season, :division, :match_date, :match_time, :has_reliable_time,
        :home_team, :away_team, :raw_home_team, :raw_away_team, :source_file, :status
    )
    ON CONFLICT(match_id) DO UPDATE SET
        match_date = excluded.match_date,
        match_time = excluded.match_time,
        has_reliable_time = excluded.has_reliable_time,
        home_team = excluded.home_team,
        away_team = excluded.away_team,
        raw_home_team = excluded.raw_home_team,
        raw_away_team = excluded.raw_away_team,
        source_file = excluded.source_file,
        status = excluded.status,
        updated_at = datetime('now', 'utc');
    """
    payload = {
        "match_id": match["match_id"],
        "season": match["season"],
        "division": match["division"],
        "match_date": match["match_date"],
        "match_time": match.get("match_time"),
        "has_reliable_time": match.get("has_reliable_time", 1),
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "raw_home_team": match.get("raw_home_team", match["home_team"]),
        "raw_away_team": match.get("raw_away_team", match["away_team"]),
        "source_file": match.get("source_file"),
        "status": match.get("status", "FINISHED"),
    }
    conn.execute(sql, payload)
    return match["match_id"]


def get_match(conn: sqlite3.Connection, match_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a match by its unique match_id."""
    cursor = conn.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def list_matches(conn: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    """List recent matches ordered chronologically."""
    cursor = conn.execute("SELECT * FROM matches ORDER BY match_date ASC, id ASC LIMIT ?", (limit,))
    return [dict(row) for row in cursor.fetchall()]


def count_matches(conn: sqlite3.Connection, season: Optional[str] = None) -> int:
    """Count total matches, optionally filtered by season."""
    if season:
        cur = conn.execute("SELECT COUNT(*) FROM matches WHERE season = ?", (season,))
    else:
        cur = conn.execute("SELECT COUNT(*) FROM matches")
    return cur.fetchone()[0]


def insert_actual_result(conn: sqlite3.Connection, result: Dict[str, Any]) -> int:
    """Insert or update the verified post-match score and stats."""
    sql = """
    INSERT INTO actual_results (
        match_id, referee, full_time_home_goals, full_time_away_goals, full_time_result,
        half_time_home_goals, half_time_away_goals, half_time_result,
        home_shots, away_shots, home_shots_on_target, away_shots_on_target,
        home_corners, away_corners, home_fouls, away_fouls,
        home_yellow_cards, away_yellow_cards, home_red_cards, away_red_cards
    ) VALUES (
        :match_id, :referee, :full_time_home_goals, :full_time_away_goals, :full_time_result,
        :half_time_home_goals, :half_time_away_goals, :half_time_result,
        :home_shots, :away_shots, :home_shots_on_target, :away_shots_on_target,
        :home_corners, :away_corners, :home_fouls, :away_fouls,
        :home_yellow_cards, :away_yellow_cards, :home_red_cards, :away_red_cards
    )
    ON CONFLICT(match_id) DO UPDATE SET
        referee = excluded.referee,
        full_time_home_goals = excluded.full_time_home_goals,
        full_time_away_goals = excluded.full_time_away_goals,
        full_time_result = excluded.full_time_result,
        half_time_home_goals = excluded.half_time_home_goals,
        half_time_away_goals = excluded.half_time_away_goals,
        half_time_result = excluded.half_time_result,
        home_shots = excluded.home_shots,
        away_shots = excluded.away_shots,
        home_shots_on_target = excluded.home_shots_on_target,
        away_shots_on_target = excluded.away_shots_on_target,
        home_corners = excluded.home_corners,
        away_corners = excluded.away_corners,
        home_fouls = excluded.home_fouls,
        away_fouls = excluded.away_fouls,
        home_yellow_cards = excluded.home_yellow_cards,
        away_yellow_cards = excluded.away_yellow_cards,
        home_red_cards = excluded.home_red_cards,
        away_red_cards = excluded.away_red_cards,
        recorded_at = datetime('now', 'utc');
    """
    params = {
        "match_id": result["match_id"],
        "referee": result.get("referee"),
        "full_time_home_goals": result["full_time_home_goals"],
        "full_time_away_goals": result["full_time_away_goals"],
        "full_time_result": result["full_time_result"],
        "half_time_home_goals": result.get("half_time_home_goals"),
        "half_time_away_goals": result.get("half_time_away_goals"),
        "half_time_result": result.get("half_time_result"),
        "home_shots": result.get("home_shots"),
        "away_shots": result.get("away_shots"),
        "home_shots_on_target": result.get("home_shots_on_target"),
        "away_shots_on_target": result.get("away_shots_on_target"),
        "home_corners": result.get("home_corners"),
        "away_corners": result.get("away_corners"),
        "home_fouls": result.get("home_fouls"),
        "away_fouls": result.get("away_fouls"),
        "home_yellow_cards": result.get("home_yellow_cards"),
        "away_yellow_cards": result.get("away_yellow_cards"),
        "home_red_cards": result.get("home_red_cards"),
        "away_red_cards": result.get("away_red_cards"),
    }
    cursor = conn.execute(sql, params)
    return cursor.lastrowid or 0


def get_actual_result(conn: sqlite3.Connection, match_id: str) -> Optional[Dict[str, Any]]:
    """Fetch verified actual outcome for a match."""
    cursor = conn.execute("SELECT * FROM actual_results WHERE match_id = ?", (match_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def insert_prediction(conn: sqlite3.Connection, prediction: Dict[str, Any]) -> int:
    """Insert an engine's prediction for a match.
    
    Enforces probabilities sum to 1.0 and valid engine identifiers via SQLite constraints.
    """
    sql = """
    INSERT INTO predictions (
        match_id, engine_id, model_version, predicted_outcome,
        home_win_prob, draw_prob, away_win_prob, confidence,
        predicted_home_goals, predicted_away_goals, supporting_evidence
    ) VALUES (
        :match_id, :engine_id, :model_version, :predicted_outcome,
        :home_win_prob, :draw_prob, :away_win_prob, :confidence,
        :predicted_home_goals, :predicted_away_goals, :supporting_evidence
    )
    ON CONFLICT(match_id, engine_id, model_version) DO UPDATE SET
        predicted_outcome = excluded.predicted_outcome,
        home_win_prob = excluded.home_win_prob,
        draw_prob = excluded.draw_prob,
        away_win_prob = excluded.away_win_prob,
        confidence = excluded.confidence,
        predicted_home_goals = excluded.predicted_home_goals,
        predicted_away_goals = excluded.predicted_away_goals,
        supporting_evidence = excluded.supporting_evidence,
        created_at = datetime('now', 'utc');
    """
    params = {
        "match_id": prediction["match_id"],
        "engine_id": prediction["engine_id"],
        "model_version": prediction["model_version"],
        "predicted_outcome": prediction["predicted_outcome"],
        "home_win_prob": round(float(prediction["home_win_prob"]), 4),
        "draw_prob": round(float(prediction["draw_prob"]), 4),
        "away_win_prob": round(float(prediction["away_win_prob"]), 4),
        "confidence": float(prediction["confidence"]),
        "predicted_home_goals": prediction.get("predicted_home_goals"),
        "predicted_away_goals": prediction.get("predicted_away_goals"),
        "supporting_evidence": prediction.get("supporting_evidence"),
    }
    cursor = conn.execute(sql, params)
    return cursor.lastrowid or 0


def get_predictions_for_match(
    conn: sqlite3.Connection, match_id: str, engine_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve all predictions made for a given match, optionally filtered by engine."""
    if engine_id:
        cursor = conn.execute(
            "SELECT * FROM predictions WHERE match_id = ? AND engine_id = ? ORDER BY id ASC",
            (match_id, engine_id),
        )
    else:
        cursor = conn.execute(
            "SELECT * FROM predictions WHERE match_id = ? ORDER BY id ASC",
            (match_id,),
        )
    return [dict(row) for row in cursor.fetchall()]


def get_prediction_evaluations(
    conn: sqlite3.Connection, engine_id: Optional[str] = None, season: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Query the evaluation view to check prediction accuracy and Brier score against actual results."""
    query = "SELECT * FROM v_prediction_evaluations WHERE 1=1"
    params: List[Any] = []
    
    if engine_id:
        query += " AND engine_id = ?"
        params.append(engine_id)
    if season:
        query += " AND season = ?"
        params.append(season)

    query += " ORDER BY match_date ASC"
    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]
