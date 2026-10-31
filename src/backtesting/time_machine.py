"""Time Machine historical data access layer.
Enforces strict calendar-date isolation (match_date < target_date).
Prevents same-day data leakage, future match contamination, and market odds exposure.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from src.backtesting.context import MatchPredictionContext
from src.database.connection import get_connection


class TimeMachine:
    """Provides time-restricted historical data queries for backtesting.
    
    Rule: A prediction for a match on date D can only access completed matches with match_date < D.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        return get_connection(self.db_path)

    def get_match_context(self, match_id: str) -> MatchPredictionContext:
        """Create a MatchPredictionContext for a specific match.
        
        The context has zero knowledge of the match's own result or future matches.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT match_id, season, division, match_date, match_time,
                       has_reliable_time, home_team, away_team, raw_home_team, raw_away_team
                FROM matches
                WHERE match_id = ?
                """,
                (match_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Match not found: {match_id}")

            return MatchPredictionContext(
                match_id=row["match_id"],
                season=row["season"],
                division=row["division"],
                match_date=row["match_date"],
                match_time=row["match_time"],
                has_reliable_time=row["has_reliable_time"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                raw_home_team=row["raw_home_team"],
                raw_away_team=row["raw_away_team"],
                _time_machine=self,
            )
        finally:
            conn.close()

    def query_team_history(
        self,
        team_name: str,
        before_date: str,
        limit: int = 10,
        venue: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query prior completed matches for a team strictly before before_date.
        
        Enforces: m.match_date < before_date.
        Excludes all same-day matches.
        """
        conn = self._get_connection()
        try:
            query = """
            SELECT 
                m.match_id, m.season, m.division, m.match_date, m.match_time,
                m.home_team, m.away_team,
                r.full_time_home_goals, r.full_time_away_goals, r.full_time_result,
                r.half_time_home_goals, r.half_time_away_goals, r.half_time_result,
                r.home_shots, r.away_shots, r.home_shots_on_target, r.away_shots_on_target,
                r.home_corners, r.away_corners, r.home_fouls, r.away_fouls,
                r.home_yellow_cards, r.away_yellow_cards, r.home_red_cards, r.away_red_cards,
                r.referee
            FROM matches m
            JOIN actual_results r ON m.match_id = r.match_id
            WHERE m.match_date < ?
            """
            params: List[Any] = [before_date]

            if venue == "HOME":
                query += " AND m.home_team = ?"
                params.append(team_name)
            elif venue == "AWAY":
                query += " AND m.away_team = ?"
                params.append(team_name)
            else:
                query += " AND (m.home_team = ? OR m.away_team = ?)"
                params.extend([team_name, team_name])

            query += " ORDER BY m.match_date DESC, m.id DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def query_head_to_head(
        self,
        team_a: str,
        team_b: str,
        before_date: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query direct head-to-head matches between two teams strictly before before_date."""
        conn = self._get_connection()
        try:
            query = """
            SELECT 
                m.match_id, m.season, m.division, m.match_date, m.match_time,
                m.home_team, m.away_team,
                r.full_time_home_goals, r.full_time_away_goals, r.full_time_result,
                r.half_time_home_goals, r.half_time_away_goals, r.half_time_result,
                r.home_shots, r.away_shots, r.home_shots_on_target, r.away_shots_on_target,
                r.home_corners, r.away_corners, r.home_fouls, r.away_fouls,
                r.home_yellow_cards, r.away_yellow_cards, r.home_red_cards, r.away_red_cards
            FROM matches m
            JOIN actual_results r ON m.match_id = r.match_id
            WHERE m.match_date < ?
              AND (
                    (m.home_team = ? AND m.away_team = ?) OR
                    (m.home_team = ? AND m.away_team = ?)
                  )
            ORDER BY m.match_date DESC, m.id DESC
            LIMIT ?
            """
            cursor = conn.execute(
                query, (before_date, team_a, team_b, team_b, team_a, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def query_league_table(
        self,
        season: str,
        division: str,
        before_date: str,
    ) -> List[Dict[str, Any]]:
        """Construct the league standings table strictly from matches completed prior to before_date."""
        conn = self._get_connection()
        try:
            query = """
            SELECT 
                m.home_team, m.away_team,
                r.full_time_home_goals, r.full_time_away_goals, r.full_time_result
            FROM matches m
            JOIN actual_results r ON m.match_id = r.match_id
            WHERE m.season = ? AND m.division = ? AND m.match_date < ?
            """
            cursor = conn.execute(query, (season, division, before_date))
            rows = cursor.fetchall()

            standings: Dict[str, Dict[str, Any]] = {}

            for r in rows:
                ht = r["home_team"]
                at = r["away_team"]
                hg = r["full_time_home_goals"]
                ag = r["full_time_away_goals"]
                res = r["full_time_result"]

                for t in (ht, at):
                    if t not in standings:
                        standings[t] = {
                            "team": t,
                            "played": 0,
                            "won": 0,
                            "drawn": 0,
                            "lost": 0,
                            "goals_for": 0,
                            "goals_against": 0,
                            "goal_difference": 0,
                            "points": 0,
                        }

                # Home record
                standings[ht]["played"] += 1
                standings[ht]["goals_for"] += hg
                standings[ht]["goals_against"] += ag
                if res == "H":
                    standings[ht]["won"] += 1
                    standings[ht]["points"] += 3
                elif res == "D":
                    standings[ht]["drawn"] += 1
                    standings[ht]["points"] += 1
                else:
                    standings[ht]["lost"] += 1

                # Away record
                standings[at]["played"] += 1
                standings[at]["goals_for"] += ag
                standings[at]["goals_against"] += hg
                if res == "A":
                    standings[at]["won"] += 1
                    standings[at]["points"] += 3
                elif res == "D":
                    standings[at]["drawn"] += 1
                    standings[at]["points"] += 1
                else:
                    standings[at]["lost"] += 1

            for s in standings.values():
                s["goal_difference"] = s["goals_for"] - s["goals_against"]

            table = list(standings.values())
            # Rank by points DESC, goal_difference DESC, goals_for DESC
            table.sort(
                key=lambda x: (x["points"], x["goal_difference"], x["goals_for"]),
                reverse=True,
            )
            return table
        finally:
            conn.close()

    def get_backtest_matches(
        self,
        season: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve matches for backtest evaluation in strict chronological order."""
        conn = self._get_connection()
        try:
            query = """
            SELECT match_id, season, division, match_date, match_time,
                   home_team, away_team, status
            FROM matches
            WHERE season = ?
            """
            params: List[Any] = [season]
            if start_date:
                query += " AND match_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND match_date <= ?"
                params.append(end_date)

            query += " ORDER BY match_date ASC, id ASC"
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
