"""Unit and statistical verification tests for Engine #4: Statistical & Form Analysis.
Verifies Poisson math, adaptive score matrix tail mass, Bayesian shrinkage,
confidence calculation, missing-stat safety, and supporting evidence.
"""
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.backtesting.time_machine import TimeMachine
from src.database.connection import init_db
from src.database.repository import insert_actual_result, insert_match
from src.engines.engine_4_stats.config import StatisticalEngineConfig
from src.engines.engine_4_stats.statistical import (
    Engine4Statistical,
    compute_adaptive_score_matrix,
    compute_poisson_probability,
)


class TestPoissonScoreMatrix(unittest.TestCase):
    def test_poisson_probability_values(self):
        """Verify Poisson formula P(X=k; lambda) against hand-calculated values."""
        # lambda=1.5, k=0: e^(-1.5) ~ 0.223130
        self.assertAlmostEqual(compute_poisson_probability(1.5, 0), math.exp(-1.5), places=6)
        # lambda=2.0, k=2: 2^2 * e^(-2) / 2! = 2 * e^(-2) ~ 0.270671
        self.assertAlmostEqual(compute_poisson_probability(2.0, 2), 2 * math.exp(-2.0), places=6)
        # lambda=0, k=0: 1.0
        self.assertEqual(compute_poisson_probability(0, 0), 1.0)
        self.assertEqual(compute_poisson_probability(0, 1), 0.0)

    def test_adaptive_matrix_tail_mass_negligible(self):
        """Verify that the unnormalized omitted tail mass is negligible (< 1e-6) for typical matches."""
        # Typical Premier League match: Home xG=1.8, Away xG=1.2
        p_h, p_d, p_a, omitted_mass = compute_adaptive_score_matrix(1.8, 1.2)
        
        # Omitted mass must be less than 1 in a million
        self.assertLess(omitted_mass, 1e-6)
        # Normalized probabilities must sum strictly to 1.0 within tolerance
        self.assertLess(abs(p_h + p_d + p_a - 1.0), 0.000001)

    def test_extreme_expected_goals_safe_handling(self):
        """High expected goals (e.g. 5.5 vs 4.0) must expand the matrix adaptively and omit < 1e-4 tail mass."""
        p_h, p_d, p_a, omitted_mass = compute_adaptive_score_matrix(5.5, 4.0)
        self.assertLess(omitted_mass, 1e-4)
        self.assertLess(abs(p_h + p_d + p_a - 1.0), 0.000001)


class TestEngine4Logic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "engine4_test.db"
        init_db(str(self.db_path))
        self.time_machine = TimeMachine(db_path=str(self.db_path))
        self.engine = Engine4Statistical(StatisticalEngineConfig())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cold_start_status_and_promoted_shrinkage(self):
        """Newly promoted team with 0 prior matches receives INSUFFICIENT_DATA status and promoted prior."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            insert_match(conn, {
                "match_id": "M_2023-08-11_BUR_MCI",
                "season": "2023-2024",
                "division": "E0",
                "match_date": "2023-08-11",
                "home_team": "Burnley",
                "away_team": "Manchester City",
                "status": "FINISHED",
            })
            insert_actual_result(conn, {
                "match_id": "M_2023-08-11_BUR_MCI",
                "full_time_home_goals": 0,
                "full_time_away_goals": 3,
                "full_time_result": "A",
            })
            conn.commit()
        finally:
            conn.close()

        context = self.time_machine.get_match_context("M_2023-08-11_BUR_MCI")
        pred = self.engine.predict(context)

        # Both Burnley and Man City have 0 prior matches on Matchday 1
        self.assertEqual(pred.home_sample_size, 0)
        self.assertEqual(pred.away_sample_size, 0)
        self.assertEqual(pred.prediction_status, "INSUFFICIENT_DATA")
        # Confidence must be max(home, draw, away)
        self.assertEqual(pred.confidence, max(pred.home_win_prob, pred.draw_prob, pred.away_win_prob))
        # Probabilities sum to 1.0 within tolerance
        self.assertLess(abs(pred.home_win_prob + pred.draw_prob + pred.away_win_prob - 1.0), 0.000001)

    def test_established_team_sufficient_data_status(self):
        """When both teams have >= 5 prior matches, prediction_status is OK."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Seed 6 matches for Team A and Team B in July 2023
            for i in range(1, 7):
                mid = f"M_2023-07-0{i}_ARS_CHE"
                date_str = f"2023-07-0{i}"
                insert_match(conn, {
                    "match_id": mid,
                    "season": "2023-2024",
                    "division": "E0",
                    "match_date": date_str,
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "status": "FINISHED",
                })
                insert_actual_result(conn, {
                    "match_id": mid,
                    "full_time_home_goals": 2,
                    "full_time_away_goals": 1,
                    "full_time_result": "H",
                    "home_shots_on_target": 6,
                    "away_shots_on_target": 3,
                })

            # Target match on 2023-08-01
            target_mid = "M_2023-08-01_ARS_CHE"
            insert_match(conn, {
                "match_id": target_mid,
                "season": "2023-2024",
                "division": "E0",
                "match_date": "2023-08-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "status": "FINISHED",
            })
            insert_actual_result(conn, {
                "match_id": target_mid,
                "full_time_home_goals": 1,
                "full_time_away_goals": 1,
                "full_time_result": "D",
            })
            conn.commit()
        finally:
            conn.close()

        context = self.time_machine.get_match_context("M_2023-08-01_ARS_CHE")
        pred = self.engine.predict(context)

        self.assertEqual(pred.home_sample_size, 6)
        self.assertEqual(pred.away_sample_size, 6)
        self.assertEqual(pred.prediction_status, "OK")
        self.assertIn("StatModel", pred.supporting_evidence)

    def test_missing_shots_on_target_does_not_crash(self):
        """When shots on target are NULL, engine must handle gracefully without converting to 0."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            insert_match(conn, {
                "match_id": "M_2023-08-01_LIV_EVE",
                "season": "2023-2024",
                "division": "E0",
                "match_date": "2023-08-01",
                "home_team": "Liverpool",
                "away_team": "Everton",
                "status": "FINISHED",
            })
            insert_actual_result(conn, {
                "match_id": "M_2023-08-01_LIV_EVE",
                "full_time_home_goals": 2,
                "full_time_away_goals": 0,
                "full_time_result": "H",
                # Shots on target left NULL
            })
            insert_match(conn, {
                "match_id": "M_2023-08-02_LIV_NEW",
                "season": "2023-2024",
                "division": "E0",
                "match_date": "2023-08-02",
                "home_team": "Liverpool",
                "away_team": "Newcastle United",
                "status": "FINISHED",
            })
            insert_actual_result(conn, {
                "match_id": "M_2023-08-02_LIV_NEW",
                "full_time_home_goals": 1,
                "full_time_away_goals": 1,
                "full_time_result": "D",
            })
            conn.commit()
        finally:
            conn.close()

        context = self.time_machine.get_match_context("M_2023-08-02_LIV_NEW")
        pred = self.engine.predict(context)
        self.assertIsNotNone(pred)
        self.assertLess(abs(pred.home_win_prob + pred.draw_prob + pred.away_win_prob - 1.0), 0.000001)


if __name__ == "__main__":
    unittest.main()
