"""Unit tests for the backtesting harness, cold-start handling, and prediction immutability.
Validates that backtest runs are independent, decoupled from live predictions,
and enforce strict numerical constraints.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.backtesting.evaluator import BacktestEvaluator
from src.database.connection import get_connection, init_db
from src.database.repository import insert_actual_result, insert_match
from src.engines.base import BaseEngine, EnginePrediction


class MockEngine(BaseEngine):
    engine_id: str = "engine_test_mock"
    model_version: str = "v1.0.0"

    def predict(self, context) -> EnginePrediction:
        # Check team history sample size
        home_hist = context.get_team_history(context.home_team, limit=5, min_required=3)
        away_hist = context.get_team_history(context.away_team, limit=5, min_required=3)

        status = "OK" if (home_hist.has_sufficient_data and away_hist.has_sufficient_data) else "INSUFFICIENT_DATA"

        return EnginePrediction(
            engine_id=self.engine_id,
            model_version=self.model_version,
            predicted_outcome="H",
            home_win_prob=0.50,
            draw_prob=0.30,
            away_win_prob=0.20,
            confidence=0.50,
            prediction_status=status,
            home_sample_size=home_hist.sample_size,
            away_sample_size=away_hist.sample_size,
        )


class TestBacktestingHarness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "harness_test.db"
        init_db(str(self.db_path))

        conn = get_connection(str(self.db_path))
        try:
            # Seed 2 historical matches
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
            insert_match(conn, {
                "match_id": "M_2023-08-12_ARS_NFO",
                "season": "2023-2024",
                "division": "E0",
                "match_date": "2023-08-12",
                "home_team": "Arsenal",
                "away_team": "Nottingham Forest",
                "status": "FINISHED",
            })
            insert_actual_result(conn, {
                "match_id": "M_2023-08-12_ARS_NFO",
                "full_time_home_goals": 2,
                "full_time_away_goals": 1,
                "full_time_result": "H",
            })
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_probability_tolerance_enforcement(self):
        """EnginePrediction must enforce ABS(prob_sum - 1.0) < 0.000001."""
        # Valid probabilities
        pred = EnginePrediction(
            engine_id="engine_test",
            model_version="v1.0.0",
            predicted_outcome="H",
            home_win_prob=0.555555,
            draw_prob=0.222222,
            away_win_prob=0.222223,
            confidence=0.555555,
        )
        self.assertAlmostEqual(pred.home_win_prob + pred.draw_prob + pred.away_win_prob, 1.0, places=6)

        # Invalid sum (1.05) must raise ValueError
        with self.assertRaises(ValueError):
            EnginePrediction(
                engine_id="engine_test",
                model_version="v1.0.0",
                predicted_outcome="H",
                home_win_prob=0.55,
                draw_prob=0.30,
                away_win_prob=0.20,
                confidence=0.55,
            )

    def test_confidence_is_max_probability(self):
        """Confidence must equal max(home, draw, away)."""
        pred = EnginePrediction(
            engine_id="engine_test",
            model_version="v1.0.0",
            predicted_outcome="D",
            home_win_prob=0.20,
            draw_prob=0.60,
            away_win_prob=0.20,
            confidence=0.40,  # Incorrect confidence passed
        )
        # Should be auto-corrected to max (0.60)
        self.assertEqual(pred.confidence, 0.60)

    def test_cold_start_handling(self):
        """On Matchday 1, teams with zero prior matches must report sample_size=0 and has_sufficient_data=False."""
        evaluator = BacktestEvaluator(db_path=str(self.db_path))
        ctx = evaluator.time_machine.get_match_context("M_2023-08-11_BUR_MCI")
        burnley_hist = ctx.get_team_history("Burnley", limit=5, min_required=3)
        self.assertEqual(burnley_hist.sample_size, 0)
        self.assertFalse(burnley_hist.has_sufficient_data)

    def test_repeated_runs_do_not_collide_and_live_predictions_untouched(self):
        """Running the same engine and version twice creates independent backtest runs without collisions."""
        evaluator = BacktestEvaluator(db_path=str(self.db_path))
        engine = MockEngine()

        res1 = evaluator.evaluate_engine(engine, season="2023-2024", run_id="RUN_TEST_1")
        res2 = evaluator.evaluate_engine(engine, season="2023-2024", run_id="RUN_TEST_2")

        self.assertEqual(res1.status, "COMPLETED")
        self.assertEqual(res2.status, "COMPLETED")

        conn = get_connection(str(self.db_path))
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM backtest_runs;")
            self.assertEqual(cur.fetchone()[0], 2)

            cur.execute("SELECT COUNT(*) FROM backtest_predictions WHERE run_id = 'RUN_TEST_1';")
            self.assertEqual(cur.fetchone()[0], 2)

            cur.execute("SELECT COUNT(*) FROM backtest_predictions WHERE run_id = 'RUN_TEST_2';")
            self.assertEqual(cur.fetchone()[0], 2)

            # Live predictions table MUST remain untouched (0 rows)
            cur.execute("SELECT COUNT(*) FROM predictions;")
            self.assertEqual(cur.fetchone()[0], 0)
        finally:
            conn.close()

    def test_incomplete_run_reports_null_metrics(self):
        """A backtest with 0 evaluated matches must report status='COMPLETED' with accuracy=None (never 0.0)."""
        evaluator = BacktestEvaluator(db_path=str(self.db_path))
        engine = MockEngine()

        # Query a non-existent season
        res = evaluator.evaluate_engine(engine, season="1990-1991", run_id="RUN_EMPTY")
        self.assertEqual(res.status, "COMPLETED")
        self.assertEqual(res.total_evaluated_matches, 0)
        self.assertIsNone(res.brier_score)
        self.assertIsNone(res.accuracy_breakdown)

        conn = get_connection(str(self.db_path))
        try:
            cur = conn.cursor()
            cur.execute("SELECT accuracy, brier_score FROM backtest_runs WHERE run_id = 'RUN_EMPTY';")
            row = cur.fetchone()
            self.assertIsNone(row["accuracy"])
            self.assertIsNone(row["brier_score"])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
