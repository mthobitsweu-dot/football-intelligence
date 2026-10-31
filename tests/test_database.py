"""Unit tests for SQLite database creation, constraints, and data integrity.
Uses Python's built-in unittest framework and runs in-memory.
"""
import sqlite3
import unittest

from src.database.connection import get_connection, init_db
from src.database.repository import (
    get_actual_result,
    get_match,
    get_prediction_evaluations,
    get_predictions_for_match,
    insert_actual_result,
    insert_match,
    insert_prediction,
)


class TestDatabaseFoundation(unittest.TestCase):
    def setUp(self):
        """Create a fresh in-memory database with our schema for each test."""
        self.conn = get_connection(":memory:")
        # Execute schema on this in-memory connection
        from src.database.connection import SCHEMA_PATH
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(schema_sql)

    def tearDown(self):
        """Close connection after test."""
        self.conn.close()

    def test_schema_tables_exist(self):
        """Verify that matches, actual_results, predictions, and views are created."""
        cursor = self.conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')"
        )
        objects = {row["name"]: row["type"] for row in cursor.fetchall()}
        
        self.assertIn("matches", objects)
        self.assertIn("actual_results", objects)
        self.assertIn("predictions", objects)
        self.assertIn("v_prediction_evaluations", objects)

    def test_insert_and_retrieve_match(self):
        """Verify inserting a match and reading it back."""
        match_data = {
            "match_id": "E0_2023-08-11_BUR_MCI",
            "season": "2023-2024",
            "division": "E0",
            "match_date": "2023-08-11",
            "match_time": "20:00:00",
            "home_team": "Burnley",
            "away_team": "Man City",
            "status": "FINISHED",
        }
        inserted_id = insert_match(self.conn, match_data)
        self.assertEqual(inserted_id, "E0_2023-08-11_BUR_MCI")

        fetched = get_match(self.conn, "E0_2023-08-11_BUR_MCI")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["home_team"], "Burnley")
        self.assertEqual(fetched["away_team"], "Man City")
        self.assertEqual(fetched["season"], "2023-2024")

    def test_foreign_key_enforcement(self):
        """Verify that predictions or results cannot be inserted without a valid match."""
        # Non-existent match_id should trigger foreign key constraint failure
        pred = {
            "match_id": "NON_EXISTENT_MATCH",
            "engine_id": "engine_4_stats",
            "model_version": "v1.0.0",
            "predicted_outcome": "H",
            "home_win_prob": 0.50,
            "draw_prob": 0.25,
            "away_win_prob": 0.25,
            "confidence": 0.70,
        }
        with self.assertRaises(sqlite3.IntegrityError):
            insert_prediction(self.conn, pred)

    def test_prediction_probability_sum_constraint(self):
        """Verify that probabilities must sum to 1.0."""
        # First insert a valid match
        insert_match(self.conn, {
            "match_id": "E0_2023-08-12_ARS_NFO",
            "season": "2023-2024",
            "division": "E0",
            "match_date": "2023-08-12",
            "home_team": "Arsenal",
            "away_team": "Nott'm Forest",
            "status": "SCHEDULED",
        })

        # Probabilities summing to 1.4 (invalid)
        invalid_pred = {
            "match_id": "E0_2023-08-12_ARS_NFO",
            "engine_id": "engine_4_stats",
            "model_version": "v1.0.0",
            "predicted_outcome": "H",
            "home_win_prob": 0.80,
            "draw_prob": 0.40,
            "away_win_prob": 0.20,
            "confidence": 0.85,
        }
        with self.assertRaises(sqlite3.IntegrityError):
            insert_prediction(self.conn, invalid_pred)

    def test_valid_prediction_and_evaluation_view(self):
        """Verify linking match, prediction, and actual result computes accuracy correctly."""
        match_id = "E0_2023-08-12_NEW_AVL"
        insert_match(self.conn, {
            "match_id": match_id,
            "season": "2023-2024",
            "division": "E0",
            "match_date": "2023-08-12",
            "home_team": "Newcastle",
            "away_team": "Aston Villa",
            "status": "FINISHED",
        })

        # Insert prediction: Home Win with 60% probability
        insert_prediction(self.conn, {
            "match_id": match_id,
            "engine_id": "engine_4_stats",
            "model_version": "v1.0.0",
            "predicted_outcome": "H",
            "home_win_prob": 0.60,
            "draw_prob": 0.25,
            "away_win_prob": 0.15,
            "confidence": 0.72,
            "predicted_home_goals": 2.1,
            "predicted_away_goals": 0.9,
            "supporting_evidence": "Newcastle home scoring average 2.1 vs Villa away conceded 1.8",
        })

        # Insert verified result: Newcastle won 5-1 (Home Win)
        insert_actual_result(self.conn, {
            "match_id": match_id,
            "full_time_home_goals": 5,
            "full_time_away_goals": 1,
            "full_time_result": "H",
            "half_time_home_goals": 2,
            "half_time_away_goals": 1,
            "half_time_result": "H",
            "home_shots": 17,
            "away_shots": 16,
            "home_shots_on_target": 13,
            "away_shots_on_target": 5,
            "home_corners": 6,
            "away_corners": 5,
        })

        # Query evaluation view
        evals = get_prediction_evaluations(self.conn, engine_id="engine_4_stats")
        self.assertEqual(len(evals), 1)
        record = evals[0]

        self.assertEqual(record["match_id"], match_id)
        self.assertEqual(record["predicted_outcome"], "H")
        self.assertEqual(record["actual_result"], "H")
        self.assertEqual(record["is_correct"], 1)
        self.assertEqual(record["actual_home_goals"], 5)
        self.assertEqual(record["actual_away_goals"], 1)
        # Standardized Summed Brier score error: (0.60-1.0)^2 + (0.25-0)^2 + (0.15-0)^2 = 0.245
        self.assertAlmostEqual(record["brier_score"], 0.245, places=3)
        self.assertLess(record["brier_score"], 0.6667)  # Better than random baseline

    def test_cascade_delete(self):
        """Verify that deleting a match deletes linked predictions and results automatically."""
        match_id = "E0_2023-08-13_CHE_LIV"
        insert_match(self.conn, {
            "match_id": match_id,
            "season": "2023-2024",
            "division": "E0",
            "match_date": "2023-08-13",
            "home_team": "Chelsea",
            "away_team": "Liverpool",
            "status": "FINISHED",
        })
        insert_prediction(self.conn, {
            "match_id": match_id,
            "engine_id": "engine_4_stats",
            "model_version": "v1.0.0",
            "predicted_outcome": "D",
            "home_win_prob": 0.33,
            "draw_prob": 0.34,
            "away_win_prob": 0.33,
            "confidence": 0.50,
        })
        insert_actual_result(self.conn, {
            "match_id": match_id,
            "full_time_home_goals": 1,
            "full_time_away_goals": 1,
            "full_time_result": "D",
        })

        # Confirm existence
        self.assertEqual(len(get_predictions_for_match(self.conn, match_id)), 1)
        self.assertIsNotNone(get_actual_result(self.conn, match_id))

        # Delete the match
        self.conn.execute("DELETE FROM matches WHERE match_id = ?", (match_id,))

        # Verify child records were automatically cascaded
        self.assertEqual(len(get_predictions_for_match(self.conn, match_id)), 0)
        self.assertIsNone(get_actual_result(self.conn, match_id))


if __name__ == "__main__":
    unittest.main()
