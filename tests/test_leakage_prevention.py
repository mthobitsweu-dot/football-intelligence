"""Automated tests specifically designed to detect data leakage.
Verifies strict calendar-date cutoff (match_date < target_date), same-day exclusion,
future match exclusion, and target match masking.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.backtesting.time_machine import TimeMachine
from src.database.connection import init_db
from src.database.repository import insert_actual_result, insert_match


class TestDataLeakagePrevention(unittest.TestCase):
    def setUp(self):
        """Set up an isolated temporary SQLite database."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "leakage_test.db"
        init_db(str(self.db_path))
        self.time_machine = TimeMachine(db_path=str(self.db_path))

        # Insert a controlled set of matches around target date 2023-08-12
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # Prior Match: 2023-08-11 (Yesterday)
            insert_match(conn, {
                "match_id": "M_2023-08-11_BUR_MCI",
                "season": "2023-2024",
                "division": "E0",
                "match_date": "2023-08-11",
                "match_time": "20:00:00",
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

            # Same-Day Match 1 (Earlier kickoff): 2023-08-12 12:30
            insert_match(conn, {
                "match_id": "M_2023-08-12_ARS_NFO_1230",
                "season": "2023-2024",
                "division": "E0",
                "match_date": "2023-08-12",
                "match_time": "12:30:00",
                "home_team": "Arsenal",
                "away_team": "Nottingham Forest",
                "status": "FINISHED",
            })
            insert_actual_result(conn, {
                "match_id": "M_2023-08-12_ARS_NFO_1230",
                "full_time_home_goals": 2,
                "full_time_away_goals": 1,
                "full_time_result": "H",
            })

            # TARGET MATCH: 2023-08-12 17:30 (Newcastle vs Aston Villa)
            insert_match(conn, {
                "match_id": "M_2023-08-12_NEW_AVL_1730",
                "season": "2023-2024",
                "division": "E0",
                "match_date": "2023-08-12",
                "match_time": "17:30:00",
                "home_team": "Newcastle United",
                "away_team": "Aston Villa",
                "status": "FINISHED",
            })
            insert_actual_result(conn, {
                "match_id": "M_2023-08-12_NEW_AVL_1730",
                "full_time_home_goals": 5,
                "full_time_away_goals": 1,
                "full_time_result": "H",
            })

            # Future Match 1 (Tomorrow): 2023-08-13
            insert_match(conn, {
                "match_id": "M_2023-08-13_CHE_LIV",
                "season": "2023-2024",
                "division": "E0",
                "match_date": "2023-08-13",
                "match_time": "16:30:00",
                "home_team": "Chelsea",
                "away_team": "Liverpool",
                "status": "FINISHED",
            })
            insert_actual_result(conn, {
                "match_id": "M_2023-08-13_CHE_LIV",
                "full_time_home_goals": 1,
                "full_time_away_goals": 1,
                "full_time_result": "D",
            })

            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_strict_same_day_exclusion(self):
        """Verify that same-day matches are strictly excluded, even if they kicked off earlier in the day."""
        context = self.time_machine.get_match_context("M_2023-08-12_NEW_AVL_1730")
        
        # Arsenal played at 12:30 on the same day (2023-08-12).
        # Under the strict calendar-date cutoff rule, Arsenal's match MUST be invisible.
        arsenal_history = context.get_team_history("Arsenal")
        self.assertEqual(arsenal_history.sample_size, 0)
        self.assertEqual(len(arsenal_history.matches), 0)

    def test_prior_matches_are_accessible(self):
        """Verify that matches strictly before the target date are accessible."""
        context = self.time_machine.get_match_context("M_2023-08-12_NEW_AVL_1730")
        
        # Man City played on 2023-08-11 (strictly before target date 2023-08-12).
        mci_history = context.get_team_history("Manchester City")
        self.assertEqual(mci_history.sample_size, 1)
        self.assertEqual(mci_history.matches[0]["full_time_away_goals"], 3)
        self.assertEqual(mci_history.matches[0]["full_time_result"], "A")

    def test_future_matches_are_strictly_excluded(self):
        """Verify that matches on tomorrow or future dates are invisible."""
        context = self.time_machine.get_match_context("M_2023-08-12_NEW_AVL_1730")
        
        # Chelsea vs Liverpool took place on 2023-08-13.
        chelsea_history = context.get_team_history("Chelsea")
        self.assertEqual(chelsea_history.sample_size, 0)

    def test_target_match_result_is_masked(self):
        """Verify that the target match's own result is completely inaccessible in context."""
        context = self.time_machine.get_match_context("M_2023-08-12_NEW_AVL_1730")
        
        # Newcastle's own match on 2023-08-12 ended 5-1.
        # It must NOT appear in Newcastle's history for this prediction.
        newcastle_history = context.get_team_history("Newcastle United")
        self.assertEqual(newcastle_history.sample_size, 0)

    def test_league_table_excludes_same_day_and_future_matches(self):
        """Verify that the dynamically constructed league table only includes matches before target_date."""
        context = self.time_machine.get_match_context("M_2023-08-12_NEW_AVL_1730")
        standings = context.get_league_table()
        
        # Only Burnley vs Man City was played prior to 2023-08-12.
        teams_in_table = {s["team"]: s["points"] for s in standings}
        self.assertIn("Manchester City", teams_in_table)
        self.assertEqual(teams_in_table["Manchester City"], 3)
        self.assertIn("Burnley", teams_in_table)
        self.assertEqual(teams_in_table["Burnley"], 0)
        
        # Arsenal, Newcastle, and Chelsea must NOT be in the standings yet
        self.assertNotIn("Arsenal", teams_in_table)
        self.assertNotIn("Newcastle United", teams_in_table)
        self.assertNotIn("Chelsea", teams_in_table)


if __name__ == "__main__":
    unittest.main()
