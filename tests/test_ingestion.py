"""Unit tests for historical data ingestion, team normalisation, and data validation."""
import tempfile
import unittest
from pathlib import Path

from src.database.connection import get_connection
from src.database.repository import get_actual_result, get_match
from src.ingestion.historical_loader import (
    IngestionAudit,
    ingest_season_csv,
    parse_date,
    parse_optional_int,
    parse_time,
)
from src.ingestion.team_mappings import normalise_team_name


class TestTeamMappings(unittest.TestCase):
    def test_canonical_mappings(self):
        canonical, slug = normalise_team_name("Man City")
        self.assertEqual(canonical, "Manchester City")
        self.assertEqual(slug, "MCI")

        canonical, slug = normalise_team_name("Nott'm Forest")
        self.assertEqual(canonical, "Nottingham Forest")
        self.assertEqual(slug, "NFO")

        canonical, slug = normalise_team_name("Arsenal")
        self.assertEqual(canonical, "Arsenal")
        self.assertEqual(slug, "ARS")

    def test_unknown_team_fallback(self):
        canonical, slug = normalise_team_name("Atlantis FC")
        self.assertEqual(canonical, "Atlantis FC")
        self.assertEqual(slug, "ATL")


class TestDataParsing(unittest.TestCase):
    def test_date_parsing(self):
        self.assertEqual(parse_date("13/08/2021"), "2021-08-13")
        self.assertEqual(parse_date("28/05/23"), "2023-05-28")
        with self.assertRaises(ValueError):
            parse_date("invalid-date-string")

    def test_time_parsing(self):
        time_str, reliable = parse_time("17:30")
        self.assertEqual(time_str, "17:30:00")
        self.assertEqual(reliable, 1)

        # Blank or missing
        time_str, reliable = parse_time("")
        self.assertIsNone(time_str)
        self.assertEqual(reliable, 0)

    def test_optional_int_null_handling(self):
        self.assertEqual(parse_optional_int("15"), 15)
        self.assertEqual(parse_optional_int("0"), 0)
        # Empty string must return None, NOT 0
        self.assertIsNone(parse_optional_int(""))
        self.assertIsNone(parse_optional_int(None))


class TestIngestionWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        # Initialize schema
        from src.database.connection import init_db
        init_db(str(self.db_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ingest_valid_csv_and_null_preservation(self):
        """Test valid CSV import and verify missing stats are stored as NULL (not 0)."""
        csv_content = (
            "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HC,AC,HF,AF,HY,AY,HR,AR\n"
            "E0,13/08/2021,20:00,Brentford,Arsenal,2,0,H,1,0,H,M Oliver,8,22,3,4,2,5,12,8,0,0,0,0\n"
            "E0,14/08/2021,12:30,Man United,Leeds,5,1,H,1,0,H,P Tierney,,,,,,,,,,,,\n"  # Missing all in-match stats
        )
        csv_file = Path(self.temp_dir.name) / "test_season.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        audit = ingest_season_csv(csv_file, season="2021-2022", db_path=str(self.db_path))
        self.assertEqual(audit.imported_count, 2)
        self.assertEqual(audit.rejected_count, 0)
        self.assertEqual(audit.duplicate_count, 0)

        conn = get_connection(str(self.db_path))
        try:
            # Check match 1 (Brentford vs Arsenal)
            m1 = get_match(conn, "E0_2021-08-13_BRE_ARS")
            self.assertIsNotNone(m1)
            self.assertEqual(m1["home_team"], "Brentford")
            self.assertEqual(m1["away_team"], "Arsenal")
            self.assertEqual(m1["raw_home_team"], "Brentford")
            self.assertEqual(m1["source_file"], "test_season.csv")

            r1 = get_actual_result(conn, "E0_2021-08-13_BRE_ARS")
            self.assertEqual(r1["full_time_home_goals"], 2)
            self.assertEqual(r1["full_time_away_goals"], 0)
            self.assertEqual(r1["full_time_result"], "H")
            self.assertEqual(r1["home_shots"], 8)

            # Check match 2 (Man United vs Leeds) - Missing shots must be None / NULL, NEVER 0
            m2 = get_match(conn, "E0_2021-08-14_MUN_LEE")
            self.assertIsNotNone(m2)
            self.assertEqual(m2["home_team"], "Manchester United")
            self.assertEqual(m2["raw_home_team"], "Man United")

            r2 = get_actual_result(conn, "E0_2021-08-14_MUN_LEE")
            self.assertIsNone(r2["home_shots"])
            self.assertIsNone(r2["away_shots"])
            self.assertIsNone(r2["home_corners"])
        finally:
            conn.close()

    def test_duplicate_and_outcome_mismatch_handling(self):
        """Verify duplicate fixtures and corrupted scores are caught and rejected."""
        csv_content = (
            "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
            "E0,14/08/2021,15:00,Chelsea,Crystal Palace,3,0,H\n"
            "E0,14/08/2021,15:00,Chelsea,Crystal Palace,3,0,H\n"  # Duplicate
            "E0,14/08/2021,15:00,Burnley,Brighton,1,2,H\n"        # Corrupted: 1-2 cannot be 'H'
        )
        csv_file = Path(self.temp_dir.name) / "bad_data.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        audit = ingest_season_csv(csv_file, season="2021-2022", db_path=str(self.db_path))
        self.assertEqual(audit.imported_count, 1)
        self.assertEqual(audit.duplicate_count, 1)
        self.assertEqual(audit.rejected_count, 1)
        self.assertEqual(len(audit.rejections), 2)


if __name__ == "__main__":
    unittest.main()
