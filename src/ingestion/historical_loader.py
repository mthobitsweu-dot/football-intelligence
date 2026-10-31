"""Historical match data ingestion engine.
Safely parses raw CSV match files from Football-Data.co.uk into SQLite.
Performs data validation, duplicate fixture detection, team normalisation,
and strict NULL handling for unrecorded statistics without data leakage.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.database.connection import get_connection, get_db_context
from src.database.repository import insert_actual_result, insert_match
from src.ingestion.team_mappings import normalise_team_name

logger = logging.getLogger(__name__)

# Mandatory columns that must exist in the CSV header
MANDATORY_COLUMNS = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}

# Tracked statistical metrics and their target database types
STAT_METRICS = [
    "HTHG", "HTAG", "HTR", "Referee",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
]


@dataclass
class IngestionAudit:
    source_file: str
    season: str
    total_rows_read: int = 0
    imported_count: int = 0
    rejected_count: int = 0
    duplicate_count: int = 0
    rejections: List[str] = field(default_factory=list)
    missing_stats_count: Dict[str, int] = field(default_factory=lambda: {col: 0 for col in STAT_METRICS})
    earliest_date: Optional[str] = None
    latest_date: Optional[str] = None

    def missing_percentages(self) -> Dict[str, float]:
        """Calculate missing value percentages for each statistic based on valid rows."""
        if self.imported_count == 0:
            return {k: 0.0 for k in self.missing_stats_count}
        return {
            k: round((count / self.imported_count) * 100.0, 2)
            for k, count in self.missing_stats_count.items()
        }


def parse_date(date_str: str) -> str:
    """Parse raw date string ('DD/MM/YYYY' or 'DD/MM/YY') into standard ISO 'YYYY-MM-DD'."""
    cleaned = date_str.strip()
    parts = cleaned.split("/")
    if len(parts) == 3:
        day, month, year = parts
        if len(year) == 2:
            year = "20" + year
        return f"{year.zfill(4)}-{month.zfill(2)}-{day.zfill(2)}"
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: '{date_str}'")


def parse_time(time_str: Optional[str]) -> Tuple[Optional[str], int]:
    """Parse kickoff time string into 'HH:MM:SS'. Returns (time_string, has_reliable_time)."""
    if not time_str or not time_str.strip():
        return None, 0
    cleaned = time_str.strip()
    parts = cleaned.split(":")
    if len(parts) == 2:
        try:
            hours, minutes = int(parts[0]), int(parts[1])
            if 0 <= hours <= 23 and 0 <= minutes <= 59:
                return f"{hours:02d}:{minutes:02d}:00", 1
        except ValueError:
            pass
    return None, 0


def parse_optional_int(val: Optional[str]) -> Optional[int]:
    """Convert string to integer or return None if missing/blank. Never defaults to zero."""
    if val is None:
        return None
    cleaned = val.strip()
    if cleaned == "":
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def parse_optional_str(val: Optional[str]) -> Optional[str]:
    """Strip string or return None if blank. Never invents placeholder strings."""
    if val is None:
        return None
    cleaned = val.strip()
    return cleaned if cleaned != "" else None


def ingest_season_csv(
    csv_path: str | Path,
    season: str,
    db_path: Optional[str] = None,
) -> IngestionAudit:
    """Ingest a historical CSV file into SQLite.
    
    Preserves original CSV untouched, validates data integrity, normalises team names,
    detects duplicates, and records NULLs for missing metrics.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    audit = IngestionAudit(source_file=path.name, season=season)
    seen_fixtures: set[Tuple[str, str]] = set()

    with path.open(mode="r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            audit.rejections.append("Empty CSV or missing header")
            return audit

        headers = set(reader.fieldnames)
        missing_mandatory = MANDATORY_COLUMNS - headers
        if missing_mandatory:
            raise ValueError(f"CSV missing mandatory columns: {missing_mandatory}")

        matches_to_insert = []
        results_to_insert = []

        for line_num, row in enumerate(reader, start=2):
            if not row.get("HomeTeam") or not row.get("AwayTeam"):
                continue

            audit.total_rows_read += 1
            raw_home = row["HomeTeam"].strip()
            raw_away = row["AwayTeam"].strip()

            # 1. Duplicate Fixture Detection within season
            fixture_pair = (raw_home, raw_away)
            if fixture_pair in seen_fixtures:
                audit.duplicate_count += 1
                audit.rejections.append(
                    f"Line {line_num}: Duplicate fixture {raw_home} vs {raw_away} in season {season}"
                )
                continue
            seen_fixtures.add(fixture_pair)

            # 2. Date and Time Validation
            try:
                match_date = parse_date(row["Date"])
            except Exception as e:
                audit.rejected_count += 1
                audit.rejections.append(f"Line {line_num}: Invalid date '{row.get('Date')}': {e}")
                continue

            match_time, has_reliable_time = parse_time(row.get("Time"))

            # Track date bounds
            if audit.earliest_date is None or match_date < audit.earliest_date:
                audit.earliest_date = match_date
            if audit.latest_date is None or match_date > audit.latest_date:
                audit.latest_date = match_date

            # 3. Score and Outcome Validation
            try:
                fthg = int(row["FTHG"])
                ftag = int(row["FTAG"])
                ftr = row["FTR"].strip().upper()
                if fthg < 0 or ftag < 0 or ftr not in ("H", "D", "A"):
                    raise ValueError(f"Invalid score values: FTHG={fthg}, FTAG={ftag}, FTR={ftr}")
                
                # Check consistency between goals and outcome
                expected_ftr = "H" if fthg > ftag else ("A" if ftag > fthg else "D")
                if ftr != expected_ftr:
                    raise ValueError(
                        f"Outcome mismatch: FTHG={fthg}, FTAG={ftag} expects '{expected_ftr}' but found '{ftr}'"
                    )
            except Exception as e:
                audit.rejected_count += 1
                audit.rejections.append(f"Line {line_num}: Score validation failure: {e}")
                continue

            # 4. Team Normalisation & Match ID Generation
            canonical_home, home_slug = normalise_team_name(raw_home)
            canonical_away, away_slug = normalise_team_name(raw_away)
            division = row.get("Div", "E0").strip()
            match_id = f"{division}_{match_date}_{home_slug}_{away_slug}"

            # 5. Optional Metrics & Strict NULL Handling
            hthg = parse_optional_int(row.get("HTHG"))
            htag = parse_optional_int(row.get("HTAG"))
            htr = parse_optional_str(row.get("HTR"))
            if htr:
                htr = htr.upper()
                if htr not in ("H", "D", "A"):
                    htr = None

            referee = parse_optional_str(row.get("Referee"))
            hs = parse_optional_int(row.get("HS"))
            as_shots = parse_optional_int(row.get("AS"))
            hst = parse_optional_int(row.get("HST"))
            ast = parse_optional_int(row.get("AST"))
            hc = parse_optional_int(row.get("HC"))
            ac = parse_optional_int(row.get("AC"))
            hf = parse_optional_int(row.get("HF"))
            af = parse_optional_int(row.get("AF"))
            hy = parse_optional_int(row.get("HY"))
            ay = parse_optional_int(row.get("AY"))
            hr = parse_optional_int(row.get("HR"))
            ar = parse_optional_int(row.get("AR"))

            # Audit missing stats
            parsed_stats = {
                "HTHG": hthg, "HTAG": htag, "HTR": htr, "Referee": referee,
                "HS": hs, "AS": as_shots, "HST": hst, "AST": ast,
                "HC": hc, "AC": ac, "HF": hf, "AF": af,
                "HY": hy, "AY": ay, "HR": hr, "AR": ar,
            }
            for stat_name, val in parsed_stats.items():
                if val is None:
                    audit.missing_stats_count[stat_name] += 1

            # Prepare payloads
            match_record = {
                "match_id": match_id,
                "season": season,
                "division": division,
                "match_date": match_date,
                "match_time": match_time,
                "has_reliable_time": has_reliable_time,
                "home_team": canonical_home,
                "away_team": canonical_away,
                "raw_home_team": raw_home,
                "raw_away_team": raw_away,
                "source_file": path.name,
                "status": "FINISHED",
            }
            result_record = {
                "match_id": match_id,
                "referee": referee,
                "full_time_home_goals": fthg,
                "full_time_away_goals": ftag,
                "full_time_result": ftr,
                "half_time_home_goals": hthg,
                "half_time_away_goals": htag,
                "half_time_result": htr,
                "home_shots": hs,
                "away_shots": as_shots,
                "home_shots_on_target": hst,
                "away_shots_on_target": ast,
                "home_corners": hc,
                "away_corners": ac,
                "home_fouls": hf,
                "away_fouls": af,
                "home_yellow_cards": hy,
                "away_yellow_cards": ay,
                "home_red_cards": hr,
                "away_red_cards": ar,
            }
            matches_to_insert.append(match_record)
            results_to_insert.append(result_record)

        # Batch insert into database within single transaction
        with get_db_context(db_path) as conn:
            for match_data, result_data in zip(matches_to_insert, results_to_insert):
                insert_match(conn, match_data)
                insert_actual_result(conn, result_data)

        audit.imported_count = len(matches_to_insert)

    return audit


def ingest_all_target_seasons(db_path: Optional[str] = None) -> List[IngestionAudit]:
    """Ingest all 3 target Premier League seasons into the database."""
    from src.ingestion.downloader import RAW_DATA_DIR, SEASONS, get_raw_filepath

    audits = []
    for season in SEASONS:
        csv_file = get_raw_filepath(season)
        if not csv_file.exists():
            raise FileNotFoundError(f"Missing raw CSV for {season} at {csv_file}")
        audit = ingest_season_csv(csv_file, season=season, db_path=db_path)
        audits.append(audit)
    return audits


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("Football Intelligence - Historical Data Ingestion")
    print("=" * 60)
    
    audits = ingest_all_target_seasons()
    
    total_imported = sum(a.imported_count for a in audits)
    total_duplicates = sum(a.duplicate_count for a in audits)
    total_rejected = sum(a.rejected_count for a in audits)
    
    for a in audits:
        print(f"\nSeason: {a.season} ({a.source_file})")
        print(f"  Rows Read:   {a.total_rows_read}")
        print(f"  Imported:    {a.imported_count}")
        print(f"  Duplicates:  {a.duplicate_count}")
        print(f"  Rejected:    {a.rejected_count}")
        print(f"  Date Range:  {a.earliest_date} to {a.latest_date}")
        missing = a.missing_percentages()
        print("  Missing Stats:")
        for metric, pct in missing.items():
            print(f"    - {metric:10s}: {pct:5.2f}% ({a.missing_stats_count[metric]} missing)")

    print("\n" + "=" * 60)
    print(f"Summary: {total_imported} matches imported, {total_duplicates} duplicates, {total_rejected} rejected.")
    
    # Verify in DB
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT season, COUNT(*) FROM matches GROUP BY season ORDER BY season;")
        print("Database Season Breakdown:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} matches")
    finally:
        conn.close()
    print("=" * 60)
