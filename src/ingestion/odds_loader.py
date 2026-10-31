"""Market benchmarks and closing odds loader.
Parses closing 1X2 odds from raw CSV files into the market_benchmarks table.
Separates market benchmark data from engine predictive inputs.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.database.connection import get_connection, get_db_context
from src.ingestion.historical_loader import parse_date
from src.ingestion.team_mappings import normalise_team_name

logger = logging.getLogger(__name__)


def calculate_implied_probabilities(
    odds_h: Optional[float],
    odds_d: Optional[float],
    odds_a: Optional[float],
    tie_tolerance: float = 0.03,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """Calculate normalized implied probabilities (summing to 1.0) and determine team market favourite.
    
    Returns (implied_h, implied_d, implied_a, favourite).
    The draw is an outcome, NOT a team favourite. Favourite can only be 'H', 'A', or 'NONE'.
    If the difference between home and away probability is less than tie_tolerance, returns 'NONE'.
    """
    if odds_h is None or odds_d is None or odds_a is None:
        return None, None, None, "NONE"
    if odds_h <= 0 or odds_d <= 0 or odds_a <= 0:
        return None, None, None, "NONE"

    inv_h = 1.0 / odds_h
    inv_d = 1.0 / odds_d
    inv_a = 1.0 / odds_a
    overround = inv_h + inv_d + inv_a

    if overround <= 0:
        return None, None, None, "NONE"

    prob_h = round(inv_h / overround, 4)
    prob_d = round(inv_d / overround, 4)
    prob_a = round(inv_a / overround, 4)

    # Determine favourite with tie tolerance
    prob_diff = abs(prob_h - prob_a)
    if prob_diff < tie_tolerance:
        favourite = "NONE"
    elif prob_h > prob_a:
        favourite = "H"
    else:
        favourite = "A"

    return prob_h, prob_d, prob_a, favourite


def parse_float_safe(val: Optional[str]) -> Optional[float]:
    """Parse float safely; returns None if missing, non-numeric, NaN, or non-positive."""
    if val is None or val.strip() == "":
        return None
    try:
        f = float(val.strip())
        return f if f > 0 else None
    except ValueError:
        return None


def load_market_benchmarks_from_csv(
    csv_path: str | Path,
    db_path: Optional[str] = None,
    tie_tolerance: float = 0.03,
) -> int:
    """Load closing odds benchmarks for a single season CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    benchmarks_to_insert = []

    with path.open(mode="r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("HomeTeam") or not row.get("AwayTeam"):
                continue

            raw_home = row["HomeTeam"].strip()
            raw_away = row["AwayTeam"].strip()
            match_date = parse_date(row["Date"])
            division = row.get("Div", "E0").strip()

            canonical_home, home_slug = normalise_team_name(raw_home)
            canonical_away, away_slug = normalise_team_name(raw_away)
            match_id = f"{division}_{match_date}_{home_slug}_{away_slug}"

            # Primary: Market Average Closing Odds
            avg_h = parse_float_safe(row.get("AvgCH"))
            avg_d = parse_float_safe(row.get("AvgCD"))
            avg_a = parse_float_safe(row.get("AvgCA"))
            mkt_h, mkt_d, mkt_a, mkt_fav = calculate_implied_probabilities(
                avg_h, avg_d, avg_a, tie_tolerance=tie_tolerance
            )

            # Secondary: Bet365 Closing Odds
            b365_h = parse_float_safe(row.get("B365CH"))
            b365_d = parse_float_safe(row.get("B365CD"))
            b365_a = parse_float_safe(row.get("B365CA"))
            b_h, b_d, b_a, b_fav = calculate_implied_probabilities(
                b365_h, b365_d, b365_a, tie_tolerance=tie_tolerance
            )

            record = {
                "match_id": match_id,
                "odds_source": f"Football-Data.co.uk ({path.name})",
                "avg_closing_home_odds": avg_h,
                "avg_closing_draw_odds": avg_d,
                "avg_closing_away_odds": avg_a,
                "market_implied_home_prob": mkt_h,
                "market_implied_draw_prob": mkt_d,
                "market_implied_away_prob": mkt_a,
                "market_favourite": mkt_fav,
                "b365_closing_home_odds": b365_h,
                "b365_closing_draw_odds": b365_d,
                "b365_closing_away_odds": b365_a,
                "b365_implied_home_prob": b_h,
                "b365_implied_draw_prob": b_d,
                "b365_implied_away_prob": b_a,
                "b365_favourite": b_fav,
                "tie_tolerance": tie_tolerance,
                "source_recorded_at": None,
            }
            benchmarks_to_insert.append(record)

    sql = """
    INSERT INTO market_benchmarks (
        match_id, odds_source,
        avg_closing_home_odds, avg_closing_draw_odds, avg_closing_away_odds,
        market_implied_home_prob, market_implied_draw_prob, market_implied_away_prob, market_favourite,
        b365_closing_home_odds, b365_closing_draw_odds, b365_closing_away_odds,
        b365_implied_home_prob, b365_implied_draw_prob, b365_implied_away_prob, b365_favourite,
        tie_tolerance, source_recorded_at, imported_at
    ) VALUES (
        :match_id, :odds_source,
        :avg_closing_home_odds, :avg_closing_draw_odds, :avg_closing_away_odds,
        :market_implied_home_prob, :market_implied_draw_prob, :market_implied_away_prob, :market_favourite,
        :b365_closing_home_odds, :b365_closing_draw_odds, :b365_closing_away_odds,
        :b365_implied_home_prob, :b365_implied_draw_prob, :b365_implied_away_prob, :b365_favourite,
        :tie_tolerance, :source_recorded_at, datetime('now', 'utc')
    )
    ON CONFLICT(match_id) DO UPDATE SET
        avg_closing_home_odds = excluded.avg_closing_home_odds,
        avg_closing_draw_odds = excluded.avg_closing_draw_odds,
        avg_closing_away_odds = excluded.avg_closing_away_odds,
        market_implied_home_prob = excluded.market_implied_home_prob,
        market_implied_draw_prob = excluded.market_implied_draw_prob,
        market_implied_away_prob = excluded.market_implied_away_prob,
        market_favourite = excluded.market_favourite,
        b365_closing_home_odds = excluded.b365_closing_home_odds,
        b365_closing_draw_odds = excluded.b365_closing_draw_odds,
        b365_closing_away_odds = excluded.b365_closing_away_odds,
        b365_implied_home_prob = excluded.b365_implied_home_prob,
        b365_implied_draw_prob = excluded.b365_implied_draw_prob,
        b365_implied_away_prob = excluded.b365_implied_away_prob,
        b365_favourite = excluded.b365_favourite,
        tie_tolerance = excluded.tie_tolerance,
        imported_at = datetime('now', 'utc');
    """

    with get_db_context(db_path) as conn:
        for b in benchmarks_to_insert:
            conn.execute(sql, b)

    return len(benchmarks_to_insert)


def load_all_market_benchmarks(db_path: Optional[str] = None) -> int:
    """Load closing odds benchmarks for all 3 downloaded Premier League seasons."""
    from src.ingestion.downloader import RAW_DATA_DIR, SEASONS, get_raw_filepath

    total = 0
    for season in SEASONS:
        csv_path = get_raw_filepath(season)
        count = load_market_benchmarks_from_csv(csv_path, db_path=db_path)
        total += count
    return total


if __name__ == "__main__":
    count = load_all_market_benchmarks()
    print(f"Loaded {count} market benchmark closing odds records into SQLite.")
