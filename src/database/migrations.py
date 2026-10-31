"""Database migration utility for Football Intelligence.
Safely updates existing SQLite tables without data loss.
"""
from __future__ import annotations

import sqlite3
from typing import Set

from src.database.connection import get_connection, get_db_context


def get_existing_tables(conn: sqlite3.Connection) -> Set[str]:
    """Return a set of table names currently in the database."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return {row["name"] for row in cursor.fetchall()}


def get_existing_columns(conn: sqlite3.Connection, table_name: str) -> Set[str]:
    """Return a set of column names currently in the specified table."""
    cursor = conn.execute(f"PRAGMA table_info({table_name});")
    return {row["name"] for row in cursor.fetchall()}


def apply_migrations(db_path: str | None = None) -> None:
    """Apply safe migrations to bring the database up to the Stage 1.3 schema."""
    with get_db_context(db_path) as conn:
        tables = get_existing_tables(conn)

        # 1. Matches table migrations
        if "matches" in tables:
            match_cols = get_existing_columns(conn, "matches")
            if "raw_home_team" not in match_cols:
                conn.execute("ALTER TABLE matches ADD COLUMN raw_home_team TEXT;")
            if "raw_away_team" not in match_cols:
                conn.execute("ALTER TABLE matches ADD COLUMN raw_away_team TEXT;")
            if "source_file" not in match_cols:
                conn.execute("ALTER TABLE matches ADD COLUMN source_file TEXT;")
            if "has_reliable_time" not in match_cols:
                conn.execute("ALTER TABLE matches ADD COLUMN has_reliable_time INTEGER NOT NULL DEFAULT 1;")

        # 2. Actual results table migrations
        if "actual_results" in tables:
            result_cols = get_existing_columns(conn, "actual_results")
            if "referee" not in result_cols:
                conn.execute("ALTER TABLE actual_results ADD COLUMN referee TEXT;")

        # 3. Market benchmarks table (Stage 1.3: Separates market odds from matches)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS market_benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT UNIQUE NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
            odds_source TEXT NOT NULL,
            -- Primary Benchmark: Market Average Closing Odds (AvgC)
            avg_closing_home_odds REAL,
            avg_closing_draw_odds REAL,
            avg_closing_away_odds REAL,
            market_implied_home_prob REAL,
            market_implied_draw_prob REAL,
            market_implied_away_prob REAL,
            market_favourite TEXT CHECK (market_favourite IN ('H', 'A', 'NONE')),
            -- Secondary Benchmark: Bet365 Closing Odds (B365C)
            b365_closing_home_odds REAL,
            b365_closing_draw_odds REAL,
            b365_closing_away_odds REAL,
            b365_implied_home_prob REAL,
            b365_implied_draw_prob REAL,
            b365_implied_away_prob REAL,
            b365_favourite TEXT CHECK (b365_favourite IN ('H', 'A', 'NONE')),
            tie_tolerance REAL NOT NULL DEFAULT 0.03,
            source_recorded_at TEXT,
            imported_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_market_benchmarks_match ON market_benchmarks(match_id);")

        # 4. Backtest runs table (Stage 1.3: Supports complete, partial, and failed runs)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            engine_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            config_version TEXT NOT NULL,
            season TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            temporal_cutoff_policy TEXT NOT NULL,
            dataset_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'RUNNING' CHECK (status IN ('RUNNING', 'COMPLETED', 'PARTIAL', 'FAILED')),
            total_evaluated_matches INTEGER DEFAULT 0,
            accuracy REAL,
            home_win_accuracy REAL,
            draw_accuracy REAL,
            away_win_accuracy REAL,
            brier_score REAL,
            log_loss REAL,
            calibration_home_error REAL,
            calibration_draw_error REAL,
            calibration_away_error REAL,
            calibration_conf_error REAL,
            false_fav_identified INTEGER,
            false_fav_disagree INTEGER,
            false_fav_failures INTEGER,
            false_fav_successful_detections INTEGER,
            notes TEXT,
            started_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
            completed_at TEXT
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backtest_runs_engine ON backtest_runs(engine_id, model_version);")

        # 5. Dedicated backtest_predictions table (Stage 1.3: Decoupled from live predictions)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
            match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
            engine_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            predicted_outcome TEXT NOT NULL CHECK (predicted_outcome IN ('H', 'D', 'A')),
            home_win_prob REAL NOT NULL CHECK (home_win_prob >= 0.0 AND home_win_prob <= 1.0),
            draw_prob REAL NOT NULL CHECK (draw_prob >= 0.0 AND draw_prob <= 1.0),
            away_win_prob REAL NOT NULL CHECK (away_win_prob >= 0.0 AND away_win_prob <= 1.0),
            confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            prediction_status TEXT NOT NULL DEFAULT 'OK' CHECK (prediction_status IN ('OK', 'INSUFFICIENT_DATA', 'DEGRADED', 'ERROR')),
            home_sample_size INTEGER NOT NULL DEFAULT 0 CHECK (home_sample_size >= 0),
            away_sample_size INTEGER NOT NULL DEFAULT 0 CHECK (away_sample_size >= 0),
            supporting_evidence TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
            CONSTRAINT backtest_prob_sum CHECK (ABS(home_win_prob + draw_prob + away_win_prob - 1.0) < 0.000001),
            UNIQUE(run_id, match_id, engine_id)
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bt_pred_run ON backtest_predictions(run_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bt_pred_match ON backtest_predictions(match_id);")

        # 6. Standardized Summed Multiclass Brier Score in evaluation view
        conn.execute("DROP VIEW IF EXISTS v_prediction_evaluations;")
        conn.execute("""
        CREATE VIEW v_prediction_evaluations AS
        SELECT 
            p.id AS prediction_id,
            p.match_id,
            m.season,
            m.division,
            m.match_date,
            m.match_time,
            m.has_reliable_time,
            m.home_team,
            m.away_team,
            m.raw_home_team,
            m.raw_away_team,
            m.source_file,
            p.engine_id,
            p.model_version,
            p.predicted_outcome,
            p.home_win_prob,
            p.draw_prob,
            p.away_win_prob,
            p.confidence,
            r.full_time_result AS actual_result,
            r.full_time_home_goals AS actual_home_goals,
            r.full_time_away_goals AS actual_away_goals,
            r.referee,
            CASE 
                WHEN p.predicted_outcome = r.full_time_result THEN 1 
                ELSE 0 
            END AS is_correct,
            -- Standardized Summed Multiclass Brier Score (Range [0.0, 2.0], Uniform Baseline = 6/9 ~ 0.6667)
            (
                (p.home_win_prob - CASE WHEN r.full_time_result = 'H' THEN 1.0 ELSE 0.0 END) * 
                (p.home_win_prob - CASE WHEN r.full_time_result = 'H' THEN 1.0 ELSE 0.0 END) +
                (p.draw_prob - CASE WHEN r.full_time_result = 'D' THEN 1.0 ELSE 0.0 END) * 
                (p.draw_prob - CASE WHEN r.full_time_result = 'D' THEN 1.0 ELSE 0.0 END) +
                (p.away_win_prob - CASE WHEN r.full_time_result = 'A' THEN 1.0 ELSE 0.0 END) * 
                (p.away_win_prob - CASE WHEN r.full_time_result = 'A' THEN 1.0 ELSE 0.0 END)
            ) AS brier_score,
            p.supporting_evidence,
            p.created_at AS predicted_at,
            r.recorded_at AS result_recorded_at
        FROM predictions p
        JOIN matches m ON p.match_id = m.match_id
        JOIN actual_results r ON p.match_id = r.match_id;
        """)
