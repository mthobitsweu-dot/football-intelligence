-- Football Intelligence SQLite Database Schema
-- Stage 1.1 Foundation + Stage 1.2 Enhancements + Stage 1.3 Backtesting Framework

PRAGMA foreign_keys = ON;

-- 1. MATCHES TABLE
-- Stores metadata for every match (historical or scheduled upcoming)
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT UNIQUE NOT NULL,                  -- Unique ID: e.g. "E0_2023-08-11_BUR_MCI"
    season TEXT NOT NULL,                           -- e.g. "2023-2024"
    division TEXT NOT NULL,                         -- e.g. "E0" (Premier League)
    match_date TEXT NOT NULL,                       -- ISO 8601 date: "YYYY-MM-DD"
    match_time TEXT,                                -- ISO 8601 time: "HH:MM:SS" (if available)
    has_reliable_time INTEGER NOT NULL DEFAULT 1 CHECK (has_reliable_time IN (0, 1)),
    home_team TEXT NOT NULL,                        -- Canonical standard team name
    away_team TEXT NOT NULL,                        -- Canonical standard team name
    raw_home_team TEXT,                             -- Original raw name from dataset
    raw_away_team TEXT,                             -- Original raw name from dataset
    source_file TEXT,                               -- Source CSV/file origin
    status TEXT NOT NULL DEFAULT 'FINISHED' CHECK (status IN ('SCHEDULED', 'FINISHED', 'POSTPONED', 'CANCELLED')),
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
);

-- 2. ACTUAL RESULTS TABLE
-- Stores verified post-match outcomes and statistics
CREATE TABLE IF NOT EXISTS actual_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT UNIQUE NOT NULL,
    referee TEXT,                                   -- Match referee
    full_time_home_goals INTEGER NOT NULL CHECK (full_time_home_goals >= 0),
    full_time_away_goals INTEGER NOT NULL CHECK (full_time_away_goals >= 0),
    full_time_result TEXT NOT NULL CHECK (full_time_result IN ('H', 'D', 'A')),
    half_time_home_goals INTEGER CHECK (half_time_home_goals IS NULL OR half_time_home_goals >= 0),
    half_time_away_goals INTEGER CHECK (half_time_away_goals IS NULL OR half_time_away_goals >= 0),
    half_time_result TEXT CHECK (half_time_result IS NULL OR half_time_result IN ('H', 'D', 'A')),
    home_shots INTEGER CHECK (home_shots IS NULL OR home_shots >= 0),
    away_shots INTEGER CHECK (away_shots IS NULL OR away_shots >= 0),
    home_shots_on_target INTEGER CHECK (home_shots_on_target IS NULL OR home_shots_on_target >= 0),
    away_shots_on_target INTEGER CHECK (away_shots_on_target IS NULL OR away_shots_on_target >= 0),
    home_corners INTEGER CHECK (home_corners IS NULL OR home_corners >= 0),
    away_corners INTEGER CHECK (away_corners IS NULL OR away_corners >= 0),
    home_fouls INTEGER CHECK (home_fouls IS NULL OR home_fouls >= 0),
    away_fouls INTEGER CHECK (away_fouls IS NULL OR away_fouls >= 0),
    home_yellow_cards INTEGER CHECK (home_yellow_cards IS NULL OR home_yellow_cards >= 0),
    away_yellow_cards INTEGER CHECK (away_yellow_cards IS NULL OR away_yellow_cards >= 0),
    home_red_cards INTEGER CHECK (home_red_cards IS NULL OR home_red_cards >= 0),
    away_red_cards INTEGER CHECK (away_red_cards IS NULL OR away_red_cards >= 0),
    recorded_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
);

-- 3. LIVE / UPCOMING PREDICTIONS TABLE (Untouched by Backtesting)
-- Stores predictions from any of the 6 engines
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL,
    engine_id TEXT NOT NULL CHECK (engine_id IN (
        'engine_0_master',
        'engine_1_motivation',
        'engine_2_environment',
        'engine_3_tactical',
        'engine_4_stats',
        'engine_5_squad'
    )),
    model_version TEXT NOT NULL,                    -- e.g. "v1.0.0"
    predicted_outcome TEXT NOT NULL CHECK (predicted_outcome IN ('H', 'D', 'A')),
    home_win_prob REAL NOT NULL CHECK (home_win_prob >= 0.0 AND home_win_prob <= 1.0),
    draw_prob REAL NOT NULL CHECK (draw_prob >= 0.0 AND draw_prob <= 1.0),
    away_win_prob REAL NOT NULL CHECK (away_win_prob >= 0.0 AND away_win_prob <= 1.0),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    predicted_home_goals REAL CHECK (predicted_home_goals IS NULL OR predicted_home_goals >= 0.0),
    predicted_away_goals REAL CHECK (predicted_away_goals IS NULL OR predicted_away_goals >= 0.0),
    supporting_evidence TEXT,                       -- JSON string or human-readable explanation
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    CONSTRAINT sum_prob_check CHECK (ROUND(home_win_prob + draw_prob + away_win_prob, 2) = 1.00),
    UNIQUE(match_id, engine_id, model_version)
);

-- 4. MARKET BENCHMARKS TABLE (Stage 1.3: Sourced from closing odds, isolated from engines)
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

-- 5. BACKTEST RUNS TABLE (Stage 1.3: Tracks historical backtest experiments)
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

-- 6. DEDICATED BACKTEST PREDICTIONS TABLE (Stage 1.3: Isolated per backtest run)
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

-- INDEXES
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season, division);
CREATE INDEX IF NOT EXISTS idx_actual_results_match ON actual_results(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_lookup ON predictions(match_id, engine_id);
CREATE INDEX IF NOT EXISTS idx_market_benchmarks_match ON market_benchmarks(match_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_engine ON backtest_runs(engine_id, model_version);
CREATE INDEX IF NOT EXISTS idx_bt_pred_run ON backtest_predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_bt_pred_match ON backtest_predictions(match_id);

-- 7. PREDICTION EVALUATION VIEW (Using Summed Multiclass Brier Score)
CREATE VIEW IF NOT EXISTS v_prediction_evaluations AS
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
