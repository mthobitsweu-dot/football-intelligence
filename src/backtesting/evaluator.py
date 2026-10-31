"""Backtest evaluator engine.
Runs any prediction engine (Engine #0 through #5) across historical matches.
Guarantees strict calendar-date isolation, immutable backtest predictions,
and detailed metric calculations with support for incomplete runs.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.backtesting.metrics import (
    AccuracyBreakdown,
    CalibrationResult,
    FalseFavouriteReport,
    calculate_3way_calibration,
    calculate_accuracy_breakdown,
    calculate_brier_score,
    calculate_false_favourite_metrics,
    calculate_log_loss,
)
from src.backtesting.time_machine import TimeMachine
from src.database.connection import get_connection, get_db_context
from src.engines.base import BaseEngine, EnginePrediction


@dataclass
class BacktestResult:
    """Full backtest evaluation output."""
    run_id: str
    engine_id: str
    model_version: str
    season: str
    status: str
    total_evaluated_matches: int
    accuracy_breakdown: Optional[AccuracyBreakdown] = None
    brier_score: Optional[float] = None
    log_loss: Optional[float] = None
    calibration: Optional[Dict[str, CalibrationResult]] = None
    false_favourites: Optional[FalseFavouriteReport] = None
    status_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class BacktestEvaluator:
    """Executes and logs reproducible backtests."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.time_machine = TimeMachine(db_path=db_path)

    def _compute_dataset_fingerprint(self, conn: sqlite3.Connection) -> str:
        """Compute an audit fingerprint of the matches and results tables."""
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), MAX(match_date), MIN(match_date) FROM matches;")
        row = cur.fetchone()
        raw_str = f"matches:{row[0]}:{row[1]}:{row[2]}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]

    def evaluate_engine(
        self,
        engine: BaseEngine,
        season: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        config_version: str = "default",
        run_id: Optional[str] = None,
    ) -> BacktestResult:
        """Execute a backtest run of engine over historical matches in season.
        
        Saves individual predictions to backtest_predictions and run audit to backtest_runs.
        Does NOT alter the live predictions table.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if run_id is None:
            clean_engine = engine.engine_id.replace("engine_", "ENG")
            clean_ver = engine.model_version.replace(".", "").upper()
            run_id = f"RUN_{now_str}_{clean_engine}_{clean_ver}_{season.replace('-', '_')}"

        matches = self.time_machine.get_backtest_matches(
            season=season, start_date=start_date, end_date=end_date
        )

        with get_db_context(self.db_path) as conn:
            fingerprint = self._compute_dataset_fingerprint(conn)
            # 1. Initialize backtest_runs record as RUNNING
            conn.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, engine_id, model_version, config_version, season,
                    start_date, end_date, temporal_cutoff_policy, dataset_fingerprint,
                    status, total_evaluated_matches, started_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, 'RUNNING', 0, datetime('now', 'utc')
                );
                """,
                (
                    run_id,
                    engine.engine_id,
                    engine.model_version,
                    config_version,
                    season,
                    start_date or (matches[0]["match_date"] if matches else "N/A"),
                    end_date or (matches[-1]["match_date"] if matches else "N/A"),
                    "STRICT_DATE: match_date < target_date",
                    fingerprint,
                ),
            )

        if not matches:
            with get_db_context(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE backtest_runs
                    SET status = 'COMPLETED', completed_at = datetime('now', 'utc')
                    WHERE run_id = ?;
                    """,
                    (run_id,),
                )
            return BacktestResult(
                run_id=run_id,
                engine_id=engine.engine_id,
                model_version=engine.model_version,
                season=season,
                status="COMPLETED",
                total_evaluated_matches=0,
            )

        predictions_list: List[Dict[str, Any]] = []
        actuals_list: List[str] = []
        benchmarks_list: List[Dict[str, Any]] = []

        try:
            with get_db_context(self.db_path) as conn:
                for match in matches:
                    mid = match["match_id"]
                    context = self.time_machine.get_match_context(mid)
                    pred = engine.predict(context)

                    # Ensure confidence is strictly max of probabilities
                    conf = max(pred.home_win_prob, pred.draw_prob, pred.away_win_prob)

                    # Save to backtest_predictions
                    conn.execute(
                        """
                        INSERT INTO backtest_predictions (
                            run_id, match_id, engine_id, model_version,
                            predicted_outcome, home_win_prob, draw_prob, away_win_prob,
                            confidence, prediction_status, home_sample_size, away_sample_size,
                            supporting_evidence, created_at
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'utc')
                        );
                        """,
                        (
                            run_id,
                            mid,
                            engine.engine_id,
                            engine.model_version,
                            pred.predicted_outcome,
                            pred.home_win_prob,
                            pred.draw_prob,
                            pred.away_win_prob,
                            conf,
                            pred.prediction_status,
                            pred.home_sample_size,
                            pred.away_sample_size,
                            pred.supporting_evidence,
                        ),
                    )

                    # Query actual result and market benchmark post-prediction
                    cur = conn.execute(
                        "SELECT full_time_result FROM actual_results WHERE match_id = ?",
                        (mid,),
                    )
                    act_row = cur.fetchone()
                    if act_row:
                        actuals_list.append(act_row["full_time_result"])

                        # Market benchmark
                        bench_cur = conn.execute(
                            "SELECT market_favourite, avg_closing_home_odds, avg_closing_draw_odds, avg_closing_away_odds FROM market_benchmarks WHERE match_id = ?",
                            (mid,),
                        )
                        b_row = bench_cur.fetchone()
                        benchmarks_list.append(dict(b_row) if b_row else {})

                        predictions_list.append({
                            "home_win_prob": pred.home_win_prob,
                            "draw_prob": pred.draw_prob,
                            "away_win_prob": pred.away_win_prob,
                            "predicted_outcome": pred.predicted_outcome,
                            "prediction_status": pred.prediction_status,
                        })

            # Calculate metrics
            acc_breakdown = calculate_accuracy_breakdown(
                [p["predicted_outcome"] for p in predictions_list], actuals_list
            )
            brier = calculate_brier_score(predictions_list, actuals_list)
            logloss = calculate_log_loss(predictions_list, actuals_list)
            cal_dict = calculate_3way_calibration(predictions_list, actuals_list)
            false_fav_report = calculate_false_favourite_metrics(
                predictions_list, actuals_list, benchmarks_list
            )

            # Performance breakdown by prediction_status (OK vs INSUFFICIENT_DATA)
            status_groups: Dict[str, Dict[str, List[Any]]] = {}
            for p, act in zip(predictions_list, actuals_list):
                st = p.get("prediction_status", "OK")
                if st not in status_groups:
                    status_groups[st] = {"preds": [], "actuals": []}
                status_groups[st]["preds"].append(p)
                status_groups[st]["actuals"].append(act)

            status_breakdown: Dict[str, Dict[str, Any]] = {}
            for st, grp in status_groups.items():
                g_preds = grp["preds"]
                g_acts = grp["actuals"]
                g_acc = sum(1 for p, a in zip(g_preds, g_acts) if p["predicted_outcome"] == a) / len(g_preds)
                status_breakdown[st] = {
                    "count": len(g_preds),
                    "accuracy": round(g_acc, 4),
                    "brier_score": calculate_brier_score(g_preds, g_acts),
                    "log_loss": calculate_log_loss(g_preds, g_acts),
                }

            # Update backtest_runs record with completed metrics
            with get_db_context(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE backtest_runs SET
                        status = 'COMPLETED',
                        total_evaluated_matches = ?,
                        accuracy = ?,
                        home_win_accuracy = ?,
                        draw_accuracy = ?,
                        away_win_accuracy = ?,
                        brier_score = ?,
                        log_loss = ?,
                        calibration_home_error = ?,
                        calibration_draw_error = ?,
                        calibration_away_error = ?,
                        calibration_conf_error = ?,
                        false_fav_identified = ?,
                        false_fav_disagree = ?,
                        false_fav_failures = ?,
                        false_fav_successful_detections = ?,
                        completed_at = datetime('now', 'utc')
                    WHERE run_id = ?;
                    """,
                    (
                        len(predictions_list),
                        acc_breakdown.overall_accuracy,
                        acc_breakdown.home_win_precision,
                        acc_breakdown.draw_precision,
                        acc_breakdown.away_win_precision,
                        brier,
                        logloss,
                        cal_dict["home"].mean_calibration_error,
                        cal_dict["draw"].mean_calibration_error,
                        cal_dict["away"].mean_calibration_error,
                        cal_dict["confidence"].mean_calibration_error,
                        false_fav_report.identified_market_favourites,
                        false_fav_report.engine_disagreements,
                        false_fav_report.market_favourite_failures,
                        false_fav_report.successful_detections,
                        run_id,
                    ),
                )

            return BacktestResult(
                run_id=run_id,
                engine_id=engine.engine_id,
                model_version=engine.model_version,
                season=season,
                status="COMPLETED",
                total_evaluated_matches=len(predictions_list),
                accuracy_breakdown=acc_breakdown,
                brier_score=brier,
                log_loss=logloss,
                calibration=cal_dict,
                false_favourites=false_fav_report,
                status_breakdown=status_breakdown,
            )

        except Exception as e:
            # Mark run as FAILED or PARTIAL; leave metrics as NULL
            with get_db_context(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE backtest_runs SET
                        status = 'FAILED',
                        notes = ?,
                        completed_at = datetime('now', 'utc')
                    WHERE run_id = ?;
                    """,
                    (f"Error: {e}", run_id),
                )
            raise
