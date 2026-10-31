"""Reproducible CLI runner for Football Intelligence historical backtests.
Executes an engine against a historical season under strict date isolation
and prints a formatted performance report.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from src.backtesting.evaluator import BacktestEvaluator, BacktestResult
from src.engines.base import BaseEngine, EnginePrediction


class NaiveBaselineEngine:
    """A naive home-advantage baseline engine for benchmarking and harness verification.
    Predicts Home Win with 45% prob, Draw with 27% prob, Away with 28% prob.
    """
    engine_id: str = "engine_baseline_home"
    model_version: str = "v1.0.0"

    def predict(self, context) -> EnginePrediction:
        return EnginePrediction(
            engine_id=self.engine_id,
            model_version=self.model_version,
            predicted_outcome="H",
            home_win_prob=0.45,
            draw_prob=0.27,
            away_win_prob=0.28,
            confidence=0.45,
            prediction_status="OK",
            supporting_evidence="Naive historical home-win baseline",
        )


def format_backtest_report(result: BacktestResult) -> str:
    """Format a BacktestResult into a clean terminal report."""
    lines = [
        "=" * 70,
        f"Football Intelligence - Backtest Report",
        "=" * 70,
        f"Run ID:                 {result.run_id}",
        f"Engine:                 {result.engine_id} ({result.model_version})",
        f"Season:                 {result.season}",
        f"Status:                 {result.status}",
        f"Evaluated Matches:      {result.total_evaluated_matches}",
        "-" * 70,
    ]

    if result.accuracy_breakdown and result.total_evaluated_matches > 0:
        ab = result.accuracy_breakdown
        lines.extend([
            "PREDICTION ACCURACY:",
            f"  Overall Accuracy:     {ab.overall_accuracy * 100:.2f}% ({ab.correct_matches}/{ab.total_matches})" if ab.overall_accuracy is not None else "  Overall Accuracy:     N/A",
            f"  Home Win Precision:   {ab.home_win_precision * 100:.2f}% | Recall: {ab.home_win_recall * 100:.2f}%" if ab.home_win_precision is not None else "  Home Win:             N/A",
            f"  Draw Precision:       {ab.draw_precision * 100:.2f}% | Recall: {ab.draw_recall * 100:.2f}%" if ab.draw_precision is not None else "  Draw:                 N/A",
            f"  Away Win Precision:   {ab.away_win_precision * 100:.2f}% | Recall: {ab.away_win_recall * 100:.2f}%" if ab.away_win_precision is not None else "  Away Win:             N/A",
            "",
            "CONFUSION MATRIX (Predicted \\ Actual):",
            f"               Actual H    Actual D    Actual A",
            f"  Predicted H:  {ab.confusion_matrix['H']['H']:8d}    {ab.confusion_matrix['H']['D']:8d}    {ab.confusion_matrix['H']['A']:8d}",
            f"  Predicted D:  {ab.confusion_matrix['D']['H']:8d}    {ab.confusion_matrix['D']['D']:8d}    {ab.confusion_matrix['D']['A']:8d}",
            f"  Predicted A:  {ab.confusion_matrix['A']['H']:8d}    {ab.confusion_matrix['A']['D']:8d}    {ab.confusion_matrix['A']['A']:8d}",
            "-" * 70,
        ])

    if result.brier_score is not None:
        lines.extend([
            "PROBABILISTIC CALIBRATION & ERROR:",
            f"  Summed Brier Score:   {result.brier_score:.4f}  (Random Baseline: 0.6667, Perfect: 0.0)",
            f"  Multiclass Log Loss:  {result.log_loss:.4f}" if result.log_loss is not None else "  Log Loss:             N/A",
            "-" * 70,
        ])

    if result.calibration:
        lines.append("CALIBRATION BY OUTCOME (Bin Range | Samples | Pred Avg | Actual Freq | Error):")
        for outcome_key, cal_res in result.calibration.items():
            lines.append(f"  [{cal_res.outcome_label}] (Mean Error: {cal_res.mean_calibration_error:.4f})" if cal_res.mean_calibration_error is not None else f"  [{cal_res.outcome_label}]")
            for b in cal_res.bins:
                flag = " [!] Small sample (<10)" if b.is_insufficient_sample and b.sample_count > 0 else ""
                pred_str = f"{b.avg_predicted_prob:.4f}" if b.avg_predicted_prob is not None else "   N/A"
                act_str = f"{b.actual_frequency:.4f}" if b.actual_frequency is not None else "   N/A"
                err_str = f"{b.calibration_error:.4f}" if b.calibration_error is not None else "   N/A"
                lines.append(
                    f"    {b.bin_low:4.2f} - {b.bin_high:4.2f} : {b.sample_count:4d} matches | Pred: {pred_str} | Act: {act_str} | Err: {err_str}{flag}"
                )
        lines.append("-" * 70)

    if result.status_breakdown:
        lines.append("PERFORMANCE BREAKDOWN BY PREDICTION STATUS:")
        for st_name, st_data in result.status_breakdown.items():
            brier_str = f"{st_data['brier_score']:.4f}" if st_data['brier_score'] is not None else "N/A"
            loss_str = f"{st_data['log_loss']:.4f}" if st_data['log_loss'] is not None else "N/A"
            lines.append(
                f"  [{st_name:17s}] : {st_data['count']:4d} matches | Accuracy: {st_data['accuracy']*100:5.2f}% | Brier: {brier_str} | LogLoss: {loss_str}"
            )
        lines.append("-" * 70)

    if result.false_favourites and result.false_favourites.identified_market_favourites > 0:
        ff = result.false_favourites
        lines.extend([
            "FALSE-FAVOURITE ANALYSIS (Primary Benchmark: AvgC Market Average):",
            f"  Market Favourites Identified:     {ff.identified_market_favourites}",
            f"  Market Favourite Failure Rate:    {ff.market_favourite_failure_rate * 100:.2f}% ({ff.market_favourite_failures}/{ff.identified_market_favourites} failed to win)" if ff.market_favourite_failure_rate is not None else "  Favourite Failure Rate:           N/A",
            f"  Engine Disagreements:             {ff.engine_disagreements} ({ff.engine_disagreement_rate * 100:.2f}% of favourites)" if ff.engine_disagreement_rate is not None else "  Engine Disagreements:             0",
            f"  Successful Upset Detections:      {ff.successful_detections}" + (f" ({ff.upset_prediction_accuracy * 100:.2f}% accuracy when disagreeing)" if ff.upset_prediction_accuracy is not None else ""),
            f"  Favourite Failed When Disagreed:  {ff.favourite_failure_when_disagreed}" + (f" ({ff.favourite_failure_when_disagreed_rate * 100:.2f}% failure rate)" if ff.favourite_failure_when_disagreed_rate is not None else ""),
            "=" * 70,
        ])

    return "\n".join(lines)


def run_cli():
    parser = argparse.ArgumentParser(description="Football Intelligence Backtest Runner")
    parser.add_argument(
        "--engine",
        type=str,
        default="engine_4_stats",
        choices=["engine_4_stats", "baseline"],
        help="Engine to evaluate (default: engine_4_stats)",
    )
    parser.add_argument("--season", type=str, default="2023-2024", help="Season to backtest (default: 2023-2024)")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Model version")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--config", type=str, default="default", help="Configuration label")
    args = parser.parse_args()

    engine: BaseEngine
    if args.engine == "engine_4_stats":
        from src.engines.engine_4_stats import Engine4Statistical
        engine = Engine4Statistical()
        engine.model_version = args.version
    else:
        engine = NaiveBaselineEngine()
        engine.model_version = args.version

    evaluator = BacktestEvaluator()
    print(f"Executing backtest for {engine.engine_id} ({engine.model_version}) on season {args.season}...")
    result = evaluator.evaluate_engine(
        engine=engine,
        season=args.season,
        start_date=args.start_date,
        end_date=args.end_date,
        config_version=args.config,
    )
    print(format_backtest_report(result))


if __name__ == "__main__":
    run_cli()
