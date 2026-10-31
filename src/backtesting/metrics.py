"""Evaluation metrics for football prediction models.
Implements standardized summed multiclass Brier score, multiclass log loss,
3-way outcome calibration, and false-favourite performance tracking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CalibrationBin:
    """A probability calibration bin with sample count and error tracking."""
    bin_low: float
    bin_high: float
    sample_count: int = 0
    is_insufficient_sample: bool = True           # Flagged true if sample_count < min_samples (default 10)
    avg_predicted_prob: Optional[float] = None
    actual_frequency: Optional[float] = None
    calibration_error: Optional[float] = None     # ABS(avg_predicted_prob - actual_frequency)


@dataclass
class CalibrationResult:
    """Calibration breakdown across probability bins."""
    outcome_label: str
    bins: List[CalibrationBin]
    mean_calibration_error: Optional[float] = None


@dataclass
class FalseFavouriteReport:
    """Detailed evaluation of engine performance against the market favourite."""
    total_evaluated: int
    identified_market_favourites: int
    engine_disagreements: int
    market_favourite_failures: int
    successful_detections: int
    favourite_failure_when_disagreed: int
    engine_disagreement_rate: Optional[float] = None
    market_favourite_failure_rate: Optional[float] = None
    upset_prediction_accuracy: Optional[float] = None
    favourite_failure_when_disagreed_rate: Optional[float] = None


@dataclass
class AccuracyBreakdown:
    """Category-specific precision, recall, and confusion matrix."""
    total_matches: int
    correct_matches: int
    overall_accuracy: Optional[float]
    home_win_precision: Optional[float]
    home_win_recall: Optional[float]
    draw_precision: Optional[float]
    draw_recall: Optional[float]
    away_win_precision: Optional[float]
    away_win_recall: Optional[float]
    confusion_matrix: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "H": {"H": 0, "D": 0, "A": 0},
            "D": {"H": 0, "D": 0, "A": 0},
            "A": {"H": 0, "D": 0, "A": 0},
        }
    )


def calculate_brier_score(
    predictions: List[Dict[str, float]],
    actual_outcomes: List[str],
) -> Optional[float]:
    """Calculate the Standardized Summed Multiclass Brier Score (Brier, 1950).
    
    Formula:
        BS = (1 / N) * SUM_{i=1..N} SUM_{c in {H, D, A}} (p_{i,c} - y_{i,c})^2
    
    Range: [0.0, 2.0]
        0.0     = Perfect prediction (100% on true outcome)
        0.6667  = Uniform random guess (1/3 on each outcome): 3 * (1/3 - 1)^2 + 2 * (1/3 - 0)^2 = 6/9
        2.0     = Worst possible forecast (100% certainty on an outcome that did not occur)
    """
    if not predictions or len(predictions) != len(actual_outcomes):
        return None

    total_error = 0.0
    for pred, actual in zip(predictions, actual_outcomes):
        y_h = 1.0 if actual == "H" else 0.0
        y_d = 1.0 if actual == "D" else 0.0
        y_a = 1.0 if actual == "A" else 0.0

        p_h = float(pred["home_win_prob"])
        p_d = float(pred["draw_prob"])
        p_a = float(pred["away_win_prob"])

        match_brier = (p_h - y_h) ** 2 + (p_d - y_d) ** 2 + (p_a - y_a) ** 2
        total_error += match_brier

    return round(total_error / len(predictions), 6)


def calculate_log_loss(
    predictions: List[Dict[str, float]],
    actual_outcomes: List[str],
    eps: float = 1e-15,
) -> Optional[float]:
    """Calculate Multiclass Log Loss (Cross-Entropy).
    
    Formula:
        LogLoss = - (1 / N) * SUM_{i=1..N} SUM_{c in {H, D, A}} y_{i,c} * ln(max(p_{i,c}, eps))
    """
    if not predictions or len(predictions) != len(actual_outcomes):
        return None

    total_loss = 0.0
    for pred, actual in zip(predictions, actual_outcomes):
        if actual == "H":
            p = max(float(pred["home_win_prob"]), eps)
        elif actual == "D":
            p = max(float(pred["draw_prob"]), eps)
        elif actual == "A":
            p = max(float(pred["away_win_prob"]), eps)
        else:
            continue
        total_loss += -math.log(p)

    return round(total_loss / len(predictions), 6)


def calculate_calibration_for_outcome(
    predicted_probs: List[float],
    actual_flags: List[int],
    num_bins: int = 5,
    min_samples: int = 10,
    outcome_label: str = "Outcome",
) -> CalibrationResult:
    """Calculate calibration within probability bins for a single binary indicator.
    
    For each bin [low, high), computes:
        calibration_error = ABS(avg_predicted_prob - actual_frequency)
    Reports exact sample counts and flags bins with fewer than min_samples.
    """
    if not predicted_probs or len(predicted_probs) != len(actual_flags):
        return CalibrationResult(outcome_label=outcome_label, bins=[])

    step = 1.0 / num_bins
    bin_intervals: List[Tuple[float, float]] = []
    for i in range(num_bins):
        low = round(i * step, 4)
        high = round((i + 1) * step, 4) if i < num_bins - 1 else 1.0
        bin_intervals.append((low, high))

    bins: List[CalibrationBin] = []
    total_weighted_error = 0.0
    total_evaluated_samples = 0

    for low, high in bin_intervals:
        # Collect items in [low, high) - last bin is [low, 1.0]
        bin_probs: List[float] = []
        bin_actuals: List[int] = []

        for p, y in zip(predicted_probs, actual_flags):
            if (low <= p < high) or (high >= 1.0 and p == 1.0):
                bin_probs.append(p)
                bin_actuals.append(y)

        n = len(bin_probs)
        is_insufficient = n < min_samples

        if n > 0:
            avg_p = sum(bin_probs) / n
            act_freq = sum(bin_actuals) / n
            cal_err = abs(avg_p - act_freq)
            total_weighted_error += cal_err * n
            total_evaluated_samples += n
        else:
            avg_p = None
            act_freq = None
            cal_err = None

        bins.append(
            CalibrationBin(
                bin_low=low,
                bin_high=high,
                sample_count=n,
                is_insufficient_sample=is_insufficient,
                avg_predicted_prob=round(avg_p, 4) if avg_p is not None else None,
                actual_frequency=round(act_freq, 4) if act_freq is not None else None,
                calibration_error=round(cal_err, 4) if cal_err is not None else None,
            )
        )

    mean_err = (
        round(total_weighted_error / total_evaluated_samples, 4)
        if total_evaluated_samples > 0
        else None
    )

    return CalibrationResult(
        outcome_label=outcome_label,
        bins=bins,
        mean_calibration_error=mean_err,
    )


def calculate_3way_calibration(
    predictions: List[Dict[str, float]],
    actual_outcomes: List[str],
    num_bins: int = 5,
    min_samples: int = 10,
) -> Dict[str, CalibrationResult]:
    """Calculate 4 separate calibration analyses: Home Win, Draw, Away Win, and Confidence."""
    home_probs = [float(p["home_win_prob"]) for p in predictions]
    home_actuals = [1 if a == "H" else 0 for a in actual_outcomes]

    draw_probs = [float(p["draw_prob"]) for p in predictions]
    draw_actuals = [1 if a == "D" else 0 for a in actual_outcomes]

    away_probs = [float(p["away_win_prob"]) for p in predictions]
    away_actuals = [1 if a == "A" else 0 for a in actual_outcomes]

    # Confidence is strictly max(home, draw, away)
    conf_probs = [
        max(float(p["home_win_prob"]), float(p["draw_prob"]), float(p["away_win_prob"]))
        for p in predictions
    ]
    conf_actuals = [
        1 if p["predicted_outcome"] == a else 0
        for p, a in zip(predictions, actual_outcomes)
    ]

    return {
        "home": calculate_calibration_for_outcome(
            home_probs, home_actuals, num_bins, min_samples, "Home Win"
        ),
        "draw": calculate_calibration_for_outcome(
            draw_probs, draw_actuals, num_bins, min_samples, "Draw"
        ),
        "away": calculate_calibration_for_outcome(
            away_probs, away_actuals, num_bins, min_samples, "Away Win"
        ),
        "confidence": calculate_calibration_for_outcome(
            conf_probs, conf_actuals, num_bins, min_samples, "Prediction Confidence"
        ),
    }


def calculate_accuracy_breakdown(
    predicted_outcomes: List[str],
    actual_outcomes: List[str],
) -> AccuracyBreakdown:
    """Calculate overall accuracy, category precision & recall, and confusion matrix."""
    n = len(predicted_outcomes)
    if n == 0 or n != len(actual_outcomes):
        return AccuracyBreakdown(
            total_matches=0,
            correct_matches=0,
            overall_accuracy=None,
            home_win_precision=None,
            home_win_recall=None,
            draw_precision=None,
            draw_recall=None,
            away_win_precision=None,
            away_win_recall=None,
        )

    matrix = {
        "H": {"H": 0, "D": 0, "A": 0},
        "D": {"H": 0, "D": 0, "A": 0},
        "A": {"H": 0, "D": 0, "A": 0},
    }

    correct = 0
    for pred, actual in zip(predicted_outcomes, actual_outcomes):
        if pred == actual:
            correct += 1
        if pred in matrix and actual in matrix[pred]:
            matrix[pred][actual] += 1

    overall_acc = round(correct / n, 4)

    def calc_prec_recall(cat: str) -> Tuple[Optional[float], Optional[float]]:
        # Precision: actual=cat given predicted=cat
        predicted_as_cat = sum(matrix[cat].values())
        true_positives = matrix[cat][cat]
        precision = (
            round(true_positives / predicted_as_cat, 4)
            if predicted_as_cat > 0
            else None
        )

        # Recall: predicted=cat given actual=cat
        actual_is_cat = sum(matrix[c][cat] for c in ("H", "D", "A"))
        recall = (
            round(true_positives / actual_is_cat, 4)
            if actual_is_cat > 0
            else None
        )
        return precision, recall

    h_prec, h_rec = calc_prec_recall("H")
    d_prec, d_rec = calc_prec_recall("D")
    a_prec, a_rec = calc_prec_recall("A")

    return AccuracyBreakdown(
        total_matches=n,
        correct_matches=correct,
        overall_accuracy=overall_acc,
        home_win_precision=h_prec,
        home_win_recall=h_rec,
        draw_precision=d_prec,
        draw_recall=d_rec,
        away_win_precision=a_prec,
        away_win_recall=a_rec,
        confusion_matrix=matrix,
    )


def calculate_false_favourite_metrics(
    predictions: List[Dict[str, Any]],
    actual_outcomes: List[str],
    market_benchmarks: List[Dict[str, Any]],
) -> FalseFavouriteReport:
    """Calculate false-favourite performance against AvgC (market average closing odds).
    
    Rules:
      1. Market favourite F_mkt in {'H', 'A'}. If 'NONE', match is excluded from favourite stats.
      2. Engine Disagreement: P_engine != F_mkt.
      3. Favourite Failure: R_actual != F_mkt (Draw or Underdog win).
      4. Successful Detection: P_engine != F_mkt AND P_engine == R_actual.
         (Draw only counts as success if engine predicted Draw; Underdog win only if engine predicted Underdog).
    """
    total = len(predictions)
    identified = 0
    disagreements = 0
    fav_failures = 0
    successful_detections = 0
    fav_failure_when_disagreed = 0

    for pred, actual, bench in zip(predictions, actual_outcomes, market_benchmarks):
        fav = bench.get("market_favourite", "NONE")
        if fav not in ("H", "A"):
            continue  # Exclude evenly matched games

        identified += 1
        predicted = pred["predicted_outcome"]

        fav_failed = (actual != fav)
        if fav_failed:
            fav_failures += 1

        engine_disagreed = (predicted != fav)
        if engine_disagreed:
            disagreements += 1
            if fav_failed:
                fav_failure_when_disagreed += 1
            # Exact upset prediction: engine must predict the exact outcome that occurred
            if predicted == actual:
                successful_detections += 1

    disagree_rate = (
        round(disagreements / identified, 4) if identified > 0 else None
    )
    failure_rate = (
        round(fav_failures / identified, 4) if identified > 0 else None
    )
    upset_acc = (
        round(successful_detections / disagreements, 4)
        if disagreements > 0
        else None
    )
    fail_when_disagree_rate = (
        round(fav_failure_when_disagreed / disagreements, 4)
        if disagreements > 0
        else None
    )

    return FalseFavouriteReport(
        total_evaluated=total,
        identified_market_favourites=identified,
        engine_disagreements=disagreements,
        market_favourite_failures=fav_failures,
        successful_detections=successful_detections,
        favourite_failure_when_disagreed=fav_failure_when_disagreed,
        engine_disagreement_rate=disagree_rate,
        market_favourite_failure_rate=failure_rate,
        upset_prediction_accuracy=upset_acc,
        favourite_failure_when_disagreed_rate=fail_when_disagree_rate,
    )
