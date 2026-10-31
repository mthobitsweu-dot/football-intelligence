"""Unit tests for statistical evaluation metrics.
Validates standardized summed multiclass Brier score, multiclass log loss,
calibration binning with sample counts, and false-favourite definitions.
"""
import math
import unittest

from src.backtesting.metrics import (
    calculate_3way_calibration,
    calculate_accuracy_breakdown,
    calculate_brier_score,
    calculate_calibration_for_outcome,
    calculate_false_favourite_metrics,
    calculate_log_loss,
)
from src.ingestion.odds_loader import calculate_implied_probabilities


class TestBrierScoreAndLogLoss(unittest.TestCase):
    def test_brier_score_perfect_prediction(self):
        """A perfect prediction (1.0 on the true outcome) must yield a Brier score of 0.0."""
        preds = [{"home_win_prob": 1.0, "draw_prob": 0.0, "away_win_prob": 0.0}]
        actuals = ["H"]
        brier = calculate_brier_score(preds, actuals)
        self.assertEqual(brier, 0.0)

    def test_brier_score_worst_prediction(self):
        """A worst possible prediction (1.0 on wrong outcome) must yield a Brier score of 2.0."""
        preds = [{"home_win_prob": 1.0, "draw_prob": 0.0, "away_win_prob": 0.0}]
        actuals = ["A"]
        # (1-0)^2 + (0-0)^2 + (0-1)^2 = 1 + 0 + 1 = 2.0
        brier = calculate_brier_score(preds, actuals)
        self.assertEqual(brier, 2.0)

    def test_brier_score_uniform_baseline(self):
        """Uniform random guess (1/3, 1/3, 1/3) must yield 6/9 ~ 0.666667."""
        preds = [{"home_win_prob": 1/3, "draw_prob": 1/3, "away_win_prob": 1/3}]
        actuals = ["H"]
        brier = calculate_brier_score(preds, actuals)
        self.assertAlmostEqual(brier, 6 / 9, places=5)

    def test_log_loss_uniform_baseline(self):
        """Uniform guess (1/3) should yield -ln(1/3) = ln(3) ~ 1.098612."""
        preds = [{"home_win_prob": 1/3, "draw_prob": 1/3, "away_win_prob": 1/3}]
        actuals = ["H"]
        loss = calculate_log_loss(preds, actuals)
        self.assertAlmostEqual(loss, math.log(3), places=4)


class TestCalibrationCalculations(unittest.TestCase):
    def test_calibration_binning_and_sample_count(self):
        """Verify calibration bin error is ABS(avg_predicted_prob - actual_frequency) and sample counts are tracked."""
        # 12 predictions around 0.55 probability for an outcome
        # 9 of them actually occurred (actual_frequency = 9/12 = 0.75)
        probs = [0.55] * 12
        actuals = [1] * 9 + [0] * 3

        result = calculate_calibration_for_outcome(
            probs, actuals, num_bins=5, min_samples=10, outcome_label="Test"
        )
        
        # Bin 0.40 - 0.60 should contain all 12 items
        target_bin = [b for b in result.bins if b.bin_low == 0.40 and b.bin_high == 0.60][0]
        self.assertEqual(target_bin.sample_count, 12)
        self.assertFalse(target_bin.is_insufficient_sample)  # 12 >= 10
        self.assertAlmostEqual(target_bin.avg_predicted_prob, 0.55, places=4)
        self.assertAlmostEqual(target_bin.actual_frequency, 0.75, places=4)
        # ABS(0.55 - 0.75) = 0.20
        self.assertAlmostEqual(target_bin.calibration_error, 0.20, places=4)

    def test_small_sample_flagging(self):
        """Bins with fewer than 10 samples must be flagged as is_insufficient_sample=True."""
        probs = [0.15] * 5  # Only 5 samples in [0.0 - 0.20)
        actuals = [0] * 5
        result = calculate_calibration_for_outcome(
            probs, actuals, num_bins=5, min_samples=10, outcome_label="LowProb"
        )
        first_bin = result.bins[0]
        self.assertEqual(first_bin.sample_count, 5)
        self.assertTrue(first_bin.is_insufficient_sample)


class TestMarketBenchmarkAndFalseFavourites(unittest.TestCase):
    def test_implied_probabilities_and_tie_tolerance(self):
        """Verify normalized implied probabilities sum to 1.0 and tie tolerance marks close games as NONE."""
        # Decimal odds: 2.0 (Home), 3.5 (Draw), 4.0 (Away)
        # Inverses: 0.5, 0.2857, 0.25 -> Overround ~ 1.0357
        # Home implied prob ~ 0.4828, Away implied prob ~ 0.2414 -> Home is clear favourite
        p_h, p_d, p_a, fav = calculate_implied_probabilities(2.0, 3.5, 4.0, tie_tolerance=0.03)
        self.assertAlmostEqual(p_h + p_d + p_a, 1.0, places=3)
        self.assertEqual(fav, "H")

        # Even match: Home odds 2.60, Away odds 2.65 -> Probs ~ 0.366 and 0.359 -> diff 0.007 < 0.03
        _, _, _, fav_even = calculate_implied_probabilities(2.60, 3.20, 2.65, tie_tolerance=0.03)
        self.assertEqual(fav_even, "NONE")

    def test_false_favourite_draw_handling(self):
        """A Draw counts as market favourite failure, but only as successful detection if engine predicted Draw."""
        benchmarks = [
            {"market_favourite": "H"},  # Match 1: Home Favourite
            {"market_favourite": "H"},  # Match 2: Home Favourite
        ]
        # Match 1: Actual result was Draw. Engine predicted Away win ('A').
        # Favourite failed, but engine was WRONG about the outcome (predicted A, actual D).
        # Match 2: Actual result was Draw. Engine predicted Draw ('D').
        # Favourite failed, and engine was RIGHT!
        preds = [
            {"predicted_outcome": "A"},
            {"predicted_outcome": "D"},
        ]
        actuals = ["D", "D"]

        report = calculate_false_favourite_metrics(preds, actuals, benchmarks)
        self.assertEqual(report.identified_market_favourites, 2)
        self.assertEqual(report.market_favourite_failures, 2)
        self.assertEqual(report.engine_disagreements, 2)
        # Only Match 2 is a successful upset detection!
        self.assertEqual(report.successful_detections, 1)
        self.assertEqual(report.upset_prediction_accuracy, 0.50)


if __name__ == "__main__":
    unittest.main()
