"""Base contract and data structures for all prediction engines.
All six prediction engines (Engine #0 through #5) adhere to this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from src.backtesting.context import MatchPredictionContext


@dataclass
class EnginePrediction:
    """Standardized prediction output produced by any of the 6 engines."""
    engine_id: str
    model_version: str
    predicted_outcome: str                                    # 'H', 'D', or 'A'
    home_win_prob: float                                      # Between 0.0 and 1.0
    draw_prob: float                                          # Between 0.0 and 1.0
    away_win_prob: float                                      # Between 0.0 and 1.0
    confidence: float                                         # Strictly max(home, draw, away)
    prediction_status: str = "OK"                             # 'OK', 'INSUFFICIENT_DATA', 'DEGRADED', 'ERROR'
    home_sample_size: int = 0
    away_sample_size: int = 0
    predicted_home_goals: Optional[float] = None
    predicted_away_goals: Optional[float] = None
    supporting_evidence: Optional[str] = None

    def __post_init__(self):
        # Strict numerical tolerance: ABS(home + draw + away - 1.0) < 0.000001
        prob_sum = self.home_win_prob + self.draw_prob + self.away_win_prob
        if abs(prob_sum - 1.0) >= 0.000001:
            raise ValueError(
                f"Probabilities must sum to 1.0 within 0.000001 tolerance. Sum was: {prob_sum:.8f}"
            )
        # Confidence must be the maximum probability
        expected_conf = max(self.home_win_prob, self.draw_prob, self.away_win_prob)
        if abs(self.confidence - expected_conf) >= 0.000001:
            self.confidence = expected_conf


@runtime_checkable
class BaseEngine(Protocol):
    """Protocol that all 6 independent prediction engines implement."""
    engine_id: str
    model_version: str

    def predict(self, context: MatchPredictionContext) -> EnginePrediction:
        """Produce a prediction using only pre-match context."""
        ...
