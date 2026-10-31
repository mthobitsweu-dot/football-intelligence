"""Engine #4: Statistical & Form Analysis.
Calculates rolling form, expected goals (Poisson), and home/away splits.
First working specialist prediction engine.
"""
from src.engines.engine_4_stats.config import StatisticalEngineConfig
from src.engines.engine_4_stats.statistical import (
    ENGINE_ID,
    MODEL_VERSION,
    Engine4Statistical,
    compute_adaptive_score_matrix,
    compute_poisson_probability,
)

__all__ = [
    "ENGINE_ID",
    "MODEL_VERSION",
    "Engine4Statistical",
    "StatisticalEngineConfig",
    "compute_adaptive_score_matrix",
    "compute_poisson_probability",
]
