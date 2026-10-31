"""Configuration dataclass for Engine #4: Statistical & Form Analysis.
Contains hyperparameter definitions, priors, shrinkage parameters, and clipping bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass(frozen=True)
class StatisticalEngineConfig:
    """Hyperparameters and tuning constants for Engine #4.
    
    All constants are versioned and documented to maintain reproducibility.
    """
    config_version: str = "v1.0.0"

    # Historical match windows
    max_history_matches: int = 25                 # Max prior matches per team to estimate scoring rates
    recent_form_window: int = 5                   # Matches in short-term form window
    min_sample_threshold: int = 5                 # Below this count, status is INSUFFICIENT_DATA

    # Baseline league defaults (used if prior historical matches < 10)
    default_league_home_goals: float = 1.55
    default_league_away_goals: float = 1.25

    # Bayesian shrinkage parameters
    shrinkage_weight_m: float = 6.0               # Prior pseudo-count weight (games)
    established_team_prior_attack: float = 1.00   # League average
    established_team_prior_defense: float = 1.00  # League average
    promoted_team_prior_attack: float = 0.85      # Below league average attack
    promoted_team_prior_defense: float = 1.20     # More porous defense than league average

    # Form modifier weights
    form_weight: float = 0.20                     # 20% influence from last-5 points
    sot_weight: float = 0.15                      # 15% influence from shots-on-target dominance

    # Expected goals clipping bounds (prevents extreme or degenerate values)
    min_expected_goals: float = 0.10
    max_expected_goals: float = 5.50

    # Score matrix calculation bounds
    min_score_matrix_max: int = 12                # Evaluate at least up to 12x12 scores
    score_matrix_sigma_factor: float = 6.0        # Upper bound: lambda + 6 * sqrt(lambda)

    # Promoted teams list for 2021-2024 to apply promoted priors during cold starts
    known_promoted_teams: Set[str] = field(
        default_factory=lambda: {
            # 2021/22 Promoted
            "Brentford", "Watford", "Norwich City",
            # 2022/23 Promoted
            "Fulham", "AFC Bournemouth", "Nottingham Forest",
            # 2023/24 Promoted
            "Burnley", "Sheffield United", "Luton Town",
        }
    )
