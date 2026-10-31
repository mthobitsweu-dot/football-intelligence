"""Configuration dataclass for Engine #1: Motivation & Psychology.
Defines provisional coefficients, threshold distances, and season constants.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotivationEngineConfig:
    """Hyperparameters and provisional design coefficients for Engine #1.
    
    All coefficients are explicitly disclosed as provisional design assumptions,
    not empirically fitted or tuned against evaluation data.
    """
    config_version: str = "v1.0.0"

    # League structure constants (20-team Premier League season)
    total_season_matches: int = 380
    matches_per_team: int = 38
    teams_in_league: int = 20
    relegation_cutoff_rank: int = 18              # Positions 18, 19, 20 are relegated
    top4_cutoff_rank: int = 4                     # Positions 1-4 qualify for Champions League

    # Sample size threshold
    min_sample_threshold: int = 5                 # Matches played before status is considered 'OK'

    # Provisional Multinomial Logistic Regression Intercepts (Baseline Anchor)
    # Calibrated to 45% Home Win, 25% Draw, 30% Away Win when all situational inputs are 0
    beta_0_draw: float = 0.0000                   # Standard reference category
    beta_0_home: float = 0.5878                   # ln(0.45 / 0.25) = ln(1.80)
    beta_0_away: float = 0.1823                   # ln(0.30 / 0.25) = ln(1.20)

    # Provisional Situational Slopes (Design Assumptions)
    beta_urgency: float = 0.3000                  # Max urgency differential (+-1.0) shifts win prob by ~6-8%
    beta_sched: float = 0.0150                    # Weak scheduling proxy slope
    beta_derby: float = 0.0000                    # Strictly neutral in v1.0.0 (no probability distortion)

    # Scheduling interval bounds
    max_interval_diff_days: int = 7               # clip(days_home - days_away, -7, 7)

    # Contextual urgency points window
    urgency_points_window: float = 6.0            # Teams within 6 points of cutoff in run-in have active urgency
    title_start_progress: float = 0.50            # Title urgency activates after 50% of season
    europe_start_progress: float = 0.50           # European qualification urgency activates after 50%
    relegation_start_progress: float = 0.25       # Relegation hazard urgency activates after 25%
