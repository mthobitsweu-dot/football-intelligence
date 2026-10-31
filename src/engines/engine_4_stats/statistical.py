"""Engine #4: Statistical & Form Analysis.
Calculates expected goals (lambda, mu) using team scoring rates, Bayesian shrinkage,
recent 5-match form, and shots-on-target dominance, generating Poisson outcome probabilities.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from src.backtesting.context import MatchPredictionContext
from src.engines.base import BaseEngine, EnginePrediction
from src.engines.engine_4_stats.config import StatisticalEngineConfig

ENGINE_ID = "engine_4_stats"
MODEL_VERSION = "v1.0.0"


def compute_poisson_probability(lmbda: float, k: int) -> float:
    """Compute Poisson probability P(X = k; lambda) using standard library math."""
    if lmbda <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)


def compute_adaptive_score_matrix(
    lambda_h: float,
    mu_a: float,
    min_max_score: int = 12,
    sigma_factor: float = 6.0,
    max_cap: int = 20,
) -> Tuple[float, float, float, float]:
    """Compute Home Win, Draw, and Away Win probabilities using an adaptive Poisson score matrix.
    
    Returns (prob_home, prob_draw, prob_away, omitted_probability_mass).
    Uses an adaptive upper bound: max(min_max_score, ceil(lambda + sigma_factor * sqrt(lambda))).
    Verifies that the unnormalized omitted tail mass is negligible before normalization.
    """
    # Safe clipping of expected goals
    safe_lambda = max(0.05, min(lambda_h, 6.0))
    safe_mu = max(0.05, min(mu_a, 6.0))

    # Adaptive upper bound based on standard deviations of Poisson
    bound_h = int(math.ceil(safe_lambda + sigma_factor * math.sqrt(safe_lambda)))
    bound_a = int(math.ceil(safe_mu + sigma_factor * math.sqrt(safe_mu)))
    max_score = min(max_cap, max(min_max_score, bound_h, bound_a))

    # Pre-calculate marginal Poisson distributions up to max_score
    p_h = [compute_poisson_probability(safe_lambda, x) for x in range(max_score + 1)]
    p_a = [compute_poisson_probability(safe_mu, y) for y in range(max_score + 1)]

    # Compute raw outcome sums
    raw_home_win = 0.0
    raw_draw = 0.0
    raw_away_win = 0.0

    for x in range(max_score + 1):
        prob_x = p_h[x]
        for y in range(max_score + 1):
            prob_xy = prob_x * p_a[y]
            if x > y:
                raw_home_win += prob_xy
            elif x == y:
                raw_draw += prob_xy
            else:
                raw_away_win += prob_xy

    total_captured_mass = raw_home_win + raw_draw + raw_away_win
    omitted_mass = max(0.0, 1.0 - total_captured_mass)

    # Normalize captured mass so probabilities sum exactly to 1.0
    if total_captured_mass > 0:
        norm_home = raw_home_win / total_captured_mass
        norm_draw = raw_draw / total_captured_mass
        norm_away = raw_away_win / total_captured_mass
    else:
        norm_home, norm_draw, norm_away = 1 / 3, 1 / 3, 1 / 3

    # Ensure numerical sum check
    residual = 1.0 - (norm_home + norm_draw + norm_away)
    norm_draw += residual

    return norm_home, norm_draw, norm_away, omitted_mass


class Engine4Statistical(BaseEngine):
    """Engine #4: Statistical & Form Analysis Engine.
    
    Generates transparent, non-overfitting predictions using:
      1. Historical scoring/conceding rates (Home/Away splits).
      2. Bayesian shrinkage to establish conservative ratings for low-sample teams.
      3. Recent 5-match form and shots-on-target dominance.
      4. Bivariate Poisson goal expectancy matrix.
    """
    engine_id: str = ENGINE_ID
    model_version: str = MODEL_VERSION

    def __init__(self, config: Optional[StatisticalEngineConfig] = None):
        self.config = config or StatisticalEngineConfig()

    def _estimate_league_baselines(self, context: MatchPredictionContext) -> Tuple[float, float]:
        """Estimate league home and away goal scoring averages strictly before target match date."""
        table = context.get_league_table()
        total_matches_played = sum(t.get("played", 0) for t in table) // 2
        total_goals_scored = sum(t.get("goals_for", 0) for t in table)

        if total_matches_played >= 10 and total_goals_scored > 0:
            avg_goals_per_game = total_goals_scored / total_matches_played
            # Typical English Premier League home split is ~55% home, 45% away
            league_home = round(avg_goals_per_game * 0.55, 3)
            league_away = round(avg_goals_per_game * 0.45, 3)
            return max(1.0, league_home), max(0.8, league_away)

        return self.config.default_league_home_goals, self.config.default_league_away_goals

    def _calculate_team_strength(
        self,
        team_name: str,
        matches: List[Dict[str, Any]],
        is_home_perspective: bool,
        league_home_avg: float,
        league_away_avg: float,
    ) -> Tuple[float, float, int]:
        """Calculate attacking strength and defensive weakness with Bayesian shrinkage.
        
        Returns (attack_strength, defense_weakness, sample_size).
        """
        k = len(matches)
        # Determine prior based on whether team was recently promoted
        is_promoted = team_name in self.config.known_promoted_teams
        prior_att = (
            self.config.promoted_team_prior_attack
            if is_promoted
            else self.config.established_team_prior_attack
        )
        prior_def = (
            self.config.promoted_team_prior_defense
            if is_promoted
            else self.config.established_team_prior_defense
        )

        if k == 0:
            return prior_att, prior_def, 0

        # Calculate goals scored and conceded in relevant context
        goals_scored = 0
        goals_conceded = 0
        for m in matches:
            if m["home_team"] == team_name:
                goals_scored += m.get("full_time_home_goals", 0) or 0
                goals_conceded += m.get("full_time_away_goals", 0) or 0
            else:
                goals_scored += m.get("full_time_away_goals", 0) or 0
                goals_conceded += m.get("full_time_home_goals", 0) or 0

        avg_scored = goals_scored / k
        avg_conceded = goals_conceded / k

        # Compare against the appropriate baseline
        if is_home_perspective:
            # Home team's attack is compared to league home scoring; defense compared to league away scoring
            raw_att = avg_scored / league_home_avg if league_home_avg > 0 else 1.0
            raw_def = avg_conceded / league_away_avg if league_away_avg > 0 else 1.0
        else:
            # Away team's attack is compared to league away scoring; defense compared to league home scoring
            raw_att = avg_scored / league_away_avg if league_away_avg > 0 else 1.0
            raw_def = avg_conceded / league_home_avg if league_home_avg > 0 else 1.0

        # Bayesian shrinkage: weight = k / (k + m)
        m_weight = self.config.shrinkage_weight_m
        shrunk_att = (k / (k + m_weight)) * raw_att + (m_weight / (k + m_weight)) * prior_att
        shrunk_def = (k / (k + m_weight)) * raw_def + (m_weight / (k + m_weight)) * prior_def

        # Bound strengths to avoid extreme skew
        bounded_att = max(0.40, min(shrunk_att, 2.50))
        bounded_def = max(0.40, min(shrunk_def, 2.50))

        return round(bounded_att, 4), round(bounded_def, 4), k

    def _calculate_form_modifier(
        self, team_name: str, recent_matches: List[Dict[str, Any]]
    ) -> float:
        """Calculate form modifier from the last up-to-5 matches based on points and goal differential."""
        k = len(recent_matches)
        if k == 0:
            return 1.00

        points = 0
        gd = 0
        for m in recent_matches:
            is_home = (m["home_team"] == team_name)
            hg = m.get("full_time_home_goals", 0) or 0
            ag = m.get("full_time_away_goals", 0) or 0
            res = m.get("full_time_result")

            if (is_home and res == "H") or (not is_home and res == "A"):
                points += 3
            elif res == "D":
                points += 1

            if is_home:
                gd += (hg - ag)
            else:
                gd += (ag - hg)

        # Points rate: pts / (3 * k) in [0.0, 1.0]
        pts_rate = points / (3 * k)
        # Baseline expectation is ~0.40 (1.2 points/match)
        diff = pts_rate - 0.40
        modifier = 1.0 + self.config.form_weight * diff
        return round(max(0.80, min(modifier, 1.25)), 4)

    def _calculate_sot_modifier(
        self, team_name: str, recent_matches: List[Dict[str, Any]]
    ) -> float:
        """Calculate shots-on-target dominance modifier over recent matches."""
        sot_for = 0
        sot_against = 0
        valid_matches = 0

        for m in recent_matches:
            is_home = (m["home_team"] == team_name)
            h_sot = m.get("home_shots_on_target")
            a_sot = m.get("away_shots_on_target")

            # Missing stats remain missing (None)
            if h_sot is None or a_sot is None:
                continue

            valid_matches += 1
            if is_home:
                sot_for += h_sot
                sot_against += a_sot
            else:
                sot_for += a_sot
                sot_against += h_sot

        if valid_matches == 0 or (sot_for + sot_against) == 0:
            return 1.00

        sotr = sot_for / (sot_for + sot_against)
        # Centered at 0.50 (equal shots on target)
        modifier = 1.0 + self.config.sot_weight * (sotr - 0.50)
        return round(max(0.85, min(modifier, 1.15)), 4)

    def predict(self, context: MatchPredictionContext) -> EnginePrediction:
        """Generate a statistical match forecast using pre-match data."""
        # 1. League Baselines
        base_h, base_a = self._estimate_league_baselines(context)

        # 2. Historical Matches
        home_hist = context.get_team_history(
            context.home_team, limit=self.config.max_history_matches
        )
        away_hist = context.get_team_history(
            context.away_team, limit=self.config.max_history_matches
        )

        # 3. Attacking and Defensive Strengths
        att_h, def_h, k_h = self._calculate_team_strength(
            context.home_team, home_hist.matches, is_home_perspective=True,
            league_home_avg=base_h, league_away_avg=base_a,
        )
        att_a, def_a, k_a = self._calculate_team_strength(
            context.away_team, away_hist.matches, is_home_perspective=False,
            league_home_avg=base_h, league_away_avg=base_a,
        )

        # 4. Short-term Form & SOT Modifiers (Last 5 matches)
        recent_h = home_hist.matches[:self.config.recent_form_window]
        recent_a = away_hist.matches[:self.config.recent_form_window]
        form_mod_h = self._calculate_form_modifier(context.home_team, recent_h)
        form_mod_a = self._calculate_form_modifier(context.away_team, recent_a)
        sot_mod_h = self._calculate_sot_modifier(context.home_team, recent_h)
        sot_mod_a = self._calculate_sot_modifier(context.away_team, recent_a)

        # 5. Expected Goals (lambda, mu)
        raw_lambda = base_h * att_h * def_a * form_mod_h * sot_mod_h
        raw_mu = base_a * att_a * def_h * form_mod_a * sot_mod_a

        lambda_h = round(max(self.config.min_expected_goals, min(raw_lambda, self.config.max_expected_goals)), 4)
        mu_a = round(max(self.config.min_expected_goals, min(raw_mu, self.config.max_expected_goals)), 4)

        # 6. Bivariate Poisson Matrix Outcome Probabilities
        prob_h, prob_d, prob_a, omitted_mass = compute_adaptive_score_matrix(
            lambda_h, mu_a,
            min_max_score=self.config.min_score_matrix_max,
            sigma_factor=self.config.score_matrix_sigma_factor,
        )

        # 7. Predicted Outcome & Confidence
        if prob_h >= prob_d and prob_h >= prob_a:
            outcome = "H"
        elif prob_a >= prob_d and prob_a >= prob_h:
            outcome = "A"
        else:
            outcome = "D"

        # Confidence strictly defined as max probability
        confidence = max(prob_h, prob_d, prob_a)

        # 8. Prediction Status & Uncertainty Communication
        has_sufficient_h = (k_h >= self.config.min_sample_threshold)
        has_sufficient_a = (k_a >= self.config.min_sample_threshold)
        if not has_sufficient_h or not has_sufficient_a:
            status = "INSUFFICIENT_DATA"
        else:
            status = "OK"

        # 9. Informative Supporting Evidence String
        evidence = (
            f"StatModel (xG: {lambda_h:.2f} - {mu_a:.2f}) | "
            f"{context.home_team} [Att: {att_h:.2f}, Def: {def_h:.2f}, Form: x{form_mod_h:.2f}, Samples: {k_h}] vs "
            f"{context.away_team} [Att: {att_a:.2f}, Def: {def_a:.2f}, Form: x{form_mod_a:.2f}, Samples: {k_a}] | "
            f"Probs: H={prob_h*100:.1f}%, D={prob_d*100:.1f}%, A={prob_a*100:.1f}% | "
            f"Status: {status} (Tail Mass: {omitted_mass:.6f})"
        )

        return EnginePrediction(
            engine_id=self.engine_id,
            model_version=self.model_version,
            predicted_outcome=outcome,
            home_win_prob=prob_h,
            draw_prob=prob_d,
            away_win_prob=prob_a,
            confidence=confidence,
            prediction_status=status,
            home_sample_size=k_h,
            away_sample_size=k_a,
            predicted_home_goals=lambda_h,
            predicted_away_goals=mu_a,
            supporting_evidence=evidence,
        )
