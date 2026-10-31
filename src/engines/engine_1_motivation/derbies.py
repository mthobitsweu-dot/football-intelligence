"""Documented registry of traditional English Premier League rivalries and derbies.
In v1.0.0, derbies are flagged for auditing and evidence reporting only.
Their mathematical probability effect is strictly neutral (beta_derby = 0.0)
until historical predictive value is empirically demonstrated.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional

# Documented traditional Premier League rivalry pairs (canonical names)
TRADITIONAL_DERBY_REGISTRY: Dict[FrozenSet[str], str] = {
    frozenset({"Arsenal", "Tottenham Hotspur"}): "North London Derby",
    frozenset({"Liverpool", "Everton"}): "Merseyside Derby",
    frozenset({"Manchester City", "Manchester United"}): "Manchester Derby",
    frozenset({"Liverpool", "Manchester United"}): "Northwest Derby",
    frozenset({"Chelsea", "Fulham"}): "West London Derby",
    frozenset({"Chelsea", "Brentford"}): "West London Derby",
    frozenset({"Aston Villa", "Wolverhampton Wanderers"}): "Midlands Derby",
    frozenset({"Crystal Palace", "Brighton & Hove Albion"}): "M23 Derby",
}


def identify_derby(home_team: str, away_team: str) -> Optional[str]:
    """Identify if a matchup is an established traditional rivalry.
    
    Returns the derby name (e.g. 'North London Derby') or None.
    """
    pair = frozenset({home_team.strip(), away_team.strip()})
    return TRADITIONAL_DERBY_REGISTRY.get(pair)
