"""Prediction context provided to specialist engines.
Strictly encapsulates pre-match historical information.
Contains zero post-match scores, same-day match outcomes, or market odds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.backtesting.time_machine import TimeMachine


@dataclass
class TeamHistory:
    """Historical completed matches for a given team prior to the target match date."""
    team_name: str
    matches: List[Dict[str, Any]]
    sample_size: int
    has_sufficient_data: bool
    min_required_samples: int = 5


@dataclass
class HeadToHeadHistory:
    """Historical direct encounters between the two teams prior to the target match date."""
    home_team: str
    away_team: str
    matches: List[Dict[str, Any]]
    sample_size: int
    has_sufficient_data: bool
    min_required_samples: int = 3


@dataclass
class MatchPredictionContext:
    """Pre-kickoff match context provided to prediction engines.
    
    Guarantees that no match played on or after match_date can be accessed.
    Excludes the target match's own results and all market odds.
    """
    match_id: str
    season: str
    division: str
    match_date: str                                   # ISO "YYYY-MM-DD"
    match_time: Optional[str]                         # ISO "HH:MM:SS" or None
    has_reliable_time: int
    home_team: str                                    # Canonical name
    away_team: str                                    # Canonical name
    raw_home_team: Optional[str] = None
    raw_away_team: Optional[str] = None
    _time_machine: Optional[TimeMachine] = field(default=None, repr=False)

    def get_team_history(
        self,
        team_name: str,
        limit: int = 10,
        venue: Optional[str] = None,
        min_required: int = 5,
    ) -> TeamHistory:
        """Fetch completed historical matches for team_name strictly prior to this match_date.
        
        Args:
            team_name: Canonical team name.
            limit: Maximum number of recent matches to return.
            venue: 'HOME', 'AWAY', or None (any venue).
            min_required: Sample size threshold to declare has_sufficient_data.
        """
        if self._time_machine is None:
            return TeamHistory(
                team_name=team_name,
                matches=[],
                sample_size=0,
                has_sufficient_data=False,
                min_required_samples=min_required,
            )

        matches = self._time_machine.query_team_history(
            team_name=team_name,
            before_date=self.match_date,
            limit=limit,
            venue=venue,
        )
        sample_size = len(matches)
        return TeamHistory(
            team_name=team_name,
            matches=matches,
            sample_size=sample_size,
            has_sufficient_data=(sample_size >= min_required),
            min_required_samples=min_required,
        )

    def get_head_to_head(
        self,
        limit: int = 5,
        min_required: int = 2,
    ) -> HeadToHeadHistory:
        """Fetch historical head-to-head matches between home_team and away_team strictly prior to match_date."""
        if self._time_machine is None:
            return HeadToHeadHistory(
                home_team=self.home_team,
                away_team=self.away_team,
                matches=[],
                sample_size=0,
                has_sufficient_data=False,
                min_required_samples=min_required,
            )

        matches = self._time_machine.query_head_to_head(
            team_a=self.home_team,
            team_b=self.away_team,
            before_date=self.match_date,
            limit=limit,
        )
        sample_size = len(matches)
        return HeadToHeadHistory(
            home_team=self.home_team,
            away_team=self.away_team,
            matches=matches,
            sample_size=sample_size,
            has_sufficient_data=(sample_size >= min_required),
            min_required_samples=min_required,
        )

    def get_league_table(self) -> List[Dict[str, Any]]:
        """Construct current league table for this season using ONLY matches completed prior to match_date."""
        if self._time_machine is None:
            return []
        return self._time_machine.query_league_table(
            season=self.season,
            division=self.division,
            before_date=self.match_date,
        )
