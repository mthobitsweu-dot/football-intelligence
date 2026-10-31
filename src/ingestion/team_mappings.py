"""Team name normalisation and 3-letter abbreviation mappings.
Preserves raw names from data sources while mapping to canonical names.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

# Mapping from raw CSV HomeTeam/AwayTeam string to canonical display name
CANONICAL_TEAM_NAMES: Dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton & Hove Albion",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Liverpool": "Liverpool",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Tottenham": "Tottenham Hotspur",
    "Watford": "Watford",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}

# 3-Letter standardized team codes for deterministic match_id generation
TEAM_SLUGS: Dict[str, str] = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "AFC Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton & Hove Albion": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Leeds United": "LEE",
    "Leicester City": "LEI",
    "Liverpool": "LIV",
    "Luton Town": "LUT",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle United": "NEW",
    "Norwich City": "NOR",
    "Nottingham Forest": "NFO",
    "Sheffield United": "SHU",
    "Southampton": "SOU",
    "Tottenham Hotspur": "TOT",
    "Watford": "WAT",
    "West Ham United": "WHU",
    "Wolverhampton Wanderers": "WOL",
}


def normalise_team_name(raw_name: str) -> Tuple[str, str]:
    """Return a tuple of (canonical_name, team_slug) for any raw team name."""
    cleaned = raw_name.strip()
    canonical = CANONICAL_TEAM_NAMES.get(cleaned, cleaned)
    
    # Resolve 3-letter slug
    if canonical in TEAM_SLUGS:
        slug = TEAM_SLUGS[canonical]
    elif cleaned in TEAM_SLUGS:
        slug = TEAM_SLUGS[cleaned]
    else:
        # Fallback: remove non-alphanumeric and take first 3 letters uppercase
        alphanumeric = re.sub(r"[^A-Za-z0-9]", "", canonical).upper()
        slug = (alphanumeric[:3]).ljust(3, "X")
        
    return canonical, slug
