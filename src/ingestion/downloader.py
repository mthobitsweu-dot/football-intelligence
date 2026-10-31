"""Historical football data downloader.
Downloads official CSV match files from Football-Data.co.uk for personal research.
Uses Python's standard library urllib.request.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

SEASONS: Dict[str, str] = {
    "2021-2022": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "2022-2023": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023-2024": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def get_raw_filepath(season: str) -> Path:
    """Format local destination filename from season string, e.g. '2023-2024' -> 'E0_2023_2024.csv'."""
    clean_season = season.replace("-", "_")
    return RAW_DATA_DIR / f"E0_{clean_season}.csv"


def download_historical_season(season: str, force: bool = False) -> Path:
    """Download a single season CSV file if not already present."""
    if season not in SEASONS:
        raise ValueError(f"Unknown season '{season}'. Available: {list(SEASONS.keys())}")

    dest_path = get_raw_filepath(season)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        logger.info("File already exists locally: %s", dest_path)
        return dest_path

    url = SEASONS[season]
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FootballIntelligence/1.0"},
    )
    logger.info("Downloading %s from %s ...", season, url)
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"Download failed with HTTP status {response.status} from {url}")
        content = response.read()

    # Save exact raw bytes to disk
    dest_path.write_bytes(content)
    logger.info("Saved %d bytes to %s", len(content), dest_path)
    return dest_path


def download_all_seasons(force: bool = False) -> List[Path]:
    """Download all three target Premier League seasons."""
    downloaded: List[Path] = []
    for season in SEASONS:
        downloaded.append(download_historical_season(season, force=force))
    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    files = download_all_seasons()
    print(f"Downloaded/Verified {len(files)} season files:")
    for f in files:
        print(f" - {f.name} ({f.stat().st_size:,} bytes)")
