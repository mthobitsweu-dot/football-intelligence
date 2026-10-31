# Football Intelligence

A free, mobile-friendly, multi-engine football prediction and analysis platform.

## Prediction Architecture

- **Engine #0: Master Consensus Engine** – Evaluates specialist predictions, disagreements, and evidence to produce the final forecast.
- **Engine #1: Motivation & Psychology** – Analyzes team stakes, manager pressure, title/relegation races, and derby intensity.
- **Engine #2: Home Advantage & Environment** – Analyzes stadium factors, travel fatigue, altitude, weather, and crowd impact.
- **Engine #3: Tactical & Matchup Analysis** – Analyzes playing formations, pressing systems, counter-attacking dynamics, and style clashes.
- **Engine #4: Statistical & Form Analysis** – Analyzes rolling weighted form, goal expectation (Poisson), and home/away goal differentials.
- **Engine #5: Squad Quality & Reliability** – Analyzes squad depth, key player absences, disciplinary suspensions, and roster consistency.

## Project Goals

- Improve football prediction accuracy.
- Detect false favourites using closing odds benchmarks.
- Track historical performance.
- Backtest predictions with strict temporal cutoff (zero data leakage).
- Evolve each engine independently.
- Completely free and locally executable (no paid APIs or hosting).

---

## Getting Started

### Prerequisites
- **Python 3.12+** (no external packages or Node.js required for data ingestion, database management, and backtesting).

### Running the Test Suite
To run all 31 unit and integration tests (database constraints, foreign keys, team normalisation, strict calendar-date isolation, multiclass Brier score, 3-way calibration, and false-favourite tracking):

```bash
python -m unittest discover tests -v
```

### Initializing the SQLite Database
To initialize tables, views, and constraints:

```bash
python -c "from src.database.connection import init_db; init_db(); print('Database ready!')"
```

### Downloading Historical Data
To download the three target English Premier League seasons (2021/22, 2022/23, 2023/24) into `data/raw/`:

```bash
python src/ingestion/downloader.py
```

### Ingesting Historical Matches & Closing Odds
To parse, validate, and load all downloaded seasons into SQLite:

```bash
python -m src.ingestion.historical_loader
python -m src.ingestion.odds_loader
```

### Running Historical Backtests
To execute a reproducible historical backtest across an entire season under strict calendar-date cutoff:

```bash
python -m src.backtesting.runner --season 2023-2024
```

---

## Directory Overview

- `data/` – Local SQLite database (`football_intelligence.db`) and raw CSV match files (`data/raw/`).
- `src/database/` – Pure SQLite connection manager, schema, migrations, and CRUD repository.
- `src/ingestion/` – Historical match dataset downloader, team normalizer, and market odds parser.
- `src/engines/` – Independent directories for all 6 prediction engines and `base.py` contract.
- `src/backtesting/` – Time machine, temporal context, evaluation metrics, and CLI runner.
- `src/web/` – Lightweight, mobile-friendly interface (built after backend validation).
- `tests/` – Automated unit and integration test suite.
