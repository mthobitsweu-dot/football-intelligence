"""SQLite database connection manager for Football Intelligence.
Uses Python's built-in sqlite3 module with foreign keys and row factory enabled.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

# Project root path (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "football_intelligence.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_db_path(custom_path: str | Path | None = None) -> Path | str:
    """Resolve the SQLite database path."""
    if custom_path == ":memory:":
        return ":memory:"
    if custom_path is not None:
        return Path(custom_path)
    return DEFAULT_DB_PATH


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Create and return a configured SQLite connection.
    
    Enforces foreign key constraints and enables dictionary-like column access.
    """
    path = get_db_path(db_path)
    
    # Ensure parent folder exists if writing to disk
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
    else:
        conn = sqlite3.connect(":memory:")

    # Crucial SQLite settings
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def get_db_context(db_path: str | Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database operations with automatic commit/rollback and cleanup."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Initialize database tables, constraints, and views from schema.sql."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found at: {SCHEMA_PATH}")

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_db_context(db_path) as conn:
        conn.executescript(schema_sql)
