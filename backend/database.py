"""SQLite database connectivity and initialization for FreshCheck AI prediction history."""

import logging
from pathlib import Path
import sqlite3
from typing import Generator

from backend.config import settings

logger = logging.getLogger("FreshCheckAI.Database")


def get_db_connection(db_path: Path = settings.DB_PATH) -> sqlite3.Connection:
    """Create and return an SQLite database connection with row factory configured.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection: Connection object with sqlite3.Row row factory.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")  # High concurrency WAL mode
    conn.execute("PRAGMA foreign_keys = ON;")
    # Ensure tables exist
    _ensure_tables_created(conn)
    return conn


def _ensure_tables_created(conn: sqlite3.Connection) -> None:
    """Ensure core schema exists on connection."""
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                temperature_celsius REAL,
                storage_condition TEXT,
                storage_duration TEXT,
                message TEXT
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
            ON prediction_history (timestamp DESC);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_category 
            ON prediction_history (category);
        """)


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency yielding a managed database connection."""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path = settings.DB_PATH) -> None:
    """Initialize the SQLite database schema if not already created.

    Creates the 'prediction_history' table to store past predictions per Phase 5.
    Explicitly does NOT store uploaded personal images permanently.
    """
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    temperature_celsius REAL,
                    storage_condition TEXT,
                    storage_duration TEXT,
                    message TEXT
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                ON prediction_history (timestamp DESC);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_category 
                ON prediction_history (category);
            """)
        logger.info(f"Database initialized successfully at: {db_path}")
    except Exception as exc:
        logger.error(f"Failed to initialize database at {db_path}: {exc}")
        raise
    finally:
        conn.close()


def check_db_health(db_path: Path = settings.DB_PATH) -> bool:
    """Verify database read/write availability."""
    try:
        conn = get_db_connection(db_path)
        with conn:
            conn.execute("SELECT 1;")
        conn.close()
        return True
    except Exception as exc:
        logger.warning(f"Database health check failed for {db_path}: {exc}")
        return False
