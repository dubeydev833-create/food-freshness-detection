"""History service for persisting and querying food freshness predictions in SQLite (Phase 5)."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import uuid

from backend.schemas.history import HistoryRecord

logger = logging.getLogger("FreshCheckAI.HistoryService")


def save_prediction(
    conn: sqlite3.Connection,
    category: str,
    predicted_class: str,
    confidence: float,
    temperature_celsius: Optional[float] = None,
    storage_condition: Optional[str] = None,
    storage_duration: Optional[str] = None,
    message: Optional[str] = None,
) -> HistoryRecord:
    """Save a prediction result into SQLite prediction_history.

    PRIVACY & SCIENTIFIC INTEGRITY:
        Only stores categorical labels, confidence, and environmental metadata.
        DOES NOT store uploaded personal images permanently.

    Args:
        conn: Active SQLite connection.
        category: Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy').
        predicted_class: Predicted class ('Fresh', 'Moderate', 'Spoiled').
        confidence: Prediction confidence score [0.0 - 1.0].
        temperature_celsius: Optional observed temperature.
        storage_condition: Optional storage condition.
        storage_duration: Optional storage duration text.
        message: Optional status or prediction message.

    Returns:
        HistoryRecord: The newly persisted prediction record.
    """
    prediction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO prediction_history (
            prediction_id,
            timestamp,
            category,
            predicted_class,
            confidence,
            temperature_celsius,
            storage_condition,
            storage_duration,
            message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            prediction_id,
            timestamp,
            category,
            predicted_class,
            confidence,
            temperature_celsius,
            storage_condition,
            storage_duration,
            message,
        ),
    )
    conn.commit()
    record_id = cursor.lastrowid or 0

    return HistoryRecord(
        id=record_id,
        prediction_id=prediction_id,
        timestamp=timestamp,
        category=category,
        predicted_class=predicted_class,
        confidence=confidence,
        temperature_celsius=temperature_celsius,
        storage_condition=storage_condition,
        storage_duration=storage_duration,
        message=message,
    )


def get_prediction_history(
    conn: sqlite3.Connection,
    limit: int = 50,
    offset: int = 0,
    category: Optional[str] = None,
) -> Tuple[List[HistoryRecord], int]:
    """Retrieve historical prediction records with pagination and optional category filtering.

    Args:
        conn: Active SQLite connection.
        limit: Maximum records to return.
        offset: Pagination offset.
        category: Optional food category filter.

    Returns:
        Tuple[List[HistoryRecord], int]: (records, total_matching_count).
    """
    cursor = conn.cursor()

    if category:
        clean_cat = category.strip().capitalize()
        count_query = "SELECT COUNT(*) FROM prediction_history WHERE LOWER(category) = LOWER(?);"
        cursor.execute(count_query, (clean_cat,))
        total_count = cursor.fetchone()[0]

        select_query = """
            SELECT id, prediction_id, timestamp, category, predicted_class,
                   confidence, temperature_celsius, storage_condition, storage_duration, message
            FROM prediction_history
            WHERE LOWER(category) = LOWER(?)
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?;
        """
        cursor.execute(select_query, (clean_cat, limit, offset))
    else:
        count_query = "SELECT COUNT(*) FROM prediction_history;"
        cursor.execute(count_query)
        total_count = cursor.fetchone()[0]

        select_query = """
            SELECT id, prediction_id, timestamp, category, predicted_class,
                   confidence, temperature_celsius, storage_condition, storage_duration, message
            FROM prediction_history
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?;
        """
        cursor.execute(select_query, (limit, offset))

    rows = cursor.fetchall()
    items = [
        HistoryRecord(
            id=row["id"],
            prediction_id=row["prediction_id"],
            timestamp=row["timestamp"],
            category=row["category"],
            predicted_class=row["predicted_class"],
            confidence=row["confidence"],
            temperature_celsius=row["temperature_celsius"],
            storage_condition=row["storage_condition"],
            storage_duration=row["storage_duration"],
            message=row["message"],
        )
        for row in rows
    ]
    return items, total_count


def clear_prediction_history(conn: sqlite3.Connection) -> int:
    """Clear all records from prediction_history table.

    Args:
        conn: Active SQLite connection.

    Returns:
        int: Count of deleted records.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prediction_history;")
    count_before = cursor.fetchone()[0]

    cursor.execute("DELETE FROM prediction_history;")
    conn.commit()
    logger.info(f"Cleared {count_before} records from prediction_history.")
    return count_before
