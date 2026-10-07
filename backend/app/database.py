from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# DATABASE PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DB_DIR = ROOT / "data" / "app"
DB_PATH = DB_DIR / "upay_sentinel.db"


# ============================================================
# CONNECTION
# ============================================================

def get_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ============================================================
# TIME
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db() -> None:
    connection = get_connection()

    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,

                recipient_new INTEGER NOT NULL,
                device_changed INTEGER NOT NULL,
                location_changed INTEGER NOT NULL,

                transactions_last_1h INTEGER NOT NULL,
                transactions_last_24h INTEGER NOT NULL,

                avg_amount_30d REAL NOT NULL,
                usual_transaction_hour INTEGER NOT NULL,
                hour INTEGER NOT NULL,
                account_age_days INTEGER NOT NULL,

                amount_ratio REAL,
                hour_distance_from_usual REAL,
                behavioral_deviation_score REAL,

                extra_json TEXT,

                created_at TEXT NOT NULL
            );


            CREATE TABLE IF NOT EXISTS risk_assessments (
                transaction_id TEXT PRIMARY KEY,

                transaction_risk REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                final_risk_score REAL NOT NULL,

                risk_level TEXT NOT NULL,
                recommended_action TEXT NOT NULL,

                model_threshold REAL,
                model_decision TEXT,

                reasons_json TEXT,

                created_at TEXT NOT NULL,

                FOREIGN KEY (transaction_id)
                    REFERENCES transactions(transaction_id)
                    ON DELETE CASCADE
            );


            CREATE TABLE IF NOT EXISTS investigation_cases (
                case_id INTEGER PRIMARY KEY AUTOINCREMENT,

                transaction_id TEXT NOT NULL,

                risk_level TEXT,
                final_risk_score REAL,

                case_status TEXT,

                case_payload_json TEXT NOT NULL,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                FOREIGN KEY (transaction_id)
                    REFERENCES transactions(transaction_id)
                    ON DELETE CASCADE
            );


            CREATE INDEX IF NOT EXISTS idx_cases_created_at
            ON investigation_cases(created_at DESC);


            CREATE INDEX IF NOT EXISTS idx_cases_transaction_id
            ON investigation_cases(transaction_id);


            CREATE INDEX IF NOT EXISTS idx_transactions_customer_id
            ON transactions(customer_id);


            CREATE INDEX IF NOT EXISTS idx_transactions_timestamp
            ON transactions(timestamp);
            """
        )

        connection.commit()

    finally:
        connection.close()


# ============================================================
# SAVE TRANSACTION + RISK + OPTIONAL CASE
# ============================================================

def save_analysis(
    transaction: dict[str, Any],
    analysis: Any,
    case: Any | None = None,
) -> int | None:
    """
    Persist one complete Sentinel analysis.

    Stores:
      1. Transaction context
      2. Risk assessment
      3. Investigation case payload (when provided)
    """

    init_db()

    if hasattr(analysis, "model_dump"):
        analysis_data = analysis.model_dump(
            mode="json"
        )
    else:
        analysis_data = dict(
            analysis or {}
        )

    if case is not None:
        if hasattr(case, "model_dump"):
            case_data = case.model_dump(
                mode="json"
            )
        else:
            case_data = dict(
                case or {}
            )
    else:
        case_data = None

    connection = get_connection()

    try:
        now = utc_now()

        # ----------------------------------------------------
        # 1. Transaction
        # ----------------------------------------------------

        transaction_id = str(
            transaction["transaction_id"]
        )

        known_fields = {
            "transaction_id",
            "customer_id",
            "recipient_id",
            "timestamp",
            "amount",
            "recipient_new",
            "device_changed",
            "location_changed",
            "transactions_last_1h",
            "transactions_last_24h",
            "avg_amount_30d",
            "usual_transaction_hour",
            "hour",
            "account_age_days",
            "amount_ratio",
            "hour_distance_from_usual",
            "behavioral_deviation_score",
        }

        extra_data = {
            key: value
            for key, value in transaction.items()
            if key not in known_fields
        }

        connection.execute(
            """
            INSERT INTO transactions (
                transaction_id,
                customer_id,
                recipient_id,
                timestamp,
                amount,
                recipient_new,
                device_changed,
                location_changed,
                transactions_last_1h,
                transactions_last_24h,
                avg_amount_30d,
                usual_transaction_hour,
                hour,
                account_age_days,
                amount_ratio,
                hour_distance_from_usual,
                behavioral_deviation_score,
                extra_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(transaction_id)
            DO UPDATE SET
                customer_id = excluded.customer_id,
                recipient_id = excluded.recipient_id,
                timestamp = excluded.timestamp,
                amount = excluded.amount,
                recipient_new = excluded.recipient_new,
                device_changed = excluded.device_changed,
                location_changed = excluded.location_changed,
                transactions_last_1h = excluded.transactions_last_1h,
                transactions_last_24h = excluded.transactions_last_24h,
                avg_amount_30d = excluded.avg_amount_30d,
                usual_transaction_hour = excluded.usual_transaction_hour,
                hour = excluded.hour,
                account_age_days = excluded.account_age_days,
                amount_ratio = excluded.amount_ratio,
                hour_distance_from_usual = excluded.hour_distance_from_usual,
                behavioral_deviation_score = excluded.behavioral_deviation_score,
                extra_json = excluded.extra_json
            """,
            (
                transaction_id,
                str(transaction["customer_id"]),
                str(transaction["recipient_id"]),
                str(transaction["timestamp"]),
                float(transaction["amount"]),
                int(transaction["recipient_new"]),
                int(transaction["device_changed"]),
                int(transaction["location_changed"]),
                int(transaction["transactions_last_1h"]),
                int(transaction["transactions_last_24h"]),
                float(transaction["avg_amount_30d"]),
                int(transaction["usual_transaction_hour"]),
                int(transaction["hour"]),
                int(transaction["account_age_days"]),
                float(
                    transaction.get(
                        "amount_ratio",
                        0.0,
                    )
                ),
                float(
                    transaction.get(
                        "hour_distance_from_usual",
                        0.0,
                    )
                ),
                float(
                    transaction.get(
                        "behavioral_deviation_score",
                        0.0,
                    )
                ),
                json.dumps(
                    extra_data,
                    ensure_ascii=False,
                ),
                now,
            ),
        )

        # ----------------------------------------------------
        # 2. Risk assessment
        # ----------------------------------------------------

        reasons = analysis_data.get(
            "reasons",
            [],
        )

        connection.execute(
            """
            INSERT INTO risk_assessments (
                transaction_id,
                transaction_risk,
                anomaly_score,
                final_risk_score,
                risk_level,
                recommended_action,
                model_threshold,
                model_decision,
                reasons_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(transaction_id)
            DO UPDATE SET
                transaction_risk = excluded.transaction_risk,
                anomaly_score = excluded.anomaly_score,
                final_risk_score = excluded.final_risk_score,
                risk_level = excluded.risk_level,
                recommended_action = excluded.recommended_action,
                model_threshold = excluded.model_threshold,
                model_decision = excluded.model_decision,
                reasons_json = excluded.reasons_json,
                created_at = excluded.created_at
            """,
            (
                transaction_id,
                float(
                    analysis_data.get(
                        "transaction_risk",
                        0.0,
                    )
                ),
                float(
                    analysis_data.get(
                        "anomaly_score",
                        0.0,
                    )
                ),
                float(
                    analysis_data.get(
                        "final_risk_score",
                        0.0,
                    )
                ),
                str(
                    analysis_data.get(
                        "risk_level",
                        "",
                    )
                ),
                str(
                    analysis_data.get(
                        "recommended_action",
                        "",
                    )
                ),
                (
                    float(
                        analysis_data["model_threshold"]
                    )
                    if analysis_data.get(
                        "model_threshold"
                    )
                    is not None
                    else None
                ),
                analysis_data.get(
                    "model_decision"
                ),
                json.dumps(
                    reasons,
                    ensure_ascii=False,
                ),
                now,
            ),
        )

        # ----------------------------------------------------
        # 3. Investigation case
        # ----------------------------------------------------

        case_id = None

        if case_data is not None:

            case_status = (
                case_data.get("case_status")
                or case_data.get("status")
            )

            risk_level = (
                case_data.get("risk_level")
                or analysis_data.get(
                    "risk_level"
                )
            )

            final_risk_score = (
                case_data.get(
                    "final_risk_score"
                )
                or analysis_data.get(
                    "final_risk_score"
                )
            )

            # Create a new persistent case.
            cursor = connection.execute(
                """
                INSERT INTO investigation_cases (
                    transaction_id,
                    risk_level,
                    final_risk_score,
                    case_status,
                    case_payload_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    risk_level,
                    (
                        float(final_risk_score)
                        if final_risk_score is not None
                        else None
                    ),
                    case_status,
                    json.dumps(
                        case_data,
                        ensure_ascii=False,
                    ),
                    now,
                    now,
                ),
            )

            case_id = int(
                cursor.lastrowid
            )

        connection.commit()

        return case_id

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ============================================================
# RECENT CASES
# ============================================================

def get_recent_cases(
    limit: int = 20,
) -> list[dict[str, Any]]:

    limit = max(
        1,
        min(
            int(limit),
            100,
        ),
    )

    init_db()

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                c.case_id,
                c.transaction_id,
                c.risk_level,
                c.final_risk_score,
                c.case_status,
                c.created_at,
                c.updated_at
            FROM investigation_cases c
            ORDER BY c.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        connection.close()


# ============================================================
# GET ONE CASE
# ============================================================

def get_case(
    case_id: int,
) -> dict[str, Any] | None:

    init_db()

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                case_id,
                transaction_id,
                risk_level,
                final_risk_score,
                case_status,
                case_payload_json,
                created_at,
                updated_at
            FROM investigation_cases
            WHERE case_id = ?
            """,
            (int(case_id),),
        ).fetchone()

        if row is None:
            return None

        result = dict(row)

        payload = json.loads(
            result.pop(
                "case_payload_json"
            )
        )

        result["case"] = payload

        return result

    finally:
        connection.close()


# ============================================================
# DATABASE HEALTH
# ============================================================

def database_health() -> dict[str, Any]:

    init_db()

    connection = get_connection()

    try:
        transaction_count = connection.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]

        assessment_count = connection.execute(
            "SELECT COUNT(*) FROM risk_assessments"
        ).fetchone()[0]

        case_count = connection.execute(
            "SELECT COUNT(*) FROM investigation_cases"
        ).fetchone()[0]

        return {
            "database": "sqlite",
            "path": str(DB_PATH),
            "transactions": transaction_count,
            "risk_assessments": assessment_count,
            "investigation_cases": case_count,
        }

    finally:
        connection.close()