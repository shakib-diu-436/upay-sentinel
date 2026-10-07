from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "transactions.csv"
)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "transactions.csv not found. "
            "Run python ml/generator.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required = [
        "transaction_id",
        "customer_id",
        "recipient_id",
        "amount",
        "recipient_new",
        "device_changed",
        "location_changed",
        "transactions_last_1h",
        "transactions_last_24h",
        "avg_amount_30d",
        "hour",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["amount_ratio"] = (
        df["amount"]
        / df["avg_amount_30d"].clip(lower=1)
    )

    df["unusual_hour"] = (
        df["hour"] < 6
    ).astype(int)

    df["high_amount"] = (
        df["amount_ratio"] >= 3
    ).astype(int)

    df["high_velocity"] = (
        df["transactions_last_1h"] >= 3
    ).astype(int)

    df["risk_signal_count"] = (
        df["recipient_new"]
        + df["device_changed"]
        + df["location_changed"]
        + df["high_amount"]
        + df["high_velocity"]
        + df["unusual_hour"]
    )

    return df


def analyze_network(
    customer_id: str | None = None,
    recipient_id: str | None = None,
    transaction_id: str | None = None,
) -> dict:
    df = load_data()

    # ----------------------------------------------------------
    # Resolve transaction -> customer / recipient
    # ----------------------------------------------------------

    if transaction_id:
        matches = df[
            df["transaction_id"].astype(str)
            == str(transaction_id)
        ]

        if not matches.empty:
            row = matches.iloc[0]

            customer_id = str(
                row["customer_id"]
            )

            recipient_id = str(
                row["recipient_id"]
            )

    # ----------------------------------------------------------
    # Find target recipient(s)
    # ----------------------------------------------------------

    target_recipients: set[str] = set()

    if recipient_id:
        target_recipients.add(
            str(recipient_id)
        )

    if customer_id:
        customer_rows = df[
            df["customer_id"].astype(str)
            == str(customer_id)
        ]

        target_recipients.update(
            customer_rows[
                "recipient_id"
            ]
            .astype(str)
            .tolist()
        )

    # ----------------------------------------------------------
    # If no target specified, select highest-degree recipients.
    # This gives the analyst an immediately useful network.
    # ----------------------------------------------------------

    if not target_recipients:

        recipient_degree = (
            df.groupby("recipient_id")[
                "customer_id"
            ]
            .nunique()
            .sort_values(
                ascending=False
            )
        )

        if recipient_degree.empty:
            return {
                "network_risk_score": 0.0,
                "network_type": "NO NETWORK DATA",
                "nodes": [],
                "edges": [],
                "signals": [],
            }

        target_recipients.add(
            str(recipient_degree.index[0])
        )

    # ----------------------------------------------------------
    # Build one-hop suspicious network
    # ----------------------------------------------------------

    relevant = df[
        df["recipient_id"]
        .astype(str)
        .isin(target_recipients)
    ].copy()

    # ----------------------------------------------------------
    # Expand to customers connected to target recipients
    # ----------------------------------------------------------

    connected_customers = (
        relevant["customer_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    # Include transactions of connected customers
    customer_activity = df[
        df["customer_id"]
        .astype(str)
        .isin(connected_customers)
    ].copy()

    # Keep graph manageable
    visible_recipients = set(
        customer_activity["recipient_id"]
        .astype(str)
        .value_counts()
        .head(8)
        .index
        .tolist()
    )

    graph_df = customer_activity[
        customer_activity["recipient_id"]
        .astype(str)
        .isin(visible_recipients)
    ].copy()

    # ----------------------------------------------------------
    # Graph nodes
    # ----------------------------------------------------------

    nodes = []

    customer_ids = (
        graph_df["customer_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    recipient_ids = (
        graph_df["recipient_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    for cid in customer_ids:
        customer_rows = graph_df[
            graph_df["customer_id"].astype(str)
            == cid
        ]

        activity_score = float(
            customer_rows[
                "risk_signal_count"
            ].mean()
        )

        nodes.append(
            {
                "id": cid,
                "label": cid,
                "type": "customer",
                "size": min(
                    20
                    + len(customer_rows) * 2,
                    50,
                ),
                "risk_activity": round(
                    activity_score,
                    2,
                ),
            }
        )

    for rid in recipient_ids:
        recipient_rows = graph_df[
            graph_df["recipient_id"].astype(str)
            == rid
        ]

        unique_customers = (
            recipient_rows[
                "customer_id"
            ]
            .astype(str)
            .nunique()
        )

        high_signal_txns = int(
            (
                recipient_rows[
                    "risk_signal_count"
                ]
                >= 2
            ).sum()
        )

        nodes.append(
            {
                "id": rid,
                "label": rid,
                "type": "recipient",
                "size": min(
                    28
                    + unique_customers * 4,
                    70,
                ),
                "connected_customers":
                    int(unique_customers),
                "high_signal_transactions":
                    high_signal_txns,
            }
        )

    # ----------------------------------------------------------
    # Graph edges
    # ----------------------------------------------------------

    edges = []

    grouped = (
        graph_df.groupby(
            [
                "customer_id",
                "recipient_id",
            ]
        )
        .agg(
            transactions=(
                "transaction_id",
                "count",
            ),
            total_amount=(
                "amount",
                "sum",
            ),
            avg_signal=(
                "risk_signal_count",
                "mean",
            ),
        )
        .reset_index()
    )

    for _, row in grouped.iterrows():

        edges.append(
            {
                "source": str(
                    row["customer_id"]
                ),
                "target": str(
                    row["recipient_id"]
                ),
                "transactions": int(
                    row["transactions"]
                ),
                "total_amount": round(
                    float(
                        row["total_amount"]
                    ),
                    2,
                ),
                "risk_signal": round(
                    float(
                        row["avg_signal"]
                    ),
                    2,
                ),
            }
        )

    # ----------------------------------------------------------
    # Mule scoring
    # ----------------------------------------------------------

    recipient_stats = defaultdict(
        lambda: {
            "customers": 0,
            "high_signal": 0,
            "transactions": 0,
            "amount": 0.0,
        }
    )

    for _, row in graph_df.iterrows():

        rid = str(
            row["recipient_id"]
        )

        recipient_stats[rid][
            "transactions"
        ] += 1

        recipient_stats[rid][
            "amount"
        ] += float(row["amount"])

    for rid in list(recipient_stats):

        rows = graph_df[
            graph_df["recipient_id"]
            .astype(str)
            == rid
        ]

        recipient_stats[rid][
            "customers"
        ] = int(
            rows["customer_id"]
            .astype(str)
            .nunique()
        )

        recipient_stats[rid][
            "high_signal"
        ] = int(
            (
                rows[
                    "risk_signal_count"
                ]
                >= 2
            ).sum()
        )

    best_recipient = None
    best_score = 0.0

    for rid, stats in recipient_stats.items():

        score = (
            stats["customers"] * 12
            + stats["high_signal"] * 7
            + min(
                stats["transactions"],
                10,
            ) * 2
        )

        score = min(
            100.0,
            float(score),
        )

        if score > best_score:
            best_score = score
            best_recipient = rid

    # ----------------------------------------------------------
    # Network signals
    # ----------------------------------------------------------

    signals = []

    if best_recipient:

        stats = recipient_stats[
            best_recipient
        ]

        if stats["customers"] >= 4:
            signals.append(
                "High fan-in: recipient is connected to "
                f"{stats['customers']} customers."
            )

        if stats["high_signal"] >= 3:
            signals.append(
                "Multiple transactions contain "
                "elevated risk signals."
            )

        if stats["transactions"] >= 5:
            signals.append(
                "High transaction concentration "
                "around a common recipient."
            )

        if stats["amount"] >= 50000:
            signals.append(
                "Large aggregate transaction volume "
                "through the recipient."
            )

    # ----------------------------------------------------------
    # Network classification
    # ----------------------------------------------------------

    if best_score >= 70:
        network_type = (
            "POSSIBLE MULE NETWORK"
        )
    elif best_score >= 45:
        network_type = (
            "ELEVATED NETWORK ACTIVITY"
        )
    else:
        network_type = (
            "NORMAL NETWORK ACTIVITY"
        )

    return {
        "network_risk_score": round(
            best_score,
            2,
        ),
        "network_type": network_type,
        "focus_recipient":
            best_recipient,
        "nodes": nodes,
        "edges": edges,
        "signals": signals,
        "network_explanation": (
            "Network risk is based on recipient "
            "connectivity, transaction concentration "
            "and repeated elevated-risk behavioural "
            "signals. It is an investigation aid, "
            "not a standalone fraud decision."
        ),
    }