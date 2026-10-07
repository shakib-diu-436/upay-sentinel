from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

N_CUSTOMERS = 3000
N_RECIPIENTS = 8000
N_DEVICES = 6000
N_LOCATIONS = 500

START_DATE = pd.Timestamp("2026-07-01")
END_DATE = pd.Timestamp("2026-09-28 23:59:59")

INJECTION_PROB = 0.04

RNG = np.random.default_rng(SEED)


ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "data"
    / "synthetic"
    / "transactions.csv"
)


# ============================================================
# HELPERS
# ============================================================

def clip(
    value: float,
    low: float,
    high: float,
) -> float:
    return float(
        np.clip(
            value,
            low,
            high,
        )
    )


def sigmoid(value: float) -> float:
    value = float(
        np.clip(
            value,
            -30.0,
            30.0,
        )
    )

    return float(
        1.0 / (1.0 + np.exp(-value))
    )


def circular_hour_distance(
    hour: int,
    usual_hour: int,
) -> int:

    difference = abs(
        hour - usual_hour
    )

    return int(
        min(
            difference,
            24 - difference,
        )
    )


def build_behavioral_deviation(
    amount_ratio: float,
    hour_distance: int,
    device_changed: int,
    location_changed: int,
    recipient_new: int,
) -> float:

    return float(
        np.clip(
            (
                0.40
                * min(
                    amount_ratio / 8.0,
                    2.0,
                )
                + 0.20
                * min(
                    hour_distance / 8.0,
                    2.0,
                )
                + 0.15
                * device_changed
                + 0.15
                * location_changed
                + 0.10
                * recipient_new
            ),
            0.0,
            1.0,
        )
    )


# ============================================================
# GENERATOR
# ============================================================

def generate(
    n: int = 20000,
) -> pd.DataFrame:
    """
    Generate customer-centric synthetic mobile-wallet
    transactions.

    Design goals:

    1. Persistent customer behaviour.
    2. Customer-specific recipient history.
    3. Device and location changes relative to normal profile.
    4. Transaction velocity calculated only from actual
       previous transactions.
    5. 30-day average amount calculated from actual previous
       customer transactions when available.
    6. A minority of transactions contain suspicious patterns.
    7. Suspicious labels are generated only from observable
       transaction and behavioural features.
    8. No hidden injection flag is used by the label model.
    9. Synthetic labels remain probabilistic rather than
       deterministic.
    """

    if n <= 0:
        raise ValueError(
            "n must be greater than 0"
        )

    rng = np.random.default_rng(SEED)

    # --------------------------------------------------------
    # Customer profiles
    # --------------------------------------------------------

    customer_count = min(
        N_CUSTOMERS,
        max(1, n // 5),
    )

    customer_ids = np.arange(
        1,
        customer_count + 1,
    )

    customer_avg_amount = np.clip(
        np.exp(
            rng.normal(
                np.log(850),
                0.55,
                customer_count,
            )
        ),
        80,
        12000,
    )

    customer_usual_hour = rng.integers(
        8,
        22,
        customer_count,
    )

    customer_home_location = rng.integers(
        1,
        N_LOCATIONS + 1,
        customer_count,
    )

    customer_primary_device = rng.integers(
        1,
        N_DEVICES + 1,
        customer_count,
    )

    customer_account_age = rng.integers(
        30,
        2500,
        customer_count,
    )

    # --------------------------------------------------------
    # Stable recipient pools
    # --------------------------------------------------------

    recipient_pools: dict[
        int,
        set[int]
    ] = {}

    for customer_num in customer_ids:

        pool_size = int(
            rng.integers(
                3,
                8,
            )
        )

        recipients = rng.choice(
            np.arange(
                1,
                N_RECIPIENTS + 1,
            ),
            size=pool_size,
            replace=False,
        )

        recipient_pools[
            int(customer_num)
        ] = set(
            int(value)
            for value in recipients
        )

    # --------------------------------------------------------
    # Give every customer at least one transaction.
    # Remaining transactions are distributed randomly.
    # --------------------------------------------------------

    remaining = (
        n - customer_count
    )

    extra_counts = rng.multinomial(
        remaining,
        np.full(
            customer_count,
            1.0 / customer_count,
        ),
    )

    event_counts = (
        np.ones(
            customer_count,
            dtype=int,
        )
        + extra_counts
    )

    # --------------------------------------------------------
    # Generate rows customer by customer.
    #
    # This guarantees chronological history for each customer.
    # --------------------------------------------------------

    rows: list[dict] = []

    for customer_idx, count in enumerate(
        event_counts
    ):

        customer_num = int(
            customer_ids[customer_idx]
        )

        base_amount = float(
            customer_avg_amount[
                customer_idx
            ]
        )

        usual_hour = int(
            customer_usual_hour[
                customer_idx
            ]
        )

        home_location = int(
            customer_home_location[
                customer_idx
            ]
        )

        primary_device = int(
            customer_primary_device[
                customer_idx
            ]
        )

        account_age_days = int(
            customer_account_age[
                customer_idx
            ]
        )

        # ----------------------------------------------------
        # Decide suspicious-pattern injections.
        #
        # IMPORTANT:
        # This flag affects observable behaviour only.
        # It is never directly used in label generation.
        # ----------------------------------------------------

        injected_flags = (
            rng.random(count)
            < INJECTION_PROB
        )

        # ----------------------------------------------------
        # Build chronological timestamps for this customer.
        #
        # Injected events often happen shortly after an earlier
        # event, creating REAL velocity rather than adding fake
        # velocity counts.
        # ----------------------------------------------------

        offsets_hours = np.zeros(
            count,
            dtype=float,
        )

        for index in range(
            1,
            count,
        ):

            if (
                injected_flags[index]
                and rng.random() < 0.80
            ):
                gap_hours = (
                    rng.uniform(
                        10,
                        50,
                    ) / 60.0
                )
            else:
                gap_hours = float(
                    np.clip(
                        rng.exponential(
                            11.0 * 24.0
                        ),
                        1.0,
                        25.0 * 24.0,
                    )
                )

            offsets_hours[index] = (
                offsets_hours[index - 1]
                + gap_hours
            )

        # Keep the complete customer stream inside
        # the requested 90-day observation window.
        max_span_hours = (
            END_DATE
            - START_DATE
        ).total_seconds() / 3600.0

        total_span = offsets_hours[-1]

        if total_span > max_span_hours:
            scale = (
                max_span_hours
                / total_span
            )

            offsets_hours *= scale

        first_offset_hours = float(
            rng.uniform(
                0,
                24,
            )
        )

        timestamps = [
            START_DATE
            + pd.Timedelta(
                hours=first_offset_hours
                + float(offset),
            )
            for offset in offsets_hours
        ]

        # ----------------------------------------------------
        # Customer history
        # ----------------------------------------------------

        known_recipients = set(
            recipient_pools[
                customer_num
            ]
        )

        prior_events: deque[
            pd.Timestamp
        ] = deque()

        amount_history: deque[
            tuple[
                pd.Timestamp,
                float,
            ]
        ] = deque()

        # ----------------------------------------------------
        # Generate each transaction
        # ----------------------------------------------------

        for index in range(
            count
        ):

            timestamp = timestamps[
                index
            ]

            injected = bool(
                injected_flags[index]
            )

            # ------------------------------------------------
            # Remove old history.
            # ------------------------------------------------

            while (
                prior_events
                and (
                    timestamp
                    - prior_events[0]
                ).total_seconds()
                > 24 * 3600
            ):
                prior_events.popleft()

            while (
                amount_history
                and (
                    timestamp
                    - amount_history[0][0]
                ).total_seconds()
                > 30 * 24 * 3600
            ):
                amount_history.popleft()

            # ------------------------------------------------
            # Actual customer 30-day average.
            # ------------------------------------------------

            if amount_history:

                avg_amount_30d = float(
                    np.mean(
                        [
                            amount
                            for _, amount
                            in amount_history
                        ]
                    )
                )

            else:

                avg_amount_30d = base_amount

            # ------------------------------------------------
            # Transaction timing
            # ------------------------------------------------

            hour = int(
                timestamp.hour
            )

            # ------------------------------------------------
            # Recipient
            # ------------------------------------------------

            if (
                injected
                and rng.random() < 0.75
            ):

                recipient_num = int(
                    rng.integers(
                        1,
                        N_RECIPIENTS + 1,
                    )
                )

                while (
                    recipient_num
                    in known_recipients
                ):
                    recipient_num = int(
                        rng.integers(
                            1,
                            N_RECIPIENTS + 1,
                        )
                    )

            elif (
                known_recipients
                and rng.random() < 0.88
            ):

                recipient_num = int(
                    rng.choice(
                        list(
                            known_recipients
                        )
                    )
                )

            else:

                recipient_num = int(
                    rng.integers(
                        1,
                        N_RECIPIENTS + 1,
                    )
                )

            recipient_new = int(
                recipient_num
                not in known_recipients
            )

            # ------------------------------------------------
            # Device and location
            # ------------------------------------------------

            if injected:

                device_changed = int(
                    rng.random() < 0.78
                )

                location_changed = int(
                    rng.random() < 0.72
                )

            else:

                device_changed = int(
                    rng.random() < 0.025
                )

                location_changed = int(
                    rng.random() < 0.035
                )

            device_num = (
                int(
                    rng.integers(
                        1,
                        N_DEVICES + 1,
                    )
                )
                if device_changed
                else primary_device
            )

            location_num = (
                int(
                    rng.integers(
                        1,
                        N_LOCATIONS + 1,
                    )
                )
                if location_changed
                else home_location
            )

            # ------------------------------------------------
            # Amount
            # ------------------------------------------------

            if injected:

                multiplier = float(
                    rng.uniform(
                        5.0,
                        18.0,
                    )
                )

                amount = clip(
                    avg_amount_30d
                    * multiplier
                    * rng.lognormal(
                        0,
                        0.12,
                    ),
                    500,
                    50000,
                )

            else:

                amount = clip(
                    avg_amount_30d
                    * rng.lognormal(
                        0,
                        0.42,
                    ),
                    20,
                    15000,
                )

            # ------------------------------------------------
            # Inject unusual time for suspicious patterns.
            # ------------------------------------------------

            if (
                injected
                and rng.random() < 0.80
            ):

                unusual_hours = [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    23,
                ]

                hour = int(
                    rng.choice(
                        unusual_hours
                    )
                )

                timestamp = timestamp.replace(
                    hour=hour
                )

            # ------------------------------------------------
            # Actual transaction velocity
            # ------------------------------------------------

            transactions_last_24h = len(
                prior_events
            )

            transactions_last_1h = sum(
                1
                for event_time
                in prior_events
                if (
                    timestamp
                    - event_time
                ).total_seconds()
                <= 3600
            )

            # ------------------------------------------------
            # Derived behavioural features
            # ------------------------------------------------

            amount_ratio = (
                amount
                / max(
                    avg_amount_30d,
                    1.0,
                )
            )

            hour_distance = (
                circular_hour_distance(
                    hour,
                    usual_hour,
                )
            )

            unusual_hour = int(
                hour < 6
            )

            behavioral_deviation = (
                build_behavioral_deviation(
                    amount_ratio,
                    hour_distance,
                    device_changed,
                    location_changed,
                    recipient_new,
                )
            )

            high_amount = int(
                amount_ratio >= 3.0
            )

            high_velocity = int(
                transactions_last_1h >= 3
            )

            # ------------------------------------------------
            # OBSERVABLE synthetic risk-generating process
            #
            # IMPORTANT:
            # No "injected" flag appears here.
            #
            # Nonlinear interactions are intentional because
            # they give the ML model meaningful interactions
            # beyond simple signal counting.
            # ------------------------------------------------

            interaction_score = (
                1.10
                * recipient_new
                * device_changed
                + 0.90
                * high_amount
                * device_changed
                + 0.80
                * high_amount
                * high_velocity
                + 0.70
                * recipient_new
                * unusual_hour
                + 0.55
                * device_changed
                * location_changed
            )

            risk_logit = (
                -5.1
                + 1.20
                * recipient_new
                + 1.55
                * device_changed
                + 1.25
                * location_changed
                + 0.32
                * min(
                    transactions_last_1h,
                    8,
                )
                + 0.08
                * min(
                    transactions_last_24h,
                    24,
                )
                + 1.45
                * unusual_hour
                + 1.15
                * np.log1p(
                    min(
                        amount_ratio,
                        20,
                    )
                )
                + 1.25
                * behavioral_deviation
                + 0.55
                * high_amount
                + 0.50
                * high_velocity
                + interaction_score
                - 0.00012
                * account_age_days
            )

            probability = sigmoid(
                risk_logit
            )

            # ------------------------------------------------
            # Probabilistic label.
            #
            # No deterministic "injected -> suspicious"
            # override is used.
            # ------------------------------------------------

            is_suspicious = int(
                rng.random()
                < probability
            )

            # ------------------------------------------------
            # Persist current event into history.
            # ------------------------------------------------

            prior_events.append(
                timestamp
            )

            amount_history.append(
                (
                    timestamp,
                    amount,
                )
            )

            known_recipients.add(
                recipient_num
            )

            # ------------------------------------------------
            # Store row
            # ------------------------------------------------

            rows.append(
                {
                    "transaction_id": (
                        f"TX{len(rows) + 1:06d}"
                    ),
                    "customer_id": (
                        f"C{customer_num:05d}"
                    ),
                    "recipient_id": (
                        f"R{recipient_num:05d}"
                    ),
                    "amount": round(
                        amount,
                        2,
                    ),
                    "timestamp": (
                        timestamp.isoformat()
                    ),
                    "hour": hour,
                    "device_id": (
                        f"D{device_num:05d}"
                    ),
                    "location_id": (
                        f"L{location_num:04d}"
                    ),
                    "recipient_new": (
                        recipient_new
                    ),
                    "device_changed": (
                        device_changed
                    ),
                    "location_changed": (
                        location_changed
                    ),
                    "transactions_last_1h": (
                        transactions_last_1h
                    ),
                    "transactions_last_24h": (
                        transactions_last_24h
                    ),
                    "avg_amount_30d": round(
                        avg_amount_30d,
                        2,
                    ),
                    "usual_transaction_hour": (
                        usual_hour
                    ),
                    "account_age_days": (
                        account_age_days
                    ),
                    "amount_ratio": round(
                        amount_ratio,
                        4,
                    ),
                    "hour_distance_from_usual": (
                        hour_distance
                    ),
                    "behavioral_deviation_score": round(
                        behavioral_deviation,
                        4,
                    ),
                    "is_suspicious": (
                        is_suspicious
                    ),
                }
            )

    # ========================================================
    # Final dataframe
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Sort globally by timestamp.
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df.sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    df["timestamp"] = (
        df["timestamp"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT,
        index=False,
    )

    # ========================================================
    # Reporting
    # ========================================================

    print(
        f"Generated {len(df):,} "
        f"synthetic transactions → {OUT}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["is_suspicious"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nClass share:"
    )

    print(
        df["is_suspicious"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .round(4)
    )

    print(
        "\nDataset shape:"
    )

    print(
        df.shape
    )

    print(
        "\nSignal prevalence:"
    )

    signal_summary = pd.DataFrame(
        {
            "Normal": [
                df.loc[
                    df["is_suspicious"] == 0,
                    "recipient_new",
                ].mean(),
                df.loc[
                    df["is_suspicious"] == 0,
                    "device_changed",
                ].mean(),
                df.loc[
                    df["is_suspicious"] == 0,
                    "location_changed",
                ].mean(),
                (
                    df.loc[
                        df["is_suspicious"] == 0,
                        "amount_ratio",
                    ] >= 3
                ).mean(),
                (
                    df.loc[
                        df["is_suspicious"] == 0,
                        "hour",
                    ] < 6
                ).mean(),
            ],
            "Suspicious": [
                df.loc[
                    df["is_suspicious"] == 1,
                    "recipient_new",
                ].mean(),
                df.loc[
                    df["is_suspicious"] == 1,
                    "device_changed",
                ].mean(),
                df.loc[
                    df["is_suspicious"] == 1,
                    "location_changed",
                ].mean(),
                (
                    df.loc[
                        df["is_suspicious"] == 1,
                        "amount_ratio",
                    ] >= 3
                ).mean(),
                (
                    df.loc[
                        df["is_suspicious"] == 1,
                        "hour",
                    ] < 6
                ).mean(),
            ],
        },
        index=[
            "New recipient",
            "Device changed",
            "Location changed",
            "Amount >= 3x average",
            "Hour before 06:00",
        ],
    )

    print(
        signal_summary.round(4)
    )

    print(
        "\nSample:"
    )

    print(
        df.head(5)
        .to_string(
            index=False
        )
    )

    return df


if __name__ == "__main__":
    generate()