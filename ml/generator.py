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

N_TRANSACTIONS = 20000

START_DATE = pd.Timestamp("2026-07-01")
END_DATE = pd.Timestamp("2026-09-28 23:59:59")

# Around 4% of transaction streams start an explicitly
# suspicious behavioral scenario.
INJECTION_PROB = 0.04

# Preserve approximately the original dataset prevalence.
TARGET_SUSPICIOUS_RATE = 0.1077


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


def sigmoid(
    values,
) -> np.ndarray:

    values = np.asarray(values, dtype=float)

    values = np.clip(
        values,
        -30.0,
        30.0,
    )

    return 1.0 / (
        1.0 + np.exp(-values)
    )


def circular_hour_distance(
    hour: int,
    usual_hour: int,
) -> int:

    difference = abs(
        int(hour) - int(usual_hour)
    )

    return int(
        min(
            difference,
            24 - difference,
        )
    )


def behavioral_deviation_score(
    amount_ratio: float,
    hour_distance: int,
    device_changed: int,
    location_changed: int,
    recipient_new: int,
) -> float:

    score = (
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
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def calibrate_intercept(
    raw_logits: np.ndarray,
    target_rate: float,
) -> float:
    """
    Find a constant intercept adjustment so that
    mean(sigmoid(raw_logits + offset)) is approximately
    equal to target_rate.

    This controls class prevalence during synthetic-data
    generation without using the target label itself.
    """

    low = -10.0
    high = 10.0

    for _ in range(80):

        middle = (
            low + high
        ) / 2.0

        probabilities = sigmoid(
            raw_logits + middle
        )

        mean_probability = float(
            probabilities.mean()
        )

        if mean_probability < target_rate:
            low = middle
        else:
            high = middle

    return (
        low + high
    ) / 2.0


# ============================================================
# GENERATOR
# ============================================================

def generate(
    n: int = N_TRANSACTIONS,
) -> pd.DataFrame:
    """
    Generate customer-centric synthetic mobile-wallet
    transactions.

    Design goals:

    1. Persistent customer profiles.
    2. Customer-specific recipient history.
    3. Device/location changes relative to customer norms.
    4. Actual historical transaction velocity.
    5. Actual previous 30-day amount history.
    6. Realistic customer transaction timing.
    7. Explicit suspicious scenarios alter observable behavior.
    8. No hidden injection variable is used in label generation.
    9. Labels are probabilistic and noisy.
    10. Nonlinear interactions create meaningful structure
        for machine-learning models.
    """

    if n <= 0:
        raise ValueError(
            "n must be greater than 0"
        )

    rng = np.random.default_rng(
        SEED
    )

    # --------------------------------------------------------
    # Customer count
    # --------------------------------------------------------

    customer_count = min(
        N_CUSTOMERS,
        max(
            1,
            n // 5,
        ),
    )

    customer_ids = np.arange(
        1,
        customer_count + 1,
    )

    # --------------------------------------------------------
    # Persistent customer profiles
    # --------------------------------------------------------

    customer_avg_amount = np.clip(
        np.exp(
            rng.normal(
                np.log(850.0),
                0.55,
                customer_count,
            )
        ),
        80.0,
        12000.0,
    )

    customer_usual_hour = rng.integers(
        8,
        22,
        customer_count,
    )

    customer_home_location = (
        rng.integers(
            1,
            N_LOCATIONS + 1,
            customer_count,
        )
    )

    customer_primary_device = (
        rng.integers(
            1,
            N_DEVICES + 1,
            customer_count,
        )
    )

    customer_account_age = (
        rng.integers(
            30,
            2500,
            customer_count,
        )
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

        pool = rng.choice(
            np.arange(
                1,
                N_RECIPIENTS + 1,
            ),
            size=pool_size,
            replace=False,
        )

        recipient_pools[
            int(customer_num)
        ] = {
            int(value)
            for value in pool
        }

    # --------------------------------------------------------
    # Transaction counts per customer
    #
    # Ensure every customer has at least one transaction.
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
    # Feature rows before labels
    # --------------------------------------------------------

    rows: list[dict] = []

    # --------------------------------------------------------
    # Generate per customer
    # --------------------------------------------------------

    for customer_idx, count in enumerate(
        event_counts
    ):

        customer_num = int(
            customer_ids[
                customer_idx
            ]
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
        # Choose distinct days for normal transactions.
        #
        # Distinct days make customer history chronological
        # while keeping normal transactions spread across
        # the 90-day observation window.
        # ----------------------------------------------------

        day_offsets = np.sort(
            rng.choice(
                np.arange(
                    0,
                    90,
                ),
                size=count,
                replace=False,
            )
        )

        timestamps = []

        for day_offset in day_offsets:

            # Normal transactions concentrate around the
            # customer's usual hour.
            hour = int(
                np.clip(
                    round(
                        rng.normal(
                            usual_hour,
                            2.0,
                        )
                    ),
                    7,
                    23,
                )
            )

            minute = int(
                rng.integers(
                    0,
                    60,
                )
            )

            second = int(
                rng.integers(
                    0,
                    60,
                )
            )

            timestamp = (
                START_DATE
                + pd.Timedelta(
                    days=int(
                        day_offset
                    ),
                    hours=hour,
                    minutes=minute,
                    seconds=second,
                )
            )

            timestamps.append(
                timestamp
            )

        # ----------------------------------------------------
        # Create actual suspicious bursts.
        #
        # A scenario start creates 4-5 transactions close
        # together. Velocity will therefore be calculated
        # from actual previous events.
        # ----------------------------------------------------

        pattern_flags = np.zeros(
            count,
            dtype=bool,
        )

        scenario_start_flags = np.zeros(
            count,
            dtype=bool,
        )

        index = 1

        while (
            index < count
        ):

            if (
                rng.random()
                < INJECTION_PROB
                and index + 3 < count
                and not pattern_flags[
                    max(
                        0,
                        index - 3,
                    ):index
                ].any()
            ):

                scenario_start_flags[
                    index
                ] = True

                # Burst length: 4 or 5 events.
                burst_length = int(
                    rng.integers(
                        4,
                        6,
                    )
                )

                burst_length = min(
                    burst_length,
                    count - index,
                )

                burst_indices = range(
                    index,
                    index
                    + burst_length,
                )

                for burst_index in (
                    burst_indices
                ):
                    pattern_flags[
                        burst_index
                    ] = True

                # Put the burst on the current event's day.
                burst_date = (
                    timestamps[index]
                    .normalize()
                )

                # 80% of suspicious scenarios occur
                # during an unusual hour.
                if (
                    rng.random()
                    < 0.80
                ):
                    burst_hour = int(
                        rng.choice(
                            [
                                0,
                                1,
                                2,
                                3,
                                4,
                                5,
                                23,
                            ]
                        )
                    )
                else:
                    burst_hour = int(
                        np.clip(
                            rng.normal(
                                usual_hour,
                                2.0,
                            ),
                            7,
                            23,
                        )
                    )

                burst_start = (
                    burst_date
                    + pd.Timedelta(
                        hours=burst_hour,
                        minutes=int(
                            rng.integers(
                                0,
                                45,
                            )
                        ),
                        seconds=int(
                            rng.integers(
                                0,
                                60,
                            )
                        ),
                    )
                )

                timestamps[index] = (
                    burst_start
                )

                # Keep each burst transaction within
                # approximately one hour.
                current_time = (
                    burst_start
                )

                for burst_index in range(
                    index + 1,
                    index
                    + burst_length,
                ):

                    current_time = (
                        current_time
                        + pd.Timedelta(
                            minutes=int(
                                rng.integers(
                                    8,
                                    18,
                                )
                            )
                        )
                    )

                    timestamps[
                        burst_index
                    ] = current_time

                index += (
                    burst_length
                )

            else:

                index += 1

        # ----------------------------------------------------
        # Customer state/history
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
        # Process events chronologically
        # ----------------------------------------------------

        for event_index in range(
            count
        ):

            timestamp = timestamps[
                event_index
            ]

            scenario_start = bool(
                scenario_start_flags[
                    event_index
                ]
            )

            in_pattern = bool(
                pattern_flags[
                    event_index
                ]
            )

            # ------------------------------------------------
            # Remove historical events outside windows.
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
            # Historical 30-day average
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
            # Current hour
            # ------------------------------------------------

            hour = int(
                timestamp.hour
            )

            # ------------------------------------------------
            # Recipient
            # ------------------------------------------------

            if scenario_start:

                # Strong suspicious scenario:
                # high probability of new recipient.
                use_new_recipient = (
                    rng.random()
                    < 0.78
                )

            elif in_pattern:

                # Burst followers still have a higher
                # chance of recipient novelty, but less
                # aggressively than the scenario start.
                use_new_recipient = (
                    rng.random()
                    < 0.30
                )

            else:

                use_new_recipient = (
                    rng.random()
                    >= 0.88
                )

            if (
                use_new_recipient
            ):

                recipient_num = int(
                    rng.integers(
                        1,
                        N_RECIPIENTS + 1,
                    )
                )

                # Ensure genuinely new for this customer.
                attempts = 0

                while (
                    recipient_num
                    in known_recipients
                    and attempts < 20
                ):

                    recipient_num = int(
                        rng.integers(
                            1,
                            N_RECIPIENTS + 1,
                        )
                    )

                    attempts += 1

            else:

                recipient_num = int(
                    rng.choice(
                        list(
                            known_recipients
                        )
                    )
                )

            recipient_new = int(
                recipient_num
                not in known_recipients
            )

            # ------------------------------------------------
            # Device/location changes
            # ------------------------------------------------

            if scenario_start:

                device_changed = int(
                    rng.random()
                    < 0.78
                )

                location_changed = int(
                    rng.random()
                    < 0.72
                )

            elif in_pattern:

                device_changed = int(
                    rng.random()
                    < 0.30
                )

                location_changed = int(
                    rng.random()
                    < 0.28
                )

            else:

                device_changed = int(
                    rng.random()
                    < 0.025
                )

                location_changed = int(
                    rng.random()
                    < 0.035
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

            if scenario_start:

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
                        0.0,
                        0.12,
                    ),
                    500,
                    50000,
                )

            elif in_pattern:

                multiplier = float(
                    rng.uniform(
                        1.5,
                        4.0,
                    )
                )

                amount = clip(
                    avg_amount_30d
                    * multiplier
                    * rng.lognormal(
                        0.0,
                        0.20,
                    ),
                    50,
                    25000,
                )

            else:

                amount = clip(
                    avg_amount_30d
                    * rng.lognormal(
                        0.0,
                        0.42,
                    ),
                    20,
                    15000,
                )

            # ------------------------------------------------
            # Actual historical velocity.
            #
            # No artificial count is added here.
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
            # Behavioral features
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
                or hour > 22
            )

            behavioral_score = (
                behavioral_deviation_score(
                    amount_ratio,
                    hour_distance,
                    device_changed,
                    location_changed,
                    recipient_new,
                )
            )

            # ------------------------------------------------
            # Store transaction row.
            #
            # Label is calculated later for all rows.
            # ------------------------------------------------

            rows.append(
                {
                    "transaction_id": "",
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
                        behavioral_score,
                        4,
                    ),
                }
            )

            # ------------------------------------------------
            # Persist current event after features are built.
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

    # ========================================================
    # DataFrame
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # --------------------------------------------------------
    # Global chronological order
    # --------------------------------------------------------

    df = (
        df.sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Sequential transaction IDs after final ordering.
    # --------------------------------------------------------

    df["transaction_id"] = [
        f"TX{index + 1:06d}"
        for index in range(
            len(df)
        )
    ]

    # --------------------------------------------------------
    # Make sure timestamp string format is consistent.
    # --------------------------------------------------------

    df["timestamp"] = (
        df["timestamp"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    )

    # ========================================================
    # Observable synthetic risk-generating process
    # ========================================================
    #
    # IMPORTANT:
    # There is NO reference to scenario_start_flags,
    # pattern_flags, or any hidden injection variable here.
    #
    # The label depends only on observable transaction
    # and behavioral signals.
    # ========================================================

    amount_ratio = df[
        "amount_ratio"
    ].to_numpy(
        dtype=float
    )

    recipient_new = df[
        "recipient_new"
    ].to_numpy(
        dtype=float
    )

    device_changed = df[
        "device_changed"
    ].to_numpy(
        dtype=float
    )

    location_changed = df[
        "location_changed"
    ].to_numpy(
        dtype=float
    )

    tx_1h = df[
        "transactions_last_1h"
    ].to_numpy(
        dtype=float
    )

    tx_24h = df[
        "transactions_last_24h"
    ].to_numpy(
        dtype=float
    )

    unusual_hour = (
        (
            df["hour"].to_numpy()
            < 6
        )
        |
        (
            df["hour"].to_numpy()
            > 22
        )
    ).astype(float)

    behavioral_score = df[
        "behavioral_deviation_score"
    ].to_numpy(
        dtype=float
    )

    account_age = df[
        "account_age_days"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Observable threshold features
    # --------------------------------------------------------

    high_amount = (
        amount_ratio >= 3.0
    ).astype(float)

    high_velocity = (
        tx_1h >= 3
    ).astype(float)

    # --------------------------------------------------------
    # Nonlinear interactions.
    #
    # These are observable combinations that a learned model
    # can represent more flexibly than a simple signal count.
    # --------------------------------------------------------

    interaction_score = (
        1.35
        * recipient_new
        * device_changed

        + 1.05
        * high_amount
        * device_changed

        + 0.95
        * high_amount
        * high_velocity

        + 0.85
        * recipient_new
        * unusual_hour

        + 0.70
        * device_changed
        * location_changed

        + 0.60
        * high_velocity
        * unusual_hour
    )

    raw_logits = (
        -5.5

        + 1.20
        * recipient_new

        + 1.55
        * device_changed

        + 1.25
        * location_changed

        + 0.34
        * np.minimum(
            tx_1h,
            8,
        )

        + 0.08
        * np.minimum(
            tx_24h,
            24,
        )

        + 1.55
        * unusual_hour

        + 1.20
        * np.log1p(
            np.minimum(
                amount_ratio,
                20,
            )
        )

        + 1.20
        * behavioral_score

        + 0.50
        * high_amount

        + 0.45
        * high_velocity

        + interaction_score

        - 0.00012
        * account_age
    )

    # --------------------------------------------------------
    # Calibrate synthetic prevalence.
    # --------------------------------------------------------

    intercept_adjustment = (
        calibrate_intercept(
            raw_logits,
            TARGET_SUSPICIOUS_RATE,
        )
    )

    probabilities = sigmoid(
        raw_logits
        + intercept_adjustment
    )

    # --------------------------------------------------------
    # Probabilistic noisy label.
    #
    # No deterministic suspicious override is used.
    # --------------------------------------------------------

    labels = (
        rng.random(
            len(df)
        )
        < probabilities
    ).astype(int)

    df["is_suspicious"] = labels

    # ========================================================
    # Save
    # ========================================================

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT,
        index=False,
    )

    # ========================================================
    # REPORTING
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
        "\nTarget suspicious rate:"
    )

    print(
        f"{TARGET_SUSPICIOUS_RATE:.4f}"
    )

    print(
        "\nMean generated probability:"
    )

    print(
        f"{probabilities.mean():.4f}"
    )

    print(
        "\nDataset shape:"
    )

    print(
        df.shape
    )

    # --------------------------------------------------------
    # Signal prevalence
    # --------------------------------------------------------

    print(
        "\nSignal prevalence:"
    )

    normal = (
        df["is_suspicious"]
        == 0
    )

    suspicious = (
        df["is_suspicious"]
        == 1
    )

    signal_summary = pd.DataFrame(
        {
            "Normal": [
                df.loc[
                    normal,
                    "recipient_new",
                ].mean(),

                df.loc[
                    normal,
                    "device_changed",
                ].mean(),

                df.loc[
                    normal,
                    "location_changed",
                ].mean(),

                (
                    df.loc[
                        normal,
                        "amount_ratio",
                    ]
                    >= 3.0
                ).mean(),

                (
                    df.loc[
                        normal,
                        "hour",
                    ]
                    < 6
                ).mean(),
            ],

            "Suspicious": [
                df.loc[
                    suspicious,
                    "recipient_new",
                ].mean(),

                df.loc[
                    suspicious,
                    "device_changed",
                ].mean(),

                df.loc[
                    suspicious,
                    "location_changed",
                ].mean(),

                (
                    df.loc[
                        suspicious,
                        "amount_ratio",
                    ]
                    >= 3.0
                ).mean(),

                (
                    df.loc[
                        suspicious,
                        "hour",
                    ]
                    < 6
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

    # --------------------------------------------------------
    # Velocity statistics
    # --------------------------------------------------------

    print(
        "\nVelocity statistics:"
    )

    print(
        "transactions_last_1h max:",
        df[
            "transactions_last_1h"
        ].max(),
    )

    print(
        "transactions_last_24h max:",
        df[
            "transactions_last_24h"
        ].max(),
    )

    print(
        "transactions_last_1h >= 3:",
        (
            df[
                "transactions_last_1h"
            ] >= 3
        ).mean(),
    )

    # --------------------------------------------------------
    # Hour statistics
    # --------------------------------------------------------

    print(
        "\nHour statistics:"
    )

    print(
        "Normal before 06:00:",
        (
            df.loc[
                normal,
                "hour",
            ]
            < 6
        ).mean(),
    )

    print(
        "Suspicious before 06:00:",
        (
            df.loc[
                suspicious,
                "hour",
            ]
            < 6
        ).mean(),
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print(
        "\nSample:"
    )

    print(
        df.head(10)
        .to_string(
            index=False
        )
    )

    return df


if __name__ == "__main__":
    generate()