from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_CUSTOMERS = 3000
N_RECIPIENTS = 8000
N_DEVICES = 6000
N_LOCATIONS = 500
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic" / "transactions.csv"


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def generate(n: int = 20000) -> pd.DataFrame:
    """
    Generate customer-centric synthetic mobile-wallet transactions.

    Design goals:
    - Each customer has persistent normal behavior (amount, time, device, location).
    - Recipients have per-customer history, so recipient_new is meaningful.
    - Device/location changes are relative to the customer's normal profile.
    - Transaction velocity is calculated from prior transactions, not future data.
    - A minority of events are injected with realistic suspicious patterns.
    - The target label is derived from multiple behavioral/contextual signals.

    All data is synthetic and contains no real personal information.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")

    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # 1. Customer profiles: persistent behavior for each customer
    # ------------------------------------------------------------------
    customer_count = min(N_CUSTOMERS, max(1, n // 5))
    customer_ids = np.arange(1, customer_count + 1)

    # Typical monthly/30-day transaction amount per customer.
    customer_avg_amount = np.clip(
        np.exp(rng.normal(np.log(850), 0.55, customer_count)),
        80,
        12000,
    )

    # Each customer has a habitual transaction hour and home location/device.
    customer_usual_hour = rng.integers(8, 22, customer_count)
    customer_home_location = rng.integers(1, N_LOCATIONS + 1, customer_count)
    customer_primary_device = rng.integers(1, N_DEVICES + 1, customer_count)
    customer_account_age = rng.integers(30, 2500, customer_count)

    # Each customer has a small stable set of frequently used recipients.
    recipient_pool = rng.integers(1, N_RECIPIENTS + 1, customer_count)

    # ------------------------------------------------------------------
    # 2. Generate an ordered transaction stream by customer.
    #    We create timestamps first so velocity can use prior events only.
    # ------------------------------------------------------------------
    rows: list[dict] = []
    known_recipients: dict[int, set[int]] = defaultdict(set)
    prior_events: dict[int, deque[pd.Timestamp]] = defaultdict(deque)

    # Rough target: ~6-8% suspicious/abnormal events after risk generation.
    # About 4% are explicitly injected high-risk scenarios.
    injection_prob = 0.04

    for i in range(n):
        customer_idx = int(rng.integers(0, customer_count))
        customer_num = int(customer_ids[customer_idx])

        base_amount = customer_avg_amount[customer_idx]
        usual_hour = int(customer_usual_hour[customer_idx])
        home_location = int(customer_home_location[customer_idx])
        primary_device = int(customer_primary_device[customer_idx])
        account_age_days = int(customer_account_age[customer_idx])

        # Use a realistic-ish 90-day window with some daytime concentration.
        day_offset = int(rng.integers(0, 90))
        hour = int(np.clip(round(rng.normal(usual_hour, 3.2)), 0, 23))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        timestamp = pd.Timestamp("2026-07-01") + pd.Timedelta(
            days=day_offset,
            hours=hour,
            minutes=minute,
            seconds=second,
        )

        # Decide whether this transaction is explicitly injected as suspicious.
        injected = bool(rng.random() < injection_prob)

        # --- Recipient behavior ----------------------------------------------
        if known_recipients[customer_num] and rng.random() < 0.88 and not injected:
            recipient_num = int(rng.choice(list(known_recipients[customer_num])))
            recipient_new = 0
        else:
            recipient_num = int(rng.integers(1, N_RECIPIENTS + 1))
            recipient_new = int(
                recipient_num not in known_recipients[customer_num]
            )

        # --- Device/location behavior ----------------------------------------
        if injected:
            device_changed = int(rng.random() < 0.78)
            location_changed = int(rng.random() < 0.72)
        else:
            device_changed = int(rng.random() < 0.025)
            location_changed = int(rng.random() < 0.035)

        device_num = (
            int(rng.integers(1, N_DEVICES + 1))
            if device_changed
            else primary_device
        )

        location_num = (
            int(rng.integers(1, N_LOCATIONS + 1))
            if location_changed
            else home_location
        )

        # --- Amount -----------------------------------------------------------
        if injected:
            # Large deviation from the customer's normal amount.
            multiplier = float(rng.uniform(5.0, 18.0))

            amount = _clip(
                base_amount * multiplier * rng.lognormal(0, 0.12),
                500,
                50000,
            )
        else:
            amount = _clip(
                base_amount * rng.lognormal(0, 0.42),
                20,
                15000,
            )

        # Make a few injected cases occur at atypical hours.
        if injected and rng.random() < 0.80:
            hour = int(
                rng.choice([0, 1, 2, 3, 4, 5, 23])
            )
            timestamp = timestamp.replace(hour=hour)

        # Keep only previous events when calculating transaction velocity.
        event_queue = prior_events[customer_num]

        while event_queue and (
            timestamp - event_queue[0]
        ).total_seconds() > 24 * 3600:
            event_queue.popleft()

        transactions_last_24h = len(event_queue)

        transactions_last_1h = sum(
            1
            for event_time in event_queue
            if (timestamp - event_time).total_seconds() <= 3600
        )

        # Inject a burst of activity for suspicious scenarios.
        if injected:
            transactions_last_1h += int(
                rng.integers(3, 8)
            )

            transactions_last_24h += int(
                rng.integers(3, 10)
            )

        # --- Behavioral deviation --------------------------------------------
        amount_ratio = amount / max(base_amount, 1.0)

        hour_distance = abs(hour - usual_hour)
        hour_distance = min(
            hour_distance,
            24 - hour_distance,
        )

        unusual_hour = int(
            hour < 6 or hour > 23
        )

        behavioral_deviation = float(
            np.clip(
                0.40 * min(amount_ratio / 8.0, 2.0)
                + 0.20 * min(hour_distance / 8.0, 2.0)
                + 0.15 * device_changed
                + 0.15 * location_changed
                + 0.10 * recipient_new,
                0,
                1,
            )
        )

        # ------------------------------------------------------------------
        # 3. Synthetic risk-generating process
        # ------------------------------------------------------------------
        risk_logit = (
            -5.1
            + 1.65 * recipient_new
            + 1.85 * device_changed
            + 1.45 * location_changed
            + 0.28 * min(transactions_last_1h, 10)
            + 0.10 * min(transactions_last_24h, 30)
            + 1.65 * unusual_hour
            + 1.25 * np.log1p(min(amount_ratio, 20))
            + 1.50 * behavioral_deviation
            + (1.20 if injected else 0.0)
            - 0.00018 * account_age_days
        )

        probability = float(
            1.0 / (1.0 + np.exp(-risk_logit))
        )

        # Strongly suspicious injected cases should almost always be labeled 1.
        # Other cases remain probabilistic to keep the dataset imperfect.
        if injected:
            is_suspicious = int(
                rng.random() < max(probability, 0.88)
            )
        else:
            is_suspicious = int(
                rng.random() < probability
            )

        # Persist this transaction as historical context for future transactions.
        event_queue.append(timestamp)
        known_recipients[customer_num].add(
            recipient_num
        )

        rows.append(
            {
                "transaction_id": f"TX{i + 1:06d}",
                "customer_id": f"C{customer_num:05d}",
                "recipient_id": f"R{recipient_num:05d}",
                "amount": round(amount, 2),
                "timestamp": timestamp.isoformat(),
                "hour": hour,
                "device_id": f"D{device_num:05d}",
                "location_id": f"L{location_num:04d}",
                "recipient_new": recipient_new,
                "device_changed": device_changed,
                "location_changed": location_changed,
                "transactions_last_1h": transactions_last_1h,
                "transactions_last_24h": transactions_last_24h,
                "avg_amount_30d": round(base_amount, 2),
                "usual_transaction_hour": usual_hour,
                "account_age_days": account_age_days,
                "amount_ratio": round(amount_ratio, 4),
                "hour_distance_from_usual": hour_distance,
                "behavioral_deviation_score": round(
                    behavioral_deviation,
                    4,
                ),
                "is_suspicious": is_suspicious,
            }
        )

    df = pd.DataFrame(rows)

    # Sort chronologically so the CSV mirrors an event stream.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    df["timestamp"] = df["timestamp"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT,
        index=False,
    )

    print(
        f"Generated {len(df):,} synthetic transactions → {OUT}"
    )

    print("\nClass distribution:")

    print(
        df["is_suspicious"]
        .value_counts()
        .sort_index()
    )

    print("\nClass share:")

    print(
        df["is_suspicious"]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    print(
        "\nDataset shape:",
        df.shape,
    )

    print("\nSample:")

    print(
        df.head(5).to_string(index=False)
    )

    return df


if __name__ == "__main__":
    generate()