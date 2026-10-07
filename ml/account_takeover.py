from __future__ import annotations


def analyze_account_takeover(
    transaction: dict,
) -> dict:

    amount = float(
        transaction.get(
            "amount",
            0,
        )
    )

    avg_amount = float(
        transaction.get(
            "avg_amount_30d",
            1,
        )
    )

    amount_ratio = (
        amount
        / max(avg_amount, 1.0)
    )

    hour = int(
        transaction.get(
            "hour",
            12,
        )
    )

    recipient_new = int(
        transaction.get(
            "recipient_new",
            0,
        )
    )

    device_changed = int(
        transaction.get(
            "device_changed",
            0,
        )
    )

    location_changed = int(
        transaction.get(
            "location_changed",
            0,
        )
    )

    transactions_last_1h = int(
        transaction.get(
            "transactions_last_1h",
            0,
        )
    )

    usual_hour = int(
        transaction.get(
            "usual_transaction_hour",
            12,
        )
    )

    # ----------------------------------------------------------
    # Circular hour distance
    # ----------------------------------------------------------

    hour_distance = abs(
        hour - usual_hour
    )

    hour_distance = min(
        hour_distance,
        24 - hour_distance,
    )

    # ----------------------------------------------------------
    # ATO signals
    # ----------------------------------------------------------

    score = 0.0

    signals = []

    if device_changed:
        score += 25

        signals.append(
            {
                "signal": "Device changed",
                "weight": 25,
                "severity": "HIGH",
            }
        )

    if location_changed:
        score += 20

        signals.append(
            {
                "signal": "Location changed",
                "weight": 20,
                "severity": "HIGH",
            }
        )

    if hour < 6:
        score += 20

        signals.append(
            {
                "signal": "Unusual transaction hour",
                "weight": 20,
                "severity": "HIGH",
            }
        )
    elif hour_distance >= 6:
        score += 12

        signals.append(
            {
                "signal": "Transaction outside usual time pattern",
                "weight": 12,
                "severity": "MEDIUM",
            }
        )

    if recipient_new:
        score += 15

        signals.append(
            {
                "signal": "New recipient",
                "weight": 15,
                "severity": "MEDIUM",
            }
        )

    if amount_ratio >= 3:
        score += 10

        signals.append(
            {
                "signal": (
                    f"Amount is {amount_ratio:.1f}× "
                    "customer baseline"
                ),
                "weight": 10,
                "severity": "MEDIUM",
            }
        )

    if transactions_last_1h >= 3:
        score += 10

        signals.append(
            {
                "signal": "High transaction velocity",
                "weight": 10,
                "severity": "MEDIUM",
            }
        )

    score = min(
        100.0,
        score,
    )

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------

    if score >= 70:
        risk_level = "HIGH"
        action = (
            "Human verification recommended before "
            "any high-impact action."
        )
    elif score >= 45:
        risk_level = "REVIEW"
        action = (
            "Review device, location and behavioural "
            "context."
        )
    else:
        risk_level = "LOW"
        action = (
            "No strong account-takeover pattern detected."
        )

    return {
        "account_takeover_score": round(
            score,
            2,
        ),
        "risk_level": risk_level,
        "signals": signals,
        "recommended_action": action,
        "explanation": (
            "Account takeover intelligence combines "
            "device, location, time, recipient, amount "
            "and velocity changes. It supports human "
            "investigation and does not autonomously "
            "block the transaction."
        ),
    }