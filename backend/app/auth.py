from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer


# ============================================================
# JWT CONFIG
# ============================================================

JWT_SECRET_KEY = os.getenv(
    "UPAY_JWT_SECRET",
)

JWT_ALGORITHM = "HS256"

JWT_EXPIRE_MINUTES = int(
    os.getenv(
        "UPAY_JWT_EXPIRE_MINUTES",
        "60",
    )
)


if not JWT_SECRET_KEY:
    raise RuntimeError(
        "UPAY_JWT_SECRET environment variable is not set."
    )


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login"
)


# ============================================================
# DEMO ANALYST CREDENTIALS
# ============================================================

DEMO_USERNAME = os.getenv(
    "UPAY_ANALYST_USERNAME",
    "analyst",
)

DEMO_PASSWORD = os.getenv(
    "UPAY_ANALYST_PASSWORD",
    "sentinel-demo-2026",
)


# ============================================================
# CREATE JWT
# ============================================================

def create_access_token(
    username: str,
) -> str:

    expire = (
        datetime.now(
            timezone.utc
        )
        + timedelta(
            minutes=JWT_EXPIRE_MINUTES
        )
    )

    payload = {
        "sub": username,
        "role": "risk_analyst",
        "exp": expire,
    }

    token = jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )

    return token


# ============================================================
# VERIFY JWT
# ============================================================

def get_current_user(
    token: str = Depends(
        oauth2_scheme
    ),
) -> dict:

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={
            "WWW-Authenticate": "Bearer",
        },
    )

    try:

        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[
                JWT_ALGORITHM
            ],
        )

        username = payload.get(
            "sub"
        )

        role = payload.get(
            "role"
        )

        if not username:
            raise credentials_exception

        return {
            "username": username,
            "role": role or "risk_analyst",
        }

    except jwt.ExpiredSignatureError:
        raise credentials_exception

    except jwt.InvalidTokenError:
        raise credentials_exception