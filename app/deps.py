import os
from fastapi import HTTPException, Header
from jose import jwt, JWTError

JWT_SECRET = os.environ.get("JWT_SECRET", "")
ALGORITHM = "HS256"


def verify_token(authorization: str = Header(...)) -> dict:
    """Shared JWT auth with the Next.js prolife-site service."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ")
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET not configured")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
