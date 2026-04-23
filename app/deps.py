import os
from fastapi import HTTPException, Request
from jose import jwt, JWTError

JWT_SECRET = os.environ.get("JWT_SECRET", "")
ALGORITHM = "HS256"


def verify_token(request: Request) -> dict:
    """Shared JWT auth with the Next.js prolife-site service."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth[7:]
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET not configured")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
