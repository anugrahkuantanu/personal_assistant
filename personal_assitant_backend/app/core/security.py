from fastapi import WebSocket, HTTPException
from typing import Optional

async def verify_token(websocket: WebSocket, token: Optional[str] = None) -> bool:
    if not token:
        return False
    # Add your token verification logic here
    # For demo purposes, we'll accept any token
    return True
