from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ChatMessage(BaseModel):
    message: str
    timestamp: datetime = datetime.now()
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    timestamp: datetime = datetime.now()
