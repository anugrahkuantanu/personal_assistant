import os
# Set tokenizers parallelism before importing any other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import websocket
from app.db.database import engine, Base
# from app.db.models import *
from app.services import chat_service  # This will import all models

# Create database tables
# Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.APP_NAME)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(websocket.router)

# @app.on_event("shutdown")
# async def shutdown_event():
#     await chat_service.cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,  # Enable auto-reload
        workers=1     # Use single worker for development
    )
