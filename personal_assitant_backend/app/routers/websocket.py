"""
WebSocket Router - Handles WebSocket connections and message routing
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
# from app.services.chat_service import chat_service, connection_manager
from app.services.chat_service import enhanced_chat_service, connection_manager
from app.tools.onecom_tools import OneComTools
from app.core.config import settings
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)

# Initialize OneCom tools with credentials from settings
onecom_tools = OneComTools(
    email_address=settings.EMAIL_ADDRESS,
    password=settings.EMAIL_PASSWORD
)

# @router.websocket("/ws/chat/{client_id}")
# async def chat_websocket(websocket: WebSocket, client_id: str):
#     """WebSocket endpoint for chat interactions"""
#     await connection_manager.connect(websocket, client_id)
#     try:
#         # Send welcome message
#         await websocket.send_json({
#             "type": "welcome",
#             "message": "Connected to chat service. You can now send messages.",
#             "timestamp": datetime.utcnow().isoformat()
#         })
        
#         while True:
#             # Receive message
#             data = await websocket.receive_text()
            
#             # Process message through chat service
#             response = await chat_service.process_message(data, client_id)
            
#             # Send response back to client
#             await websocket.send_json(response)
            
#     except WebSocketDisconnect:
#         connection_manager.disconnect(client_id)
#     except Exception as e:
#         await websocket.send_json({
#             "type": "error",
#             "message": f"Error: {str(e)}",
#             "timestamp": datetime.utcnow().isoformat()
#         })
#         connection_manager.disconnect(client_id)

@router.websocket("/ws/chat/{client_id}")
async def enhanced_chat_websocket(websocket: WebSocket, client_id: str):
    """Enhanced WebSocket endpoint with workflow support"""
    await connection_manager.connect(websocket, client_id)
    
    try:
        # Send enhanced welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "🤖 Enhanced AI Assistant Connected! I can autonomously handle complex meeting scheduling, email management, and calendar coordination.",
            "capabilities": [
                "Autonomous meeting scheduling workflows",
                "Multi-step email coordination", 
                "Intelligent calendar management",
                "Context-aware conversation"
            ],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                # Try to parse as JSON for workflow commands
                command_data = json.loads(data)
                if "command" in command_data:
                    # Handle workflow command
                    print(f"Received command: {command_data['command']}")
                    response = await enhanced_chat_service.handle_workflow_command(command_data, client_id)
                else:
                    # Process as regular message
                    response = await enhanced_chat_service.process_message(data, client_id)
            except json.JSONDecodeError:
                # Process as regular text message
                response = await enhanced_chat_service.process_message(data, client_id)
            
            # Send response
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Service error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        })
        connection_manager.disconnect(client_id)
# Export for use in other modules
__all__ = ['router']

# Enhanced chat service that can work with OneComAI Agent
