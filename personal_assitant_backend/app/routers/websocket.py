"""
WebSocket Router - Handles WebSocket connections and message routing
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.chat_service import chat_service, connection_manager
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

@router.websocket("/ws/chat/{client_id}")
async def chat_websocket(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for chat interactions"""
    await connection_manager.connect(websocket, client_id)
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to chat service. You can now send messages.",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            # Process message through chat service
            response = await chat_service.process_message(data, client_id)
            
            # Send response back to client
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        })
        connection_manager.disconnect(client_id)

@router.websocket("/ws/agent/{client_id}")
async def agent_websocket(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for structured agent commands"""
    await connection_manager.connect(websocket, client_id)
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to agent service. You can now send structured commands.",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                # Parse command
                command = json.loads(data)
                response = await _handle_structured_command(command, client_id)
                
            except json.JSONDecodeError:
                response = {
                    "type": "error",
                    "message": "Invalid JSON command",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Send response back to client
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(client_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        })
        connection_manager.disconnect(client_id)

async def _handle_structured_command(command: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """Handle structured commands for the agent"""
    action = command.get("action")
    
    if action == "test_connection":
        return {
            "type": "success",
            "message": "Connection successful",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    elif action == "get_summary":
        # Get calendar events
        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            executor,
            onecom_tools.calendar_tool.get_events,
            7  # Next 7 days
        )
        
        # Get recent emails
        emails = await loop.run_in_executor(
            executor,
            onecom_tools.email_tool.read_emails,
            10  # Last 10 emails
        )
        
        return {
            "type": "summary",
            "events": events,
            "emails": emails,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    elif action == "get_emails":
        limit = command.get("limit", 10)
        loop = asyncio.get_event_loop()
        emails = await loop.run_in_executor(
            executor,
            onecom_tools.email_tool.read_emails,
            limit
        )
        
        return {
            "type": "emails",
            "emails": emails,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    elif action == "get_events":
        days = command.get("days", 7)
        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            executor,
            onecom_tools.calendar_tool.get_events,
            days
        )
        
        return {
            "type": "events",
            "events": events,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    elif action == "send_email":
        recipient = command.get("recipient")
        subject = command.get("subject")
        body = command.get("body")
        
        if not all([recipient, subject, body]):
            return {
                "type": "error",
                "message": "Missing required email fields",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            executor,
            onecom_tools.email_tool.send_email,
            recipient,
            subject,
            body
        )
        
        return {
            "type": "email_sent" if success else "error",
            "message": "Email sent successfully" if success else "Failed to send email",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    else:
        return {
            "type": "error",
            "message": f"Unknown action: {action}",
            "timestamp": datetime.utcnow().isoformat()
        }

# Export for use in other modules
__all__ = ['router']

# Enhanced chat service that can work with OneComAI Agent
