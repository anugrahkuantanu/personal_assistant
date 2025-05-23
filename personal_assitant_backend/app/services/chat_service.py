"""
Enhanced Chat Service with AI Integration
Handles chat processing and AI agent integration
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any
from fastapi import WebSocket
from app.core.config import settings
from app.tools.onecom_tools import OneComTools
from app.services.ai_agent_service import AIAgentService
from openai import AsyncOpenAI

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: Any, client_id: str):
        if client_id in self.active_connections:
            if isinstance(message, dict):
                await self.active_connections[client_id].send_json(message)
            else:
                await self.active_connections[client_id].send_json({
                    "message": str(message),
                    "timestamp": datetime.utcnow().isoformat()
                })

class EnhancedChatService:
    """Enhanced chat service with AI agent integration"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.conversation_history: Dict[str, list] = {}
        
        # Initialize OneCom tools
        self.onecom_tools = OneComTools(
            email_address=settings.EMAIL_ADDRESS,
            password=settings.EMAIL_PASSWORD
        )
        
        # Initialize AI Agent Service
        self.ai_agent_service = AIAgentService(self.onecom_tools)
    
    async def process_message(self, message: str, client_id: str) -> Dict[str, Any]:
        """Process user message with AI agent capabilities"""
        try:
            # Check if this is an agent-related request
            if self._is_agent_request(message):
                return await self._handle_agent_request(message, client_id)
            else:
                return await self._handle_regular_chat(message, client_id)
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Sorry, I encountered an error: {str(e)}",
                "type": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _is_agent_request(self, message: str) -> bool:
        """Determine if message requires agent functionality"""
        agent_keywords = [
            'calendar', 'schedule', 'meeting', 'appointment', 'events',
            'email', 'emails', 'send email', 'write email', 'compose',
            'plan', 'summary', 'today', 'tomorrow', 'next week',
            'free time', 'available', 'busy', 'check my'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in agent_keywords)
    
    async def _handle_agent_request(self, message: str, client_id: str) -> Dict[str, Any]:
        """Handle agent-powered requests"""
        
        # Use AI Agent Service to process the request
        result = await self.ai_agent_service.process_user_request(message, client_id)
        
        if result["success"]:
            response = result["response"]
            
            # Format response for WebSocket
            return {
                "success": True,
                "message": response.get("message", "Request processed successfully"),
                "type": response.get("type", "agent_response"),
                "data": {
                    "intent": result.get("intent", {}),
                    "events_count": response.get("events_count", 0),
                    "emails_count": response.get("emails_analyzed", 0),
                    "action_result": response.get("action_result"),
                    "email_content": response.get("email_content")
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "success": False,
                "message": f"I couldn't process your request: {result.get('error', 'Unknown error')}",
                "type": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _handle_regular_chat(self, message: str, client_id: str) -> Dict[str, Any]:
        """Handle regular chat without agent functionality"""
        
        # Initialize conversation history
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        
        # Add user message to history
        self.conversation_history[client_id].append({"role": "user", "content": message})
        
        system_prompt = """You are a helpful personal assistant. You have access to the user's email and calendar through special commands, but for general conversation, just chat normally.

If the user asks about calendar, emails, scheduling, or planning, let them know you can help with that using phrases like:
- "I can check your calendar for that"
- "I can analyze your emails" 
- "I can help you compose and send emails"
- "I can look at your schedule"

For other topics, have a normal helpful conversation."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.conversation_history[client_id]
                ],
                temperature=0.7,
                max_tokens=300,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history[client_id].append({"role": "assistant", "content": assistant_message})
            
            # Keep conversation history manageable
            if len(self.conversation_history[client_id]) > 20:
                self.conversation_history[client_id] = self.conversation_history[client_id][-20:]
            
            return {
                "success": True,
                "message": assistant_message,
                "type": "chat_response",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Sorry, I couldn't process your message: {str(e)}",
                "type": "error",
                "timestamp": datetime.utcnow().isoformat()
            }

# Global instances
connection_manager = ConnectionManager()
chat_service = EnhancedChatService()

# Export for use in other modules
__all__ = ['chat_service', 'connection_manager', 'EnhancedChatService']