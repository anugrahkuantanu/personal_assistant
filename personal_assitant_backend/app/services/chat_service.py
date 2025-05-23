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
from app.agents.agent_factory import AgentFactory
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
        
        # Initialize Agent Factory
        self.agent_factory = AgentFactory(self.onecom_tools)
    
    async def process_message(self, message: str, client_id: str) -> Dict[str, Any]:
        """Process user message with AI agent capabilities"""
        try:
            # Determine which agent to use
            agent_type = await self._determine_agent_type(message)
            
            if agent_type:
                # Get context from other agents if needed
                context = await self._get_agent_context(agent_type)
                
                # Process with appropriate agent
                return await self.agent_factory.process_request(
                    message, agent_type, context
                )
            else:
                return await self._handle_regular_chat(message, client_id)
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Sorry, I encountered an error: {str(e)}",
                "type": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _determine_agent_type(self, message: str) -> str:
        """Determine which agent should handle the message"""
        system_prompt = """Analyze the user's message and determine which type of agent should handle it.
        Options:
        - email: For email-related tasks (reading, composing, sending emails)
        - schedule: For calendar and scheduling tasks
        - None: For general conversation
        
        Return just the agent type as a string, or None if no specific agent is needed."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            agent_type = response.choices[0].message.content.strip().lower()
            return agent_type if agent_type in ["email", "schedule"] else None
            
        except Exception:
            return None
    
    async def _get_agent_context(self, agent_type: str) -> Dict[str, Any]:
        """Get context from other agents if needed"""
        context = {}
        
        if agent_type == "email":
            # Get calendar data for email composition
            schedule_agent = self.agent_factory.get_agent("schedule")
            calendar_data = await schedule_agent._get_calendar_data(7)
            context["calendar_data"] = calendar_data
        
        return context
    
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