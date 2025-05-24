"""
Factory Agent with Conversation Context - Main Orchestrator
File: app/agents/factory_agent.py

Handles intelligent routing with conversation memory and context tracking
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
from app.core.config import settings
from app.tools.onecom_tools import OneComTools

class ConversationManager:
    """Manages conversation history and context for each client"""
    
    def __init__(self):
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.user_context: Dict[str, Dict] = {}  # Store extracted user info
        self.session_metadata: Dict[str, Dict] = {}
    
    def add_message(self, client_id: str, role: str, content: str, message_type: str = "chat"):
        """Add message to conversation history"""
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "type": message_type
        }
        
        self.conversation_history[client_id].append(message)
        
        # Keep only last 50 messages to prevent memory issues
        if len(self.conversation_history[client_id]) > 50:
            self.conversation_history[client_id] = self.conversation_history[client_id][-50:]
    
    def get_conversation_history(self, client_id: str, last_n: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        history = self.conversation_history.get(client_id, [])
        return history[-last_n:] if history else []
    
    def update_user_context(self, client_id: str, context_updates: Dict):
        """Update user context with new information"""
        if client_id not in self.user_context:
            self.user_context[client_id] = {
                "preferred_emails": [],
                "meeting_preferences": {},
                "common_locations": [],
                "timezone": None,
                "work_hours": {"start": 9, "end": 17},
                "last_updated": datetime.now().isoformat()
            }
        
        # Merge new context
        for key, value in context_updates.items():
            if key == "preferred_emails" and value:
                # Add new emails to list, avoid duplicates
                existing = self.user_context[client_id].get("preferred_emails", [])
                for email in value if isinstance(value, list) else [value]:
                    if email not in existing:
                        existing.append(email)
                self.user_context[client_id]["preferred_emails"] = existing
            elif key == "common_locations" and value:
                # Add new locations
                existing = self.user_context[client_id].get("common_locations", [])
                for location in value if isinstance(value, list) else [value]:
                    if location not in existing:
                        existing.append(location)
                self.user_context[client_id]["common_locations"] = existing
            else:
                self.user_context[client_id][key] = value
        
        self.user_context[client_id]["last_updated"] = datetime.now().isoformat()
    
    def get_user_context(self, client_id: str) -> Dict:
        """Get user context"""
        return self.user_context.get(client_id, {})
    
    def extract_context_from_message(self, message: str) -> Dict:
        """Extract potential context information from message"""
        context = {}
        message_lower = message.lower()
        
        # Extract email addresses
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        if emails:
            context["preferred_emails"] = emails
        
        # Extract time preferences
        time_patterns = [
            r'\d{1,2}:\d{2}\s*(am|pm)',
            r'\d{1,2}\s*(am|pm)',
            r'morning|afternoon|evening'
        ]
        for pattern in time_patterns:
            if re.search(pattern, message_lower):
                context["meeting_preferences"] = {"time_mentioned": True}
                break
        
        # Extract location mentions
        location_keywords = ['office', 'zoom', 'teams', 'meeting room', 'conference']
        for keyword in location_keywords:
            if keyword in message_lower:
                context["common_locations"] = [keyword]
                break
        
        return context

class AgentFactory:
    """
    Agent Factory with Conversation Memory
    
    Routes requests to appropriate agents while maintaining conversation context
    """
    
    def __init__(self, onecom_tools: OneComTools):
        self.onecom_tools = onecom_tools
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize agents (lazy loading)
        self._scheduling_agent = None
        self._email_agent = None
        
        # Conversation and context management
        self.conversation_manager = ConversationManager()
        self.active_sessions: Dict[str, Dict] = {}
    
    @property
    def scheduling_agent(self):
        """Lazy load scheduling agent"""
        if self._scheduling_agent is None:
            from app.agents.scheduling_agent import SchedulingAgent
            self._scheduling_agent = SchedulingAgent(self.onecom_tools)
        return self._scheduling_agent
    
    @property
    def email_agent(self):
        """Lazy load email agent"""
        if self._email_agent is None:
            from app.agents.email_agent import EmailAgent
            self._email_agent = EmailAgent(self.onecom_tools)
        return self._email_agent
    
    def _extract_message_content(self, user_message: str) -> str:
        """Extract clean message from JSON wrapper if needed"""
        try:
            if user_message.strip().startswith('{') and user_message.strip().endswith('}'):
                parsed = json.loads(user_message)
                if isinstance(parsed, dict):
                    for field in ['message', 'text', 'content', 'query']:
                        if field in parsed:
                            return str(parsed[field])
                return str(parsed)
            return user_message
        except (json.JSONDecodeError, TypeError):
            return user_message
    
    async def process_request(self, user_message: str, client_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point with conversation context and memory
        """
        try:
            # Extract clean message
            clean_message = self._extract_message_content(user_message)
            print(f"AgentFactory processing: {clean_message}")
            
            # Add user message to conversation history
            self.conversation_manager.add_message(client_id, "user", clean_message)
            
            # Extract and update user context from message
            message_context = self.conversation_manager.extract_context_from_message(clean_message)
            if message_context:
                self.conversation_manager.update_user_context(client_id, message_context)
            
            # Get conversation history and user context
            conversation_history = self.conversation_manager.get_conversation_history(client_id, 10)
            user_context = self.conversation_manager.get_user_context(client_id)
            
            # Analyze which agent should handle this with context
            routing_decision = await self._analyze_routing_with_context(
                clean_message, client_id, conversation_history, user_context
            )
            print(f"Routing decision: {routing_decision}")
            
            # Prepare enhanced context for agents
            enhanced_context = context or {}
            enhanced_context.update({
                "conversation_history": conversation_history,
                "user_context": user_context,
                "routing_decision": routing_decision
            })
            
            # Route to appropriate agent
            if routing_decision["agent"] == "scheduling":
                result = await self._route_to_scheduling(clean_message, client_id, routing_decision, enhanced_context)
            elif routing_decision["agent"] == "email":
                result = await self._route_to_email(clean_message, client_id, routing_decision, enhanced_context)
            else:
                result = await self._handle_general_chat(clean_message, client_id, conversation_history)
            
            # Add assistant response to conversation history
            assistant_message = result.get("message", "")
            if assistant_message:
                self.conversation_manager.add_message(
                    client_id, "assistant", assistant_message, result.get("type", "response")
                )
            
            return result
                
        except Exception as e:
            print(f"Error in AgentFactory: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            error_message = f"Error processing request: {str(e)}"
            self.conversation_manager.add_message(client_id, "assistant", error_message, "error")
            
            return {
                "success": False,
                "message": error_message,
                "type": "error"
            }
    
    async def _analyze_routing_with_context(self, message: str, client_id: str, 
                                          conversation_history: List[Dict], user_context: Dict) -> Dict[str, Any]:
        """
        Intelligent routing analysis with conversation context
        """
        
        # Format conversation history for LLM
        history_text = self._format_conversation_for_llm(conversation_history)
        
        system_prompt = """You are an AI request router with conversation memory. Analyze the user's message along with conversation context to determine which agent should handle it.

        AVAILABLE AGENTS:
        1. "scheduling" - ALL calendar and scheduling tasks:
        - View calendar, find free time, schedule meetings, create events
        - ANY task involving time, calendar, meetings, appointments

        2. "email" - ALL email tasks:
        - Read emails, compose emails, send emails
        - ANY task involving email communication

        3. "general" - Everything else:
        - General questions, casual conversation, help requests

        CONVERSATION CONTEXT:
        Recent conversation history:
        {history}

        User context (remembered information):
        {user_context}

        Current message: "{message}"

        IMPORTANT CONTEXT CONSIDERATIONS:
        - If the user previously mentioned email addresses, meeting times, or preferences, they are available in user_context
        - If this continues a previous scheduling or email conversation, consider that context
        - Look for follow-up responses like "yes", "that works", "send it", "schedule it"
        - If user says "schedule it" or "send it" after discussing meeting/email, route accordingly

        Return JSON:
        {{
            "agent": "scheduling|email|general",
            "confidence": 0.0-1.0,
            "reasoning": "why you chose this agent considering conversation context",
            "task_type": "specific task description",
            "continues_previous": true/false,
            "context_used": ["what context information was helpful"]
        }}

        Examples with context:
        - Previous: User mentioned "john@company.com", Current: "schedule the meeting" → scheduling
        - Previous: Discussed meeting times, Current: "send the invitation" → scheduling  
        - Previous: Talked about project email, Current: "send it now" → email"""

        try:
            prompt = system_prompt.format(
                history=history_text,
                user_context=json.dumps(user_context, indent=2),
                message=message
            )
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Analyze this request with conversation context."}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean and parse JSON
            if not response_content.startswith('{'):
                start_idx = response_content.find('{')
                if start_idx != -1:
                    response_content = response_content[start_idx:]
            
            if not response_content.endswith('}'):
                end_idx = response_content.rfind('}')
                if end_idx != -1:
                    response_content = response_content[:end_idx + 1]
            
            routing_decision = json.loads(response_content)
            
            # Update session context
            self._update_session_context(client_id, message, routing_decision)
            
            return routing_decision
            
        except Exception as e:
            print(f"Error in routing analysis: {e}")
            return self._fallback_routing(message)
    
    def _format_conversation_for_llm(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for LLM analysis"""
        if not conversation_history:
            return "No previous conversation."
        
        formatted_messages = []
        for msg in conversation_history[-8:]:  # Last 8 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Truncate long messages
            timestamp = msg.get("timestamp", "")
            
            formatted_messages.append(f"{role.title()}: {content}")
        
        return "\n".join(formatted_messages)
    
    async def _route_to_scheduling(self, message: str, client_id: str, routing_decision: Dict, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route to scheduling agent with full context"""
        try:
            result = await self.scheduling_agent.process_request(message, context)
            result["routed_to"] = "scheduling_agent"
            
            # Update user context based on scheduling result
            if result.get("success") and result.get("type") in ["workflow_started", "proposal_sent"]:
                context_updates = {}
                if "workflow_id" in result:
                    context_updates["last_workflow_id"] = result["workflow_id"]
                if result.get("recipient"):
                    context_updates["preferred_emails"] = [result["recipient"]]
                
                if context_updates:
                    self.conversation_manager.update_user_context(client_id, context_updates)
            
            return result
            
        except Exception as e:
            print(f"Error routing to scheduling agent: {e}")
            return {
                "success": False,
                "message": f"Error in scheduling: {str(e)}",
                "type": "error"
            }
    
    async def _route_to_email(self, message: str, client_id: str, routing_decision: Dict, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route to email agent with full context"""
        try:
            result = await self.email_agent.process_request(message, context)
            result["routed_to"] = "email_agent"
            
            # Update user context based on email result
            if result.get("success") and result.get("type") == "email_sent":
                context_updates = {}
                if result.get("recipient"):
                    context_updates["preferred_emails"] = [result["recipient"]]
                
                if context_updates:
                    self.conversation_manager.update_user_context(client_id, context_updates)
            
            return result
            
        except Exception as e:
            print(f"Error routing to email agent: {e}")
            return {
                "success": False,
                "message": f"Error in email processing: {str(e)}",
                "type": "error"
            }
    
    async def _handle_general_chat(self, message: str, client_id: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Handle general chat with conversation context"""
        
        # Format recent conversation for context
        history_text = self._format_conversation_for_llm(conversation_history)
        
        system_prompt = """You are a helpful personal assistant with conversation memory. Provide natural, helpful responses that reference previous conversation when appropriate.

        You have access to:
        - Scheduling capabilities (calendar, meetings, appointments)
        - Email management (read, compose, send emails)

        Keep responses conversational and reference previous context when relevant.

        Recent conversation:
        {history}

        Respond naturally while being helpful about your capabilities."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt.format(history=history_text)},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return {
                "success": True,
                "message": response.choices[0].message.content,
                "type": "general_chat",
                "routed_to": "general_chat"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error in general chat: {str(e)}",
                "type": "error"
            }
    
    def _update_session_context(self, client_id: str, message: str, routing_decision: Dict):
        """Track session context"""
        if client_id not in self.active_sessions:
            self.active_sessions[client_id] = {
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "interaction_count": 0
            }
        
        session = self.active_sessions[client_id]
        session["last_activity"] = datetime.now().isoformat()
        session["interaction_count"] += 1
        session["last_agent"] = routing_decision["agent"]
        session["last_confidence"] = routing_decision.get("confidence", 0.0)
    
    def _fallback_routing(self, message: str) -> Dict[str, Any]:
        """Simple keyword-based fallback"""
        message_lower = message.lower()
        
        schedule_keywords = ["schedule", "meeting", "calendar", "appointment", "time", "when", "free", "busy"]
        email_keywords = ["email", "send", "compose", "write", "message", "mail"]
        
        if any(keyword in message_lower for keyword in schedule_keywords):
            agent = "scheduling"
        elif any(keyword in message_lower for keyword in email_keywords):
            agent = "email"
        else:
            agent = "general"
        
        return {
            "agent": agent,
            "confidence": 0.7,
            "reasoning": f"Fallback routing based on keywords",
            "task_type": f"{agent} task",
            "continues_previous": False,
            "context_used": []
        }
    
    def get_session_status(self, client_id: str) -> Dict[str, Any]:
        """Get comprehensive session status"""
        return {
            "session": self.active_sessions.get(client_id, {}),
            "conversation_history": self.conversation_manager.get_conversation_history(client_id),
            "user_context": self.conversation_manager.get_user_context(client_id),
            "scheduling_workflows": self.scheduling_agent.get_active_workflows() if self._scheduling_agent else []
        }
    
    def cleanup_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions and conversation history"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up active sessions
        to_remove = []
        for client_id, session in self.active_sessions.items():
            last_activity = datetime.fromisoformat(session["last_activity"])
            if last_activity < cutoff:
                to_remove.append(client_id)
        
        for client_id in to_remove:
            del self.active_sessions[client_id]
            # Also clean up conversation history
            if client_id in self.conversation_manager.conversation_history:
                del self.conversation_manager.conversation_history[client_id]
            if client_id in self.conversation_manager.user_context:
                del self.conversation_manager.user_context[client_id]
        
        # Clean up agent workflows
        if self._scheduling_agent:
            self.scheduling_agent.cleanup_completed_workflows(max_age_hours)
            
        if to_remove:
            print(f"Cleaned up {len(to_remove)} old sessions with conversation history")