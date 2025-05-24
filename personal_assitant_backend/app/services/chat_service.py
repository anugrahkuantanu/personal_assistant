"""
Enhanced Chat Service with Conversation Context Integration
File: app/services/enhanced_chat_service.py

Provides intelligent message processing with conversation memory and context tracking
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import WebSocket
from app.core.config import settings
from app.tools.onecom_tools import OneComTools
from app.agents.factory_agent import AgentFactory  # Updated to use context-aware factory
from openai import AsyncOpenAI

class ConnectionManager:
    """Enhanced connection manager with session tracking"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_metadata: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = {
            "connected_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "message_count": 0
        }

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]

    async def send_personal_message(self, message: Any, client_id: str):
        if client_id in self.active_connections:
            try:
                if isinstance(message, dict):
                    await self.active_connections[client_id].send_json(message)
                else:
                    await self.active_connections[client_id].send_json({
                        "message": str(message),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Update activity tracking
                if client_id in self.client_metadata:
                    self.client_metadata[client_id]["last_activity"] = datetime.utcnow().isoformat()
                    self.client_metadata[client_id]["message_count"] += 1
                    
            except Exception as e:
                print(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast_status_update(self, status: Dict[str, Any]):
        """Broadcast status updates to all connected clients"""
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json({
                    "type": "status_update",
                    "status": status,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception:
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

class EnhancedChatService:
    """
    Enhanced chat service with conversation memory and context tracking
    
    Features:
    - Full conversation history and context memory
    - Intelligent agent routing with context awareness
    - User preference learning and retention
    - Workflow notifications and progress tracking
    - Performance monitoring and analytics
    """
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize OneCom tools
        self.onecom_tools = OneComTools(
            email_address=settings.EMAIL_ADDRESS,
            password=settings.EMAIL_PASSWORD
        )
        
        # Initialize Context-Aware Agent Factory
        self.agent_factory = AgentFactory(self.onecom_tools)
        
        # Enhanced service statistics
        self.service_stats = {
            "total_requests": 0,
            "successful_workflows": 0,
            "active_meetings_scheduled": 0,
            "emails_sent": 0,
            "context_hits": 0,  # Times context was helpful
            "follow_up_requests": 0,  # Requests that used previous context
            "average_response_time": 0.0,
            "service_started": datetime.utcnow().isoformat()
        }
    
    async def process_message(self, message: str, client_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced message processing with conversation context and memory
        """
        start_time = datetime.utcnow()
        
        try:
            self.service_stats["total_requests"] += 1
            
            # Enhanced context with client information
            enhanced_context = context or {}
            enhanced_context.update({
                "client_metadata": self.connection_manager.client_metadata.get(client_id, {}),
                "service_context": {
                    "processing_start": start_time.isoformat(),
                    "request_id": f"{client_id}_{self.service_stats['total_requests']}"
                }
            })
            
            # Process through context-aware agent factory
            result = await self.agent_factory.process_request(message, client_id, enhanced_context)
            
            # Update context-related statistics
            await self._update_context_metrics(result, client_id)
            
            # Track success metrics
            await self._update_success_metrics(result)
            
            # Update response time statistics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_response_time_stats(processing_time)
            
            # Add processing metadata to response
            result.update({
                "processing_time_ms": processing_time * 1000,
                "request_id": enhanced_context["service_context"]["request_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "context_aware": True  # Indicates this service has context memory
            })
            
            # Handle special workflow responses with context awareness
            await self._handle_workflow_notifications_with_context(result, client_id)
            
            return result
            
        except Exception as e:
            error_response = {
                "success": False,
                "message": f"Service error: {str(e)}",
                "type": "service_error",
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "context_aware": True
            }
            
            # Log error for monitoring
            await self._log_error(e, client_id, message, error_response)
            
            return error_response
    
    async def _update_context_metrics(self, result: Dict[str, Any], client_id: str):
        """Update metrics related to context usage"""
        
        # Check if context was used in routing
        routing_info = result.get("routing_decision", {})
        if routing_info.get("continues_previous"):
            self.service_stats["follow_up_requests"] += 1
        
        context_used = routing_info.get("context_used", [])
        if context_used:
            self.service_stats["context_hits"] += 1
        
        # Check if this was a successful context-aware interaction
        if (result.get("success") and 
            result.get("type") in ["proposal_sent", "email_sent", "event_created"] and
            routing_info.get("confidence", 0) > 0.8):
            
            # High confidence successful action - context was likely helpful
            self.service_stats["context_hits"] += 1
    
    async def _update_success_metrics(self, result: Dict[str, Any]):
        """Update success metrics based on result"""
        if result.get("success"):
            result_type = result.get("type", "")
            
            # Track successful workflows
            if result_type in ["workflow_started", "proposal_sent", "event_created"]:
                self.service_stats["successful_workflows"] += 1
            
            # Track meetings scheduled
            if result_type in ["proposal_sent", "meeting_confirmed"]:
                self.service_stats["active_meetings_scheduled"] += 1
            
            # Track emails sent
            if result_type == "email_sent":
                self.service_stats["emails_sent"] += 1
    
    async def _handle_workflow_notifications_with_context(self, result: Dict[str, Any], client_id: str):
        """Handle workflow notifications with context awareness"""
        
        result_type = result.get("type", "")
        
        # Context-aware workflow progress updates
        if result_type == "workflow_started":
            context_indicator = ""
            if result.get("missing_info"):
                context_indicator = " I'll remember what you tell me for future requests."
            
            await self.connection_manager.send_personal_message({
                "type": "workflow_progress",
                "stage": "started",
                "message": f"🔄 Starting workflow...{context_indicator}",
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
        
        # Enhanced meeting proposal notifications
        elif result_type == "proposal_sent":
            context_note = ""
            if result.get("routed_to") == "scheduling_agent":
                # Check if context was used
                user_context = self.agent_factory.conversation_manager.get_user_context(client_id)
                if user_context.get("preferred_emails"):
                    context_note = " (using your preferred contact from our conversation history)"
            
            await self.connection_manager.send_personal_message({
                "type": "meeting_proposal_sent",
                "recipient": result.get("recipient"),
                "proposed_slots": result.get("proposed_slots", []),
                "message": f"📧 Meeting proposal sent successfully{context_note}!",
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
        
        # Enhanced email sent notifications
        elif result_type == "email_sent":
            context_note = ""
            if result.get("routed_to") == "email_agent":
                # Check if this was a follow-up
                routing_decision = result.get("routing_decision", {})
                if routing_decision.get("continues_previous"):
                    context_note = " (continuing our previous conversation)"
            
            await self.connection_manager.send_personal_message({
                "type": "email_sent_notification",
                "recipient": result.get("recipient"),
                "subject": result.get("subject"),
                "message": f"✅ Email sent successfully{context_note}!",
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
        
        # Enhanced event creation notifications
        elif result_type == "event_created":
            await self.connection_manager.send_personal_message({
                "type": "event_created_notification",
                "event_details": result.get("event_details", {}),
                "message": "📅 Calendar event created successfully!",
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
        
        # Context-aware info requests
        elif result_type == "info_request":
            context_indicator = ""
            user_context = self.agent_factory.conversation_manager.get_user_context(client_id)
            if user_context:
                context_indicator = " I'll use this information to help with future requests too."
            
            await self.connection_manager.send_personal_message({
                "type": "info_request_notification",
                "missing_info": result.get("missing_info", []),
                "message": f"📝 Need additional information{context_indicator}",
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
    
    async def get_client_status(self, client_id: str) -> Dict[str, Any]:
        """Get comprehensive status with conversation context"""
        
        # Get session status from context-aware factory
        session_status = self.agent_factory.get_session_status(client_id)
        
        # Get connection metadata
        connection_metadata = self.connection_manager.client_metadata.get(client_id, {})
        
        # Get conversation summary
        conversation_history = self.agent_factory.conversation_manager.get_conversation_history(client_id)
        user_context = self.agent_factory.conversation_manager.get_user_context(client_id)
        
        return {
            "client_id": client_id,
            "connection_status": "connected" if client_id in self.connection_manager.active_connections else "disconnected",
            "connection_metadata": connection_metadata,
            "session_status": session_status,
            "conversation_summary": {
                "total_messages": len(conversation_history),
                "recent_messages": len([m for m in conversation_history if m.get("timestamp", "2000-01-01") > (datetime.now() - timedelta(hours=1)).isoformat()]),
                "user_context_items": len(user_context),
                "has_email_preferences": bool(user_context.get("preferred_emails")),
                "has_meeting_preferences": bool(user_context.get("meeting_preferences"))
            },
            "service_stats": self.service_stats
        }
    
    async def handle_workflow_command(self, command: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle workflow commands with context awareness"""
        
        command_type = command.get("command")
        
        if command_type == "get_conversation_history":
            # Get conversation history
            conversation_history = self.agent_factory.conversation_manager.get_conversation_history(client_id, 20)
            return {
                "success": True,
                "conversation_history": conversation_history,
                "type": "conversation_history_response"
            }
        
        elif command_type == "get_user_context":
            # Get user context and preferences
            user_context = self.agent_factory.conversation_manager.get_user_context(client_id)
            return {
                "success": True,
                "user_context": user_context,
                "type": "user_context_response"
            }
        
        elif command_type == "clear_context":
            # Clear user context but keep conversation history
            if client_id in self.agent_factory.conversation_manager.user_context:
                del self.agent_factory.conversation_manager.user_context[client_id]
            return {
                "success": True,
                "message": "User context cleared. Conversation history preserved.",
                "type": "context_cleared"
            }
        
        elif command_type == "get_session_status":
            # Get comprehensive session status
            client_status = await self.get_client_status(client_id)
            return {
                "success": True,
                "session_status": client_status,
                "type": "session_status_response"
            }
        
        elif command_type == "get_active_workflows":
            # Get active workflows from scheduling agent
            try:
                scheduling_agent = self.agent_factory.scheduling_agent
                active_workflows = scheduling_agent.get_active_workflows()
                return {
                    "success": True,
                    "active_workflows": active_workflows,
                    "type": "active_workflows_response"
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error getting workflows: {str(e)}",
                    "type": "error"
                }
        
        elif command_type == "cancel_workflow":
            workflow_id = command.get("workflow_id")
            if workflow_id:
                try:
                    scheduling_agent = self.agent_factory.scheduling_agent
                    if workflow_id in scheduling_agent.active_workflows:
                        del scheduling_agent.active_workflows[workflow_id]
                        return {
                            "success": True,
                            "message": f"Workflow {workflow_id} cancelled",
                            "type": "workflow_cancelled"
                        }
                    else:
                        return {
                            "success": False,
                            "message": "Workflow not found",
                            "type": "error"
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "message": f"Error cancelling workflow: {str(e)}",
                        "type": "error"
                    }
        
        elif command_type == "reset_session":
            # Reset entire session including conversation history and context
            try:
                if client_id in self.agent_factory.active_sessions:
                    del self.agent_factory.active_sessions[client_id]
                if client_id in self.agent_factory.conversation_manager.conversation_history:
                    del self.agent_factory.conversation_manager.conversation_history[client_id]
                if client_id in self.agent_factory.conversation_manager.user_context:
                    del self.agent_factory.conversation_manager.user_context[client_id]
                
                return {
                    "success": True,
                    "message": "Complete session reset - conversation history and context cleared",
                    "type": "session_reset"
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error resetting session: {str(e)}",
                    "type": "error"
                }
        
        elif command_type == "system_health":
            # Enhanced system health with context metrics
            return {
                "success": True,
                "system_health": {
                    "service_stats": self.service_stats,
                    "active_connections": len(self.connection_manager.active_connections),
                    "total_active_sessions": len(self.agent_factory.active_sessions),
                    "total_conversation_histories": len(self.agent_factory.conversation_manager.conversation_history),
                    "total_user_contexts": len(self.agent_factory.conversation_manager.user_context),
                    "context_hit_rate": (self.service_stats["context_hits"] / max(self.service_stats["total_requests"], 1)) * 100,
                    "follow_up_rate": (self.service_stats["follow_up_requests"] / max(self.service_stats["total_requests"], 1)) * 100,
                    "onecom_status": await self._check_onecom_status()
                },
                "type": "system_health_response"
            }
        
        else:
            return {
                "success": False,
                "message": f"Unknown command: {command_type}",
                "type": "error"
            }
    
    async def _check_onecom_status(self) -> Dict[str, bool]:
        """Check OneComTools connectivity status"""
        try:
            # Test email connectivity with timeout
            email_test = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.onecom_tools.email_tool.read_emails, 1
                ), timeout=10
            )
            email_status = bool(email_test)
            
            # Test calendar connectivity with timeout
            calendar_test = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.onecom_tools.calendar_tool.get_events, 1
                ), timeout=10
            )
            calendar_status = bool(calendar_test)
            
            return {
                "email_connected": email_status,
                "calendar_connected": calendar_status,
                "overall_healthy": email_status and calendar_status
            }
        except Exception as e:
            print(f"OneComTools health check failed: {e}")
            return {
                "email_connected": False,
                "calendar_connected": False,
                "overall_healthy": False
            }
    
    def _update_response_time_stats(self, processing_time: float):
        """Update rolling average response time"""
        current_avg = self.service_stats["average_response_time"]
        total_requests = self.service_stats["total_requests"]
        
        # Calculate new rolling average
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.service_stats["average_response_time"] = round(new_avg, 3)
    
    async def _log_error(self, error: Exception, client_id: str, message: str, response: Dict[str, Any]):
        """Enhanced error logging with context information"""
        
        # Get user context for debugging
        user_context = self.agent_factory.conversation_manager.get_user_context(client_id)
        conversation_length = len(self.agent_factory.conversation_manager.get_conversation_history(client_id))
        
        error_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_id": client_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "user_message": message[:200],  # Truncate for privacy
            "response": response,
            "service_stats": self.service_stats,
            "context_info": {
                "conversation_length": conversation_length,
                "has_user_context": bool(user_context),
                "context_keys": list(user_context.keys()) if user_context else []
            }
        }
        
        # In production, this would go to a proper logging system
        print(f"ERROR LOG: {json.dumps(error_log, indent=2)}")
    
    async def cleanup_old_data(self):
        """Enhanced cleanup including conversation history"""
        try:
            # Clean up old sessions and conversation data using factory's method
            self.agent_factory.cleanup_sessions(max_age_hours=24)
            
            # Clean up old connection metadata (48 hours)
            cutoff = datetime.utcnow() - timedelta(hours=48)
            to_remove = []
            
            for client_id, metadata in self.connection_manager.client_metadata.items():
                if client_id not in self.connection_manager.active_connections:
                    connected_at = datetime.fromisoformat(metadata["connected_at"])
                    if connected_at < cutoff:
                        to_remove.append(client_id)
            
            for client_id in to_remove:
                del self.connection_manager.client_metadata[client_id]
            
            if to_remove:
                print(f"Cleanup completed: removed {len(to_remove)} old client records")
            
            # Log cleanup statistics
            total_conversations = len(self.agent_factory.conversation_manager.conversation_history)
            total_contexts = len(self.agent_factory.conversation_manager.user_context)
            print(f"Active conversations: {total_conversations}, Active user contexts: {total_contexts}")
            
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        async def cleanup_task():
            while True:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_old_data()
        
        async def stats_reporting_task():
            while True:
                await asyncio.sleep(300)  # Run every 5 minutes
                stats_summary = {
                    "total_requests": self.service_stats["total_requests"],
                    "context_hit_rate": f"{(self.service_stats['context_hits'] / max(self.service_stats['total_requests'], 1)) * 100:.1f}%",
                    "follow_up_rate": f"{(self.service_stats['follow_up_requests'] / max(self.service_stats['total_requests'], 1)) * 100:.1f}%",
                    "avg_response_time": f"{self.service_stats['average_response_time']:.3f}s",
                    "active_conversations": len(self.agent_factory.conversation_manager.conversation_history)
                }
                print(f"Service Stats: {json.dumps(stats_summary, indent=2)}")
        
        # Start background tasks
        asyncio.create_task(cleanup_task())
        asyncio.create_task(stats_reporting_task())
        
        print("Enhanced Chat Service with Context Memory started!")
        print(f"Initial stats: {json.dumps(self.service_stats, indent=2)}")
    
    async def graceful_shutdown(self):
        """Graceful shutdown with context preservation option"""
        try:
            print("Starting graceful shutdown...")
            
            # Optionally save conversation contexts to persistent storage here
            # await self._save_conversation_contexts_to_db()
            
            # Close all active connections
            for client_id in list(self.connection_manager.active_connections.keys()):
                try:
                    await self.connection_manager.active_connections[client_id].close()
                except:
                    pass
                self.connection_manager.disconnect(client_id)
            
            # Final cleanup
            await self.cleanup_old_data()
            
            print("Graceful shutdown completed")
            print(f"Final stats: {json.dumps(self.service_stats, indent=2)}")
            
        except Exception as e:
            print(f"Error during graceful shutdown: {e}")

# Global instances
connection_manager = ConnectionManager()
enhanced_chat_service = EnhancedChatService()

# Export for use in other modules
__all__ = ['enhanced_chat_service', 'connection_manager', 'EnhancedChatService']

# Integration functions for backward compatibility
async def process_message(message: str, client_id: str) -> Dict[str, Any]:
    """Backward compatible message processing function"""
    return await enhanced_chat_service.process_message(message, client_id)

def get_chat_service():
    """Get the enhanced chat service instance"""
    return enhanced_chat_service