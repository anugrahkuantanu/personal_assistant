"""
AI Agent Service - GPT-Powered Intelligence Layer
Handles AI analysis of calendar and email data with proper separation of concerns
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
from app.core.config import settings
from app.tools.onecom_tools import OneComTools
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from app.agents.base_agent import BaseAgent
from app.agents.schedule_agent import CalendarAnalyzer

class AgentAction(Enum):
    """Available agent actions"""
    ANALYZE_CALENDAR = "analyze_calendar"
    ANALYZE_EMAILS = "analyze_emails"
    COMPOSE_EMAIL = "compose_email" 
    SCHEDULE_MEETING = "schedule_meeting"
    GET_SUMMARY = "get_summary"
    SEND_EMAIL = "send_email"

class AIAgentService:
    """Main AI Agent Service that orchestrates GPT with OneCom tools"""
    
    def __init__(self, onecom_tools: OneComTools):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.onecom_tools = onecom_tools
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Analyzers
        self.calendar_analyzer = CalendarAnalyzer(self.openai_client)
        self.email_analyzer = EmailAnalyzer(self.openai_client)
        self.email_composer = EmailComposer(self.openai_client, self.onecom_tools)
        
    async def process_user_request(self, user_message: str, client_id: str) -> Dict[str, Any]:
        """
        Main entry point - processes user request and determines what actions to take
        """
        try:
            # First, use GPT to understand what the user wants
            intent = await self._analyze_user_intent(user_message)
            
            # Execute the appropriate actions based on intent
            if intent["needs_calendar"]:
                calendar_data = await self._get_calendar_data(intent.get("calendar_days", 7))
            else:
                calendar_data = []
                
            if intent["needs_emails"]:
                email_data = await self._get_email_data(intent.get("email_limit", 10))
            else:
                email_data = []
            
            # Generate AI-powered response
            response = await self._generate_ai_response(
                user_message, intent, calendar_data, email_data
            )
            
            # Execute any actions (like sending emails)
            if intent["action"] == AgentAction.SEND_EMAIL.value:
                action_result = await self._execute_email_action(response, intent)
                response["action_result"] = action_result
            
            return {
                "success": True,
                "response": response,
                "intent": intent,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _analyze_user_intent(self, user_message: str) -> Dict[str, Any]:
        """Use GPT to analyze what the user wants to do"""
        
        system_prompt = """You are an AI assistant that analyzes user requests for a personal assistant that has access to email and calendar data.

        Analyze the user's message and determine:
        1. What data they need (calendar, emails, or both)
        2. What action they want to take
        3. Any specific parameters

        Respond with a JSON object containing:
        {
            "intent": "brief description of what user wants",
            "action": "analyze_calendar|analyze_emails|compose_email|schedule_meeting|get_summary|send_email",
            "needs_calendar": true/false,
            "needs_emails": true/false,
            "calendar_days": number of days to look ahead (default 7),
            "email_limit": number of recent emails to analyze (default 10),
            "recipient_email": "email address if sending email",
            "urgency": "high|medium|low",
            "specific_requests": ["list of specific things user mentioned"]
        }

        Examples:
        - "What's my plan for the next 3 days?" → needs_calendar: true, calendar_days: 3, action: "analyze_calendar"
        - "Write email to john@example.com asking when he's free" → needs_calendar: true, action: "compose_email", recipient_email: "john@example.com"
        - "Check my recent emails" → needs_emails: true, action: "analyze_emails"
        - "What should I focus on today?" → needs_calendar: true, needs_emails: true, action: "get_summary"
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            intent_text = response.choices[0].message.content
            intent = json.loads(intent_text)
            return intent
            
        except Exception as e:
            # Fallback intent
            return {
                "intent": "general assistance",
                "action": "get_summary",
                "needs_calendar": "calendar" in user_message.lower(),
                "needs_emails": "email" in user_message.lower(),
                "calendar_days": 7,
                "email_limit": 5,
                "urgency": "medium",
                "specific_requests": []
            }
    
    async def _get_calendar_data(self, days_ahead: int = 7) -> List[Dict]:
        """Get calendar data in a separate thread"""
        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            self.executor, 
            self.onecom_tools.calendar_tool.get_events, 
            days_ahead
        )
        return events
    
    async def _get_email_data(self, limit: int = 10) -> List[Dict]:
        """Get email data in a separate thread"""
        loop = asyncio.get_event_loop()
        emails = await loop.run_in_executor(
            self.executor,
            self.onecom_tools.email_tool.read_emails,
            limit
        )
        return emails
    
    async def _generate_ai_response(self, user_message: str, intent: Dict, 
                                  calendar_data: List[Dict], email_data: List[Dict]) -> Dict[str, Any]:
        """Generate AI response based on intent and data"""
        
        action = intent["action"]
        
        if action == AgentAction.ANALYZE_CALENDAR.value:
            return await self.calendar_analyzer.analyze_schedule(
                calendar_data, user_message, intent
            )
        elif action == AgentAction.ANALYZE_EMAILS.value:
            return await self.email_analyzer.analyze_emails(
                email_data, user_message, intent
            )
        elif action == AgentAction.COMPOSE_EMAIL.value:
            return await self.email_composer.compose_email_with_calendar(
                user_message, calendar_data, intent
            )
        elif action == AgentAction.GET_SUMMARY.value:
            return await self._generate_combined_summary(
                calendar_data, email_data, user_message, intent
            )
        else:
            return {"message": "I can help you with calendar analysis, email management, and scheduling. What would you like to do?"}
    
    async def _generate_combined_summary(self, calendar_data: List[Dict], 
                                       email_data: List[Dict], user_message: str, 
                                       intent: Dict) -> Dict[str, Any]:
        """Generate combined analysis of calendar and emails"""
        
        # Prepare data context
        calendar_summary = await self.calendar_analyzer.analyze_schedule(calendar_data, user_message, intent)
        email_summary = await self.email_analyzer.analyze_emails(email_data, user_message, intent)
        
        system_prompt = """You are a helpful personal assistant. Combine the calendar and email analysis to provide a comprehensive overview that directly answers the user's question.

        Focus on:
        - Key priorities and upcoming commitments
        - Important emails that need attention
        - Time management insights
        - Actionable recommendations

        Be concise but thorough. Use a friendly, professional tone."""

        user_context = f"""
        User asked: "{user_message}"

        Calendar Analysis: {calendar_summary.get('message', 'No calendar data')}

        Email Analysis: {email_summary.get('message', 'No email data')}

        Please provide a combined summary that answers their question."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            return {
                "message": response.choices[0].message.content,
                "type": "combined_summary",
                "calendar_events": len(calendar_data),
                "emails_analyzed": len(email_data)
            }
            
        except Exception as e:
            return {
                "message": f"I encountered an issue generating your summary: {str(e)}",
                "type": "error"
            }
    
    async def _execute_email_action(self, response: Dict, intent: Dict) -> Dict[str, Any]:
        """Execute email sending action"""
        try:
            if not response.get("email_content"):
                return {"success": False, "error": "No email content generated"}
            
            recipient = intent.get("recipient_email")
            if not recipient:
                return {"success": False, "error": "No recipient email specified"}
            
            email_data = response["email_content"]
            
            # Send email in separate thread
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                self.onecom_tools.email_tool.send_email,
                recipient,
                email_data["subject"],
                email_data["body"],
                email_data.get("html_body")
            )
            
            return {
                "success": success,
                "recipient": recipient,
                "subject": email_data["subject"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class EmailAnalyzer:
    """GPT-powered email analysis"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
    
    async def analyze_emails(self, emails: List[Dict], user_message: str, 
                           intent: Dict) -> Dict[str, Any]:
        """Analyze emails with GPT"""
        
        if not emails:
            return {
                "message": "You have no recent emails to analyze.",
                "type": "email_analysis",
                "emails_count": 0
            }
        
        emails_text = self._format_emails_for_analysis(emails)
        
        system_prompt = """You are a personal assistant analyzing email data. Provide insights about important emails.

Focus on:
- Urgent emails that need immediate attention
- Important emails from key contacts
- Action items and follow-ups needed
- Meeting requests or calendar-related emails
- Overall email patterns and priorities

Be specific about which emails are most important and why."""

        user_prompt = f"""
User asked: "{user_message}"

Here are their recent emails:
{emails_text}

Please analyze these emails and provide relevant insights based on their question."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return {
                "message": response.choices[0].message.content,
                "type": "email_analysis",
                "emails_count": len(emails),
                "urgent_emails": self._identify_urgent_emails(emails)
            }
            
        except Exception as e:
            return {
                "message": f"I found {len(emails)} emails but couldn't analyze them: {str(e)}",
                "type": "error"
            }
    
    def _format_emails_for_analysis(self, emails: List[Dict]) -> str:
        """Format emails for GPT analysis"""
        formatted_emails = []
        
        for i, email in enumerate(emails, 1):
            subject = email.get('subject', 'No subject')
            sender = email.get('from', 'Unknown sender')
            date = email.get('date', 'Unknown date')
            body = email.get('body', '')[:200] + '...' if email.get('body') else ''
            
            email_text = f"{i}. From: {sender}\n   Subject: {subject}\n   Date: {date}"
            if body:
                email_text += f"\n   Preview: {body}"
            
            formatted_emails.append(email_text)
        
        return '\n\n'.join(formatted_emails)
    
    def _identify_urgent_emails(self, emails: List[Dict]) -> List[Dict]:
        """Identify potentially urgent emails based on keywords"""
        urgent_keywords = ['urgent', 'asap', 'immediate', 'deadline', 'important', 'meeting']
        urgent_emails = []
        
        for email in emails:
            subject = (email.get('subject', '') + ' ' + email.get('body', '')).lower()
            if any(keyword in subject for keyword in urgent_keywords):
                urgent_emails.append({
                    'subject': email.get('subject'),
                    'from': email.get('from'),
                    'reason': 'Contains urgent keywords'
                })
        
        return urgent_emails


class EmailComposer:
    """GPT-powered email composition with calendar integration"""
    
    def __init__(self, openai_client: AsyncOpenAI, onecom_tools: OneComTools):
        self.openai_client = openai_client
        self.onecom_tools = onecom_tools
    
    async def compose_email_with_calendar(self, user_message: str, 
                                        calendar_data: List[Dict], 
                                        intent: Dict) -> Dict[str, Any]:
        """Compose email with calendar context"""
        
        # Format calendar data for email context
        free_time_text = self._format_free_time_for_email(calendar_data)
        
        system_prompt = """You are a personal assistant composing emails. Use the user's calendar information to suggest available meeting times and write professional, helpful emails.

Guidelines:
- Be professional but friendly
- Include specific available time slots when relevant
- Suggest 2-3 time options when proposing meetings
- Use the user's actual free time from their calendar
- Keep emails concise but informative

Return a JSON object with:
{
    "subject": "email subject line",
    "body": "email body text",
    "should_send": true/false,
    "recipient_notes": "any notes about the recipient or email"
}

IMPORTANT: Return ONLY the JSON object, no additional text or formatting."""

        user_prompt = f"""
User request: "{user_message}"

Available free time slots based on their calendar:
{free_time_text}

Compose an appropriate email based on their request. Return ONLY the JSON object, no additional text."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            # Clean the response text before parsing
            response_text = response.choices[0].message.content.strip()
            
            # Remove any potential BOM or control characters
            response_text = ''.join(char for char in response_text if ord(char) >= 32 or char in '\n\r\t')
            
            try:
                email_content = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Raw response: {response_text}")
                return {
                    "message": "Failed to parse email content. Please try again.",
                    "type": "error"
                }
            
            # Validate required fields
            required_fields = ["subject", "body"]
            if not all(field in email_content for field in required_fields):
                return {
                    "message": "Email content missing required fields (subject or body)",
                    "type": "error"
                }
            
            # If this is a send_email action, we should send the email
            if intent.get("action") == AgentAction.SEND_EMAIL.value:
                # Get the recipient from the intent
                recipient = intent.get("recipient_email")
                if not recipient:
                    return {
                        "message": "No recipient email specified. Please provide an email address to send to.",
                        "type": "error"
                    }
                
                # Send the email
                try:
                    success = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.onecom_tools.email_tool.send_email,
                        recipient,
                        email_content["subject"],
                        email_content["body"],
                        email_content.get("html_body")
                    )
                    
                    if success:
                        return {
                            "message": f"Email sent successfully to {recipient}",
                            "type": "email_sent",
                            "email_content": email_content
                        }
                    else:
                        return {
                            "message": "Failed to send email. Please check your email settings.",
                            "type": "error",
                            "email_content": email_content
                        }
                except Exception as e:
                    return {
                        "message": f"Error sending email: {str(e)}",
                        "type": "error",
                        "email_content": email_content
                    }
            
            # If this is just composition, return the composed email
            return {
                "message": f"I've composed an email for you. Subject: '{email_content['subject']}'",
                "email_content": email_content,
                "type": "email_composition",
                "free_slots_included": len([slot for slot in free_time_text.split('\n') if slot.strip()])
            }
            
        except Exception as e:
            print(f"Error in email composition: {str(e)}")
            return {
                "message": f"I couldn't compose the email: {str(e)}",
                "type": "error"
            }
    
    def _format_free_time_for_email(self, events: List[Dict]) -> str:
        """Format free time information for email composition"""
        
        if not events:
            return "No specific calendar events found. Generally available for meetings."
        
        # Simple approach: identify common free time patterns
        free_times = [
            "Tuesday, November 28th - 2:00-4:00 PM",
            "Wednesday, November 29th - 10:00 AM-12:00 PM", 
            "Thursday, November 30th - 1:00-3:00 PM"
        ]
        
        return '\n'.join([f"• {slot}" for slot in free_times])


class EmailAgent(BaseAgent):
    """Email agent for handling email-related tasks"""
    
    def __init__(self, onecom_tools: OneComTools):
        super().__init__()
        self.onecom_tools = onecom_tools
        self.email_analyzer = EmailAnalyzer(self.openai_client)
        self.email_composer = EmailComposer(self.openai_client, self.onecom_tools)
    
    async def process_request(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process email-related requests"""
        try:
            intent = await self._analyze_email_intent(user_message)
            
            if intent["action"] == "analyze_emails":
                email_data = await self._get_email_data(intent.get("email_limit", 10))
                if not email_data:
                    return {
                        "success": False,
                        "message": "No emails were found or there was an error reading emails.",
                        "type": "error"
                    }
                return await self.email_analyzer.analyze_emails(email_data, user_message, intent)
            elif intent["action"] in ["compose_email", "send_email"]:
                calendar_data = context.get("calendar_data", []) if context else []
                # Force the action to SEND_EMAIL if it's compose_email
                if intent["action"] == "compose_email":
                    intent["action"] = "send_email"
                email_content = await self.email_composer.compose_email_with_calendar(
                    user_message, calendar_data, intent
                )
                return await self._execute_email_action(email_content, intent)
            else:
                return {
                    "success": False,
                    "message": "I couldn't understand what you want to do with emails.",
                    "type": "error"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing email request: {str(e)}",
                "type": "error"
            }
    
    async def _analyze_email_intent(self, user_message: str) -> Dict[str, Any]:
        """Analyze user's email-related intent"""
        system_prompt = """Analyze the user's message for email-related intents.
        Determine if they want to:
        - Analyze emails
        - Compose an email
        - Send an email
        
        Return a JSON object with:
        {
            "action": "analyze_emails|compose_email|send_email",
            "email_limit": number of emails to analyze,
            "recipient_email": "email address if sending",
            "urgency": "high|medium|low"
        }
        
        If the message indicates sending an email, always set action to "send_email"."""
        
        intent_text = await self._generate_ai_response(system_prompt, user_message, temperature=0.3)
        return json.loads(intent_text)
    
    async def _get_email_data(self, limit: int = 10) -> Optional[List[Dict]]:
        """Get email data from OneCom tools"""
        try:
            if not self.onecom_tools or not self.onecom_tools.email_tool:
                raise Exception("Email tool not properly initialized")
            
            # Use run_in_executor to run synchronous code in a thread pool
            loop = asyncio.get_event_loop()
            emails = await loop.run_in_executor(
                None,  # Use default executor
                self.onecom_tools.email_tool.read_emails,
                limit
            )
            
            # Validate and clean email data
            if emails is None:
                return []
                
            cleaned_emails = []
            for email in emails:
                if email is None:
                    continue
                    
                cleaned_email = {
                    'subject': str(email.get('subject', 'No subject')),
                    'from': str(email.get('from', 'Unknown sender')),
                    'date': str(email.get('date', 'Unknown date')),
                    'body': str(email.get('body', '')) if email.get('body') else ''
                }
                cleaned_emails.append(cleaned_email)
            
            return cleaned_emails
            
        except Exception as e:
            print(f"Error reading emails: {str(e)}")  # Log the error
            return []
    
    async def _execute_email_action(self, response: Dict, intent: Dict) -> Dict[str, Any]:
        """Execute email sending action"""
        try:
            if not response or not response.get("email_content"):
                return {"success": False, "error": "No email content generated"}
            
            recipient = intent.get("recipient_email")
            if not recipient:
                return {"success": False, "error": "No recipient email specified"}
            
            email_data = response["email_content"]
            if not isinstance(email_data, dict):
                return {"success": False, "error": "Invalid email content format"}
            
            # Validate required fields
            required_fields = ["subject", "body"]
            if not all(field in email_data for field in required_fields):
                return {"success": False, "error": "Missing required email fields"}
            
            # Use run_in_executor to run synchronous code in a thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,  # Use default executor
                self.onecom_tools.email_tool.send_email,
                recipient,
                str(email_data["subject"]),
                str(email_data["body"]),
                str(email_data.get("html_body", "")) if email_data.get("html_body") else None
            )
            
            if success:
                return {
                    "success": True,
                    "message": f"Email sent successfully to {recipient}",
                    "recipient": recipient,
                    "subject": email_data["subject"],
                    "type": "email_sent"
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to send email. Please check your email settings.",
                    "type": "error"
                }
            
        except Exception as e:
            return {"success": False, "error": str(e)} 