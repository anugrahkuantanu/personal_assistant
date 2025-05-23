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
        self.email_composer = EmailComposer(self.openai_client)
        
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


class CalendarAnalyzer:
    """GPT-powered calendar analysis"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
    
    async def analyze_schedule(self, events: List[Dict], user_message: str, 
                             intent: Dict) -> Dict[str, Any]:
        """Analyze calendar events with GPT"""
        
        if not events:
            return {
                "message": "You have no upcoming events in your calendar.",
                "type": "calendar_analysis",
                "events_count": 0
            }
        
        # Prepare events for GPT analysis
        events_text = self._format_events_for_analysis(events)
        
        system_prompt = """You are a personal assistant analyzing calendar data. Provide insights about the user's schedule.

        Focus on:
        - Key appointments and commitments
        - Time conflicts or busy periods
        - Free time availability
        - Scheduling patterns and recommendations
        - Priority items that need attention

        Be specific and actionable in your analysis."""

        user_prompt = f"""
User asked: "{user_message}"

Here are their upcoming calendar events:
{events_text}

Please analyze their schedule and provide relevant insights based on their question."""

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
                "type": "calendar_analysis",
                "events_count": len(events),
                "free_time_slots": self._identify_free_time(events)
            }
            
        except Exception as e:
            return {
                "message": f"I found {len(events)} events but couldn't analyze them: {str(e)}",
                "type": "error"
            }
    
    def _format_events_for_analysis(self, events: List[Dict]) -> str:
        """Format events for GPT analysis"""
        formatted_events = []
        
        for event in events:
            start = event.get('start', 'Unknown time')
            end = event.get('end', 'Unknown time')
            summary = event.get('summary', 'No title')
            location = event.get('location', '')
            description = event.get('description', '')
            
            event_text = f"• {summary}"
            if start:
                if hasattr(start, 'strftime'):
                    event_text += f" - {start.strftime('%A, %B %d at %I:%M %p')}"
                else:
                    event_text += f" - {start}"
            
            if location:
                event_text += f" (Location: {location})"
            
            if description:
                event_text += f" - {description[:100]}..."
            
            formatted_events.append(event_text)
        
        return '\n'.join(formatted_events)
    
    def _identify_free_time(self, events: List[Dict]) -> List[Dict]:
        """Identify free time slots between events"""
        # Simple free time identification
        # This could be enhanced with more sophisticated logic
        free_slots = []
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda e: e.get('start', datetime.now()))
        
        # Find gaps between events (simplified)
        for i in range(len(sorted_events) - 1):
            current_end = sorted_events[i].get('end')
            next_start = sorted_events[i + 1].get('start')
            
            if current_end and next_start and hasattr(current_end, 'hour') and hasattr(next_start, 'hour'):
                gap_hours = (next_start - current_end).total_seconds() / 3600
                if gap_hours > 1:  # At least 1 hour gap
                    free_slots.append({
                        "start": current_end,
                        "end": next_start,
                        "duration_hours": gap_hours
                    })
        
        return free_slots


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
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
    
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
}"""

        user_prompt = f"""
User request: "{user_message}"

Available free time slots based on their calendar:
{free_time_text}

Compose an appropriate email based on their request."""

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
            
            email_content = json.loads(response.choices[0].message.content)
            
            return {
                "message": f"I've composed an email for you. Subject: '{email_content['subject']}'",
                "email_content": email_content,
                "type": "email_composition",
                "free_slots_included": len([slot for slot in free_time_text.split('\n') if slot.strip()])
            }
            
        except Exception as e:
            return {
                "message": f"I couldn't compose the email: {str(e)}",
                "type": "error"
            }
    
    def _format_free_time_for_email(self, events: List[Dict]) -> str:
        """Format free time information for email composition"""
        
        if not events:
            return "No specific calendar events found. Generally available for meetings."
        
        # Simple approach: identify common free time patterns
        free_times = []
        
        # Check common meeting times
        common_times = [
            "Monday 9:00-11:00 AM",
            "Tuesday 2:00-4:00 PM", 
            "Wednesday 10:00-12:00 PM",
            "Thursday 1:00-3:00 PM",
            "Friday 9:00-11:00 AM"
        ]
        
        # In a real implementation, you'd analyze the actual calendar events
        # For now, provide some sample availability
        free_times = [
            "Tuesday, November 28th - 2:00-4:00 PM",
            "Wednesday, November 29th - 10:00 AM-12:00 PM", 
            "Thursday, November 30th - 1:00-3:00 PM"
        ]
        
        return '\n'.join([f"• {slot}" for slot in free_times]) 