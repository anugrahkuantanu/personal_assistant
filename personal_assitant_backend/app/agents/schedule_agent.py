"""
Schedule Agent - Handles calendar and scheduling tasks
"""

import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
from app.agents.base_agent import BaseAgent
from app.tools.onecom_tools import OneComTools
import asyncio

class ScheduleAgent(BaseAgent):
    """Schedule agent for handling calendar and scheduling tasks"""
    
    def __init__(self, onecom_tools: OneComTools):
        super().__init__()
        self.onecom_tools = onecom_tools
        self.calendar_analyzer = CalendarAnalyzer(self.openai_client)
    
    async def process_request(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process schedule-related requests"""
        try:
            intent = await self._analyze_schedule_intent(user_message)
            
            if intent["action"] == "analyze_calendar":
                calendar_data = await self._get_calendar_data(intent.get("calendar_days", 7))
                return await self.calendar_analyzer.analyze_schedule(calendar_data, user_message, intent)
            elif intent["action"] == "schedule_meeting":
                calendar_data = await self._get_calendar_data(intent.get("calendar_days", 7))
                return await self._handle_meeting_scheduling(calendar_data, user_message, intent)
            else:
                return {
                    "success": False,
                    "message": "I couldn't understand what you want to do with your schedule.",
                    "type": "error"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing schedule request: {str(e)}",
                "type": "error"
            }
    
    async def _analyze_schedule_intent(self, user_message: str) -> Dict[str, Any]:
        """Analyze user's schedule-related intent"""
        system_prompt = """Analyze the user's message for schedule-related intents.
        Determine if they want to:
        - Analyze their calendar
        - Schedule a meeting
        - Check availability
        
        Return a JSON object with:
        {
            "action": "analyze_calendar|schedule_meeting",
            "calendar_days": number of days to look ahead,
            "meeting_duration": duration in minutes if scheduling,
            "priority": "high|medium|low"
        }"""
        
        intent_text = await self._generate_ai_response(system_prompt, user_message, temperature=0.3)
        return json.loads(intent_text)
    
    async def _get_calendar_data(self, days_ahead: int = 7) -> List[Dict]:
        """Get calendar data from OneCom tools"""
        try:
            # Use run_in_executor to run synchronous code in a thread pool
            loop = asyncio.get_event_loop()
            events = await loop.run_in_executor(
                None,  # Use default executor
                self.onecom_tools.calendar_tool.get_events,
                days_ahead
            )
            return events
        except Exception as e:
            raise Exception(f"Error getting calendar data: {str(e)}")
    
    async def _handle_meeting_scheduling(self, calendar_data: List[Dict], 
                                       user_message: str, intent: Dict) -> Dict[str, Any]:
        """Handle meeting scheduling requests"""
        try:
            # Analyze available slots
            free_slots = self.calendar_analyzer._identify_free_time(calendar_data)
            
            if not free_slots:
                return {
                    "success": False,
                    "message": "I couldn't find any suitable time slots for your meeting.",
                    "type": "error"
                }
            
            # Format available slots for response
            available_slots = [
                f"{slot['start'].strftime('%A, %B %d at %I:%M %p')} - "
                f"{slot['end'].strftime('%I:%M %p')}"
                for slot in free_slots[:3]  # Show top 3 slots
            ]
            
            return {
                "success": True,
                "message": "Here are some available time slots for your meeting:",
                "type": "meeting_slots",
                "available_slots": available_slots,
                "total_slots": len(free_slots)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error scheduling meeting: {str(e)}",
                "type": "error"
            }


class CalendarAnalyzer:
    """Calendar analysis functionality"""
    
    def __init__(self, openai_client):
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
        free_slots = []
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda e: e.get('start', datetime.now()))
        
        # Find gaps between events
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