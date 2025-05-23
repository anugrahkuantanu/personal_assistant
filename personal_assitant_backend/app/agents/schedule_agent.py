"""
Schedule Agent - Handles calendar and scheduling tasks
"""

import json
from typing import Dict, List, Any
from datetime import datetime, timedelta, timezone
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
                # Convert all datetime objects to timezone-aware
                calendar_data = self._ensure_timezone_aware(calendar_data)
                return await self.calendar_analyzer.analyze_schedule(calendar_data, user_message, intent)
            elif intent["action"] == "schedule_meeting":
                calendar_data = await self._get_calendar_data(intent.get("calendar_days", 7))
                # Convert all datetime objects to timezone-aware
                calendar_data = self._ensure_timezone_aware(calendar_data)
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
        system_prompt = """You are an AI assistant that analyzes user requests about calendar and scheduling.

Analyze the user's message and determine:
1. What they want to do with their calendar
2. How many days ahead they want to look
3. If they want to schedule a meeting, how long it should be

Return a JSON object with:
{
    "action": "analyze_calendar|schedule_meeting",
    "calendar_days": number of days to look ahead (default 7),
    "meeting_duration": duration in minutes if scheduling (default 60),
    "priority": "high|medium|low",
    "specific_requests": ["list of specific things user mentioned"]
}

Examples:
- "What's my schedule for next week?" → action: "analyze_calendar", calendar_days: 7
- "When am I free tomorrow?" → action: "analyze_calendar", calendar_days: 1
- "Schedule a meeting with John" → action: "schedule_meeting", meeting_duration: 60
- "Find a 2-hour slot next week" → action: "schedule_meeting", meeting_duration: 120, calendar_days: 7
- "Show me my calendar" → action: "analyze_calendar", calendar_days: 7

IMPORTANT: Default to "analyze_calendar" if unsure, as it's safer than scheduling."""

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
            
            # Ensure required fields exist
            if "action" not in intent:
                intent["action"] = "analyze_calendar"  # Default to analyze_calendar
            if "calendar_days" not in intent:
                intent["calendar_days"] = 7  # Default to 7 days
            if "meeting_duration" not in intent:
                intent["meeting_duration"] = 60  # Default to 1 hour
            if "priority" not in intent:
                intent["priority"] = "medium"
            if "specific_requests" not in intent:
                intent["specific_requests"] = []
            
            return intent
            
        except Exception as e:
            print(f"Error analyzing schedule intent: {str(e)}")
            # Return a safe default intent
            return {
                "action": "analyze_calendar",
                "calendar_days": 7,
                "meeting_duration": 60,
                "priority": "medium",
                "specific_requests": []
            }
    
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

    def _ensure_timezone_aware(self, events: List[Dict]) -> List[Dict]:
        """Convert all datetime objects in events to timezone-aware"""
        for event in events:
            if 'start' in event and event['start'] and not event['start'].tzinfo:
                event['start'] = event['start'].replace(tzinfo=timezone.utc)
            if 'end' in event and event['end'] and not event['end'].tzinfo:
                event['end'] = event['end'].replace(tzinfo=timezone.utc)
        return events

class CalendarAnalyzer:
    """Calendar analysis functionality"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        # Define business hours (9 AM to 5 PM)
        self.business_hours_start = 9  # 9 AM
        self.business_hours_end = 17   # 5 PM
        self.working_days = [0, 1, 2, 3, 4]  # Monday to Friday
    
    def _format_free_time_slots(self, free_slots: List[Dict]) -> List[Dict]:
        """Format free time slots for JSON serialization"""
        formatted_slots = []
        for slot in free_slots:
            formatted_slots.append({
                "start": slot["start"].isoformat() if isinstance(slot["start"], datetime) else str(slot["start"]),
                "end": slot["end"].isoformat() if isinstance(slot["end"], datetime) else str(slot["end"]),
                "duration_hours": slot["duration_hours"]
            })
        return formatted_slots

    async def analyze_schedule(self, events: List[Dict], user_message: str, 
                             intent: Dict) -> Dict[str, Any]:
        """Analyze calendar events with GPT"""
        if not events:
            return {
                "message": "You have no upcoming events in your calendar.",
                "type": "calendar_analysis",
                "events_count": 0
            }
        
        # Ensure all datetime objects are timezone-aware
        events = self._ensure_timezone_aware(events)
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
            
            # Format free time slots for JSON serialization
            free_slots = self._identify_free_time(events)
            formatted_slots = self._format_free_time_slots(free_slots)
            
            return {
                "message": response.choices[0].message.content,
                "type": "calendar_analysis",
                "events_count": len(events),
                "free_time_slots": formatted_slots
            }
            
        except Exception as e:
            print(f"Error analyzing calendar: {str(e)}")
            return {
                "message": f"I found {len(events)} events but couldn't analyze them: {str(e)}",
                "type": "error"
            }
    
    def _ensure_timezone_aware(self, events: List[Dict]) -> List[Dict]:
        """Convert all datetime objects in events to timezone-aware"""
        for event in events:
            if 'start' in event and event['start'] and not event['start'].tzinfo:
                event['start'] = event['start'].replace(tzinfo=timezone.utc)
            if 'end' in event and event['end'] and not event['end'].tzinfo:
                event['end'] = event['end'].replace(tzinfo=timezone.utc)
        return events
    
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
        """Identify free time slots between events with business hours consideration"""
        free_slots = []
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda e: e.get('start', datetime.now(timezone.utc)))
        
        # Get the date range from events
        if not sorted_events:
            # If no events, return business hours for next 7 days
            start_date = datetime.now(timezone.utc).replace(hour=self.business_hours_start, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=7)
            return self._get_business_hours_slots(start_date, end_date)
        
        first_event_date = sorted_events[0]['start'].date()
        last_event_date = sorted_events[-1]['end'].date()
        
        # Process each day in the range
        current_date = first_event_date
        while current_date <= last_event_date:
            if current_date.weekday() in self.working_days:
                # Get events for this day
                day_events = [
                    e for e in sorted_events 
                    if e['start'].date() == current_date or e['end'].date() == current_date
                ]
                
                # Sort day events by start time
                day_events.sort(key=lambda e: e['start'])
                
                # Start of business day
                day_start = datetime.combine(current_date, datetime.min.time()).replace(
                    hour=self.business_hours_start, minute=0, second=0, microsecond=0,
                    tzinfo=timezone.utc
                )
                
                # End of business day
                day_end = datetime.combine(current_date, datetime.min.time()).replace(
                    hour=self.business_hours_end, minute=0, second=0, microsecond=0,
                    tzinfo=timezone.utc
                )
                
                # Find free slots for this day
                if not day_events:
                    # If no events, entire business day is free
                    free_slots.append({
                        "start": day_start,
                        "end": day_end,
                        "duration_hours": self.business_hours_end - self.business_hours_start
                    })
                else:
                    # Check time before first event
                    first_event_start = day_events[0]['start']
                    if first_event_start > day_start:
                        gap_hours = (first_event_start - day_start).total_seconds() / 3600
                        if gap_hours >= 1:  # At least 1 hour gap
                            free_slots.append({
                                "start": day_start,
                                "end": first_event_start,
                                "duration_hours": gap_hours
                            })
                    
                    # Check gaps between events
                    for i in range(len(day_events) - 1):
                        current_end = day_events[i]['end']
                        next_start = day_events[i + 1]['start']
                        
                        if current_end < next_start:
                            gap_hours = (next_start - current_end).total_seconds() / 3600
                            if gap_hours >= 1:  # At least 1 hour gap
                                free_slots.append({
                                    "start": current_end,
                                    "end": next_start,
                                    "duration_hours": gap_hours
                                })
                    
                    # Check time after last event
                    last_event_end = day_events[-1]['end']
                    if last_event_end < day_end:
                        gap_hours = (day_end - last_event_end).total_seconds() / 3600
                        if gap_hours >= 1:  # At least 1 hour gap
                            free_slots.append({
                                "start": last_event_end,
                                "end": day_end,
                                "duration_hours": gap_hours
                            })
            
            current_date += timedelta(days=1)
        
        return free_slots
    
    def _get_business_hours_slots(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate free slots for business hours when no events exist"""
        free_slots = []
        current_date = start_date.date()
        
        while current_date <= end_date.date():
            if current_date.weekday() in self.working_days:
                day_start = datetime.combine(current_date, datetime.min.time()).replace(
                    hour=self.business_hours_start, minute=0, second=0, microsecond=0,
                    tzinfo=timezone.utc
                )
                day_end = datetime.combine(current_date, datetime.min.time()).replace(
                    hour=self.business_hours_end, minute=0, second=0, microsecond=0,
                    tzinfo=timezone.utc
                )
                
                free_slots.append({
                    "start": day_start,
                    "end": day_end,
                    "duration_hours": self.business_hours_end - self.business_hours_start
                })
            
            current_date += timedelta(days=1)
        
        return free_slots 