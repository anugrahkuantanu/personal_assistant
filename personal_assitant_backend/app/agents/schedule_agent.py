"""
Schedule Agent - Handles calendar and scheduling tasks
"""

import json
from typing import Dict, List, Any, Tuple
from datetime import datetime, time, timedelta, timezone
from app.agents.base_agent import BaseAgent
from app.tools.onecom_tools import OneComTools
import asyncio

class ScheduleAgent(BaseAgent):
    """Schedule agent for handling calendar and scheduling tasks"""
    
    def __init__(self, onecom_tools: OneComTools):
        super().__init__()
        self.onecom_tools = onecom_tools
        self.calendar_analyzer = CalendarAnalyzer(self.openai_client)
    
    async def _analyze_schedule_intent(self, user_message: str) -> Dict[str, Any]:
        """Analyze user's schedule-related intent"""
        system_prompt = """You are an advanced AI assistant for calendar and scheduling. Your job is to extract ALL relevant details from the user's message for any scheduling or calendar request.

Analyze the user's message and return a JSON object with these fields (fill as many as possible):
{
  "action": "analyze_calendar|find_free_slots|schedule_meeting|create_event|show_events|other",
  "calendar_days": number of days to look ahead (default 7),
  "date": "specific date or day if mentioned (e.g. '2025-05-26' or 'Monday')",
  "time": "specific time or time range if mentioned (e.g. '14:00', 'afternoon', '9am-11am')",
  "meeting_duration": duration in minutes if scheduling (default 60),
  "attendees": ["list of attendees if mentioned"],
  "event_title": "title or subject of the event/meeting if mentioned",
  "location": "location if mentioned",
  "priority": "high|medium|low",
  "specific_requests": ["list of specific things user mentioned (e.g. 'free slot', 'busy time', 'add event')"]
}

Instructions:
- If the user asks for free/busy time, set action to "find_free_slots".
- If the user wants to create/schedule a meeting or event, set action to "schedule_meeting" or "create_event" and extract all possible details.
- If the user wants to see their schedule, set action to "analyze_calendar" or "show_events".
- Always extract as much detail as possible (date, time, duration, attendees, etc), even if not all are present.
- If unsure, default to "analyze_calendar".

Examples:
- "What's my schedule for next week?" → action: "analyze_calendar", calendar_days: 7
- "When am I free on Monday?" → action: "find_free_slots", date: "Monday"
- "Schedule a meeting with John on Friday at 2pm for 30 minutes" → action: "schedule_meeting", attendees: ["John"], date: "Friday", time: "14:00", meeting_duration: 30
- "Add dentist appointment next Thursday at 10am" → action: "create_event", event_title: "dentist appointment", date: "next Thursday", time: "10:00"
- "Show me all my events tomorrow" → action: "show_events", date: "tomorrow"

Return ONLY the JSON object, no extra text. If a field is not present, omit it."""

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

    async def process_request(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process schedule-related requests"""
        try:
            intent = await self._analyze_schedule_intent(user_message)
            action = intent.get("action", "analyze_calendar")
            # Use all arguments from LLM intent
            calendar_days = intent.get("calendar_days", 7)
            date = intent.get("date")
            time_str = intent.get("time")
            meeting_duration = intent.get("meeting_duration", 60)
            attendees = intent.get("attendees", [])
            event_title = intent.get("event_title")
            location = intent.get("location")
            # Find free slots
            if action in ["find_free_slots", "schedule_meeting", "create_event"]:
                calendar_data = await self._get_calendar_data(calendar_days)
                calendar_data = self._ensure_timezone_aware(calendar_data)
                free_slots = self.calendar_analyzer._identify_free_time(calendar_data, date=date, time_str=time_str, duration=meeting_duration)
                if action == "find_free_slots":
                    if not free_slots:
                        return {
                            "success": False,
                            "message": "No free slots found for your request.",
                            "type": "no_free_slots"
                        }
                    formatted_slots = self.calendar_analyzer._format_free_time_slots(free_slots)
                    return {
                        "success": True,
                        "message": "Here are your available free slots:",
                        "type": "free_slots",
                        "free_slots": formatted_slots
                    }
                elif action in ["schedule_meeting", "create_event"]:
                    if not free_slots:
                        return {
                            "success": False,
                            "message": "No available time slots to schedule your event.",
                            "type": "no_free_slots"
                        }
                    # Pick the first available slot that matches criteria
                    slot = free_slots[0]
                    # Create the event using all arguments
                    event_created = self.onecom_tools.calendar_tool.create_event(
                        summary=event_title or "Meeting",
                        start_time=slot["start"],
                        end_time=slot["end"],
                        description=f"Scheduled via assistant. Attendees: {', '.join(attendees)}" if attendees else "Scheduled via assistant.",
                        location=location or ""
                    )
                    if event_created:
                        return {
                            "success": True,
                            "message": f"Event '{event_title or 'Meeting'}' scheduled successfully.",
                            "type": "event_created",
                            "event": {
                                "title": event_title or "Meeting",
                                "start": slot["start"].isoformat(),
                                "end": slot["end"].isoformat(),
                                "attendees": attendees,
                                "location": location
                            }
                        }
                    else:
                        return {
                            "success": False,
                            "message": "Failed to create the event. Please try again.",
                            "type": "event_creation_failed"
                        }
            elif action in ["analyze_calendar", "show_events"]:
                calendar_data = await self._get_calendar_data(calendar_days)
                calendar_data = self._ensure_timezone_aware(calendar_data)
                return await self.calendar_analyzer.analyze_schedule(calendar_data, user_message, intent)
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
    
    def _identify_free_time(self, events: List[Dict], date: str = None, time_str: str = None, duration: int = 60) -> List[Dict]:
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
        
        # Filter free slots by date and time if specified
        if date and time_str:
            free_slots = self._filter_free_slots_by_date_time(free_slots, date, time_str, duration)
        
        return free_slots
    
    def _filter_free_slots_by_date_time(self, free_slots: List[Dict], date: str, time_str: str, duration: int) -> List[Dict]:
        """Filter free slots by specific date, time, and duration"""
        filtered_slots = []
        
        # Parse the target date
        target_date = None
        if date.lower() == "today":
            target_date = datetime.now(timezone.utc).date()
        elif date.lower() == "tomorrow":
            target_date = (datetime.now(timezone.utc) + timedelta(days=1)).date()
        else:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                # Try parsing as a relative date (e.g., "next Friday")
                target_date = self._parse_relative_date(date)
        
        if target_date:
            # Filter slots that match the target date
            for slot in free_slots:
                if slot['start'].date() == target_date:
                    filtered_slots.append(slot)
        
        # Further filter by time range if specified
        if time_str and filtered_slots:
            time_ranges = self._parse_time_ranges(time_str)
            if time_ranges:
                final_filtered_slots = []
                for slot in filtered_slots:
                    slot_start = slot['start'].time()
                    slot_end = slot['end'].time()
                    for start_time, end_time in time_ranges:
                        if slot_start >= start_time and slot_end <= end_time:
                            final_filtered_slots.append(slot)
                            break
                filtered_slots = final_filtered_slots
        
        # Filter by minimum duration
        if duration and filtered_slots:
            filtered_slots = [
                slot for slot in filtered_slots 
                if (slot['end'] - slot['start']).total_seconds() / 60 >= duration
            ]
        
        return filtered_slots
    
    def _parse_relative_date(self, date_str: str) -> datetime:
        """Parse relative date expressions (e.g., 'next Friday')"""
        today = datetime.now(timezone.utc)
        weekday_mapping = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6
        }
        
        # Check for "next" prefix
        if date_str.startswith("next "):
            date_str = date_str[5:]
            is_next = True
        else:
            is_next = False
        
        # Normalize the date string
        date_str = date_str.strip().lower()
        
        if date_str in weekday_mapping:
            # Exact match for a weekday
            target_weekday = weekday_mapping[date_str]
            days_ahead = (target_weekday - today.weekday() + 7) % 7
            if days_ahead == 0 and not is_next:
                # If today is the target day, return today
                return today
            else:
                # Calculate the target date
                target_date = today + timedelta(days=days_ahead + (7 if is_next else 0))
                return target_date.replace(hour=self.business_hours_start, minute=0, second=0, microsecond=0)
        
        return None
    
    def _parse_time_ranges(self, time_str: str) -> List[Tuple[time, time]]:
        """Parse time range expressions (e.g., '9am-11am')"""
        time_ranges = []
        time_str = time_str.strip().lower()
        
        # Split by comma for multiple ranges
        for time_range in time_str.split(","):
            time_range = time_range.strip()
            if "-" in time_range:
                start_end = time_range.split("-")
                if len(start_end) == 2:
                    start_time = self._parse_time_expression(start_end[0].strip())
                    end_time = self._parse_time_expression(start_end[1].strip())
                    if start_time and end_time:
                        time_ranges.append((start_time, end_time))
            else:
                # Single time point (e.g., "2pm")
                single_time = self._parse_time_expression(time_range)
                if single_time:
                    time_ranges.append((single_time, single_time))
        
        return time_ranges
    
    def _parse_time_expression(self, time_expr: str) -> time:
        """Parse a time expression (e.g., '2pm', '14:00')"""
        time_expr = time_expr.strip().lower()
        
        try:
            if "am" in time_expr or "pm" in time_expr:
                # 12-hour format with am/pm
                return datetime.strptime(time_expr, "%I:%M %p").time()
            else:
                # 24-hour format
                return datetime.strptime(time_expr, "%H:%M").time()
        except ValueError:
            return None
    
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