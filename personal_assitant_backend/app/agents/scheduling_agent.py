"""
Fixed Scheduling Agent with Robust Error Handling
File: app/agents/scheduling_agent.py

Handles all scheduling with improved error handling and proper task execution
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
from enum import Enum
from app.agents.base_agent import BaseAgent
from app.tools.onecom_tools import OneComTools
import uuid
import re

class WorkflowState(Enum):
    """States for complex scheduling workflows"""
    ANALYZING = "analyzing"
    COLLECTING_INFO = "collecting_info"
    FINDING_SLOTS = "finding_slots"
    SENDING_PROPOSAL = "sending_proposal"
    WAITING_RESPONSE = "waiting_response"
    BOOKING = "booking"
    COMPLETED = "completed"
    ERROR = "error"

class SchedulingWorkflow:
    """Enhanced workflow tracking with conversation context"""
    
    def __init__(self, workflow_id: str, original_message: str):
        self.workflow_id = workflow_id
        self.original_message = original_message
        self.state = WorkflowState.ANALYZING
        self.participants = []
        self.subject = "Meeting"
        self.duration = 60
        self.preferred_dates = []
        self.preferred_times = []
        self.location = ""
        self.proposed_slots = []
        self.conversation_context = []
        self.user_preferences = {}
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict:
        return {
            "workflow_id": self.workflow_id,
            "state": self.state.value,
            "participants": self.participants,
            "subject": self.subject,
            "duration": self.duration,
            "preferred_dates": self.preferred_dates,
            "preferred_times": self.preferred_times,
            "location": self.location,
            "proposed_slots": self.proposed_slots,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class SchedulingAgent(BaseAgent):
    """
    Fixed Scheduling Agent with Robust Error Handling
    """
    
    def __init__(self, onecom_tools: OneComTools):
        super().__init__()
        self.onecom_tools = onecom_tools
        self.calendar_analyzer = CalendarAnalyzer(self.openai_client)
        self.active_workflows: Dict[str, SchedulingWorkflow] = {}
    
    async def process_request(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process scheduling request with improved error handling"""
        try:
            print(f"SchedulingAgent processing: {user_message}")
            
            # Extract conversation context
            conversation_history = context.get("conversation_history", []) if context else []
            user_context = context.get("user_context", {}) if context else {}
            
            print(f"User context: {user_context}")
            print(f"Conversation history length: {len(conversation_history)}")
            
            # Enhanced task analysis with fallback logic
            task_analysis = await self._analyze_scheduling_task_with_robust_fallback(
                user_message, conversation_history, user_context
            )
            print(f"Task analysis: {task_analysis}")
            
            # Route based on task analysis
            if task_analysis["task_type"] in ["schedule_meeting", "create_event"]:
                return await self._handle_meeting_scheduling_with_context(
                    user_message, task_analysis, conversation_history, user_context
                )
            elif task_analysis["task_type"] == "view_calendar":
                return await self._handle_calendar_view_with_context(
                    user_message, task_analysis, conversation_history, user_context
                )
            elif task_analysis["task_type"] == "find_free_time":
                return await self._handle_find_free_time_with_context(
                    user_message, task_analysis, conversation_history, user_context
                )
            else:
                # Default to calendar view for unknown tasks
                return await self._handle_calendar_view_with_context(
                    user_message, task_analysis, conversation_history, user_context
                )
                
        except Exception as e:
            print(f"Error in SchedulingAgent: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "message": f"Error in scheduling: {str(e)}",
                "type": "error"
            }
    
    async def _analyze_scheduling_task_with_robust_fallback(self, message: str, 
                                                          conversation_history: List[Dict], 
                                                          user_context: Dict) -> Dict[str, Any]:
        """Analyze scheduling task with robust fallback logic"""
        
        # First, try keyword-based analysis for immediate classification
        keyword_analysis = self._analyze_with_keywords(message)
        
        try:
            # Try AI analysis
            ai_analysis = await self._try_ai_analysis(message, conversation_history, user_context)
            
            # If AI analysis succeeds and makes sense, use it
            if ai_analysis and ai_analysis.get("task_type") and ai_analysis.get("confidence", 0) > 0.5:
                print("Using AI analysis result")
                return ai_analysis
            else:
                print("AI analysis failed or low confidence, using keyword fallback")
                return keyword_analysis
                
        except Exception as e:
            print(f"AI analysis failed: {e}, using keyword fallback")
            return keyword_analysis
    
    def _analyze_with_keywords(self, message: str) -> Dict[str, Any]:
        """Robust keyword-based analysis as fallback"""
        message_lower = message.lower()
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        
        # Determine task type
        if any(word in message_lower for word in ["schedule", "meeting", "appointment", "book"]):
            if emails or "send" in message_lower or "email" in message_lower:
                task_type = "schedule_meeting"
            else:
                task_type = "create_event"
        elif any(word in message_lower for word in ["view", "show", "see", "check", "what's", "calendar"]):
            task_type = "view_calendar"
        elif any(word in message_lower for word in ["free", "available", "slot", "time"]):
            task_type = "find_free_time"
        else:
            # Default based on presence of emails
            task_type = "schedule_meeting" if emails else "view_calendar"
        
        # Extract basic information
        extracted_info = {
            "participants": emails,
            "subject": self._extract_subject_keywords(message),
            "duration": self._extract_duration_keywords(message),
            "dates": self._extract_date_keywords(message),
            "times": self._extract_time_keywords(message)
        }
        
        return {
            "task_type": task_type,
            "complexity": "complex" if emails else "simple",
            "confidence": 0.8,
            "is_follow_up": False,
            "extracted_info": extracted_info,
            "context_info_used": {},
            "missing_info": self._identify_missing_info_from_keywords(extracted_info),
            "next_action": f"execute_{task_type}",
            "user_intent": f"User wants to {task_type.replace('_', ' ')}"
        }
    
    def _extract_subject_keywords(self, message: str) -> str:
        """Extract meeting subject from keywords"""
        # Look for common patterns
        patterns = [
            r"meeting (?:with \w+ )?(?:to|about) (.+?)(?:\.|$|\s+(?:for|at|on))",
            r"discuss (?:about )?(.+?)(?:\.|$|\s+(?:for|at|on))",
            r"talk about (.+?)(?:\.|$|\s+(?:for|at|on))",
            r"(?:regarding|about) (.+?)(?:\.|$|\s+(?:for|at|on))"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                # Clean up common endings
                subject = re.sub(r'\s+(?:please|and|send|email).*$', '', subject, flags=re.IGNORECASE)
                return subject
        
        return "Meeting"
    
    def _extract_duration_keywords(self, message: str) -> int:
        """Extract duration from keywords"""
        # Look for duration patterns
        patterns = [
            r"(\d+)\s*hour",
            r"(\d+)\s*hr",
            r"(\d+)\s*minutes?",
            r"(\d+)\s*min"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                if "hour" in match.group(0).lower() or "hr" in match.group(0).lower():
                    return value * 60
                else:
                    return value
        
        return 60  # Default 1 hour
    
    def _extract_date_keywords(self, message: str) -> List[str]:
        """Extract dates from keywords"""
        dates = []
        message_lower = message.lower()
        
        # Common date keywords
        if "today" in message_lower:
            dates.append("today")
        elif "tomorrow" in message_lower:
            dates.append("tomorrow")
        elif "monday" in message_lower:
            dates.append("monday")
        elif "tuesday" in message_lower:
            dates.append("tuesday")
        elif "wednesday" in message_lower:
            dates.append("wednesday")
        elif "thursday" in message_lower:
            dates.append("thursday")
        elif "friday" in message_lower:
            dates.append("friday")
        
        return dates
    
    def _extract_time_keywords(self, message: str) -> List[str]:
        """Extract times from keywords"""
        times = []
        
        # Time patterns
        time_patterns = [
            r"\d{1,2}:\d{2}\s*(?:am|pm)",
            r"\d{1,2}\s*(?:am|pm)",
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            times.extend(matches)
        
        # Common time keywords
        message_lower = message.lower()
        if "morning" in message_lower:
            times.append("morning")
        elif "afternoon" in message_lower:
            times.append("afternoon")
        elif "evening" in message_lower:
            times.append("evening")
        
        return times
    
    def _identify_missing_info_from_keywords(self, extracted_info: Dict) -> List[str]:
        """Identify missing information"""
        missing = []
        
        if not extracted_info.get("participants"):
            missing.append("participant email")
        
        return missing
    
    async def _try_ai_analysis(self, message: str, conversation_history: List[Dict], 
                              user_context: Dict) -> Optional[Dict[str, Any]]:
        """Try AI analysis with proper error handling"""
        
        history_text = self._format_conversation_history(conversation_history)
        
        system_prompt = f"""Analyze this scheduling request and return ONLY valid JSON.

CONVERSATION CONTEXT:
{history_text}

USER CONTEXT:
{json.dumps(user_context, indent=2)}

CURRENT MESSAGE: "{message}"

TASK TYPES:
1. "schedule_meeting" - Schedule meeting with others (needs email coordination)
2. "create_event" - Create simple calendar event
3. "view_calendar" - View/check calendar
4. "find_free_time" - Find available time slots

IMPORTANT DATE/TIME EXTRACTION:
- Extract ONLY what user explicitly mentions
- For day names like "Tuesday", use just "Tuesday" (don't add specific dates)
- For corrections like "not Monday, Tuesday", extract "Tuesday"
- Don't hallucinate specific dates unless user provides them

Return ONLY this JSON structure (no other text):
{{
    "task_type": "schedule_meeting|create_event|view_calendar|find_free_time",
    "complexity": "simple|complex",
    "confidence": 0.0-1.0,
    "is_follow_up": true/false,
    "extracted_info": {{
        "participants": ["email addresses"],
        "subject": "meeting topic",
        "duration": 60,
        "dates": ["ONLY what user said - like Tuesday, not Tuesday May 27"],
        "times": ["times mentioned"],
        "location": "location"
    }},
    "context_info_used": {{}},
    "missing_info": ["what's missing"],
    "next_action": "what to do next",
    "user_intent": "what user wants"
}}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Analyze the scheduling request."}
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            response_content = response.choices[0].message.content.strip()
            print(f"AI Analysis Response: {response_content}")
            
            # Clean the response
            if not response_content.startswith('{'):
                start_idx = response_content.find('{')
                if start_idx != -1:
                    response_content = response_content[start_idx:]
                else:
                    return None
            
            if not response_content.endswith('}'):
                end_idx = response_content.rfind('}')
                if end_idx != -1:
                    response_content = response_content[:end_idx + 1]
                else:
                    return None
            
            return json.loads(response_content)
            
        except Exception as e:
            print(f"AI analysis error: {e}")
            return None
    
    async def _handle_meeting_scheduling_with_context(self, message: str, task_analysis: Dict, 
                                                     conversation_history: List[Dict], 
                                                     user_context: Dict) -> Dict[str, Any]:
        """Handle meeting scheduling with full workflow"""
        try:
            workflow_id = str(uuid.uuid4())
            workflow = SchedulingWorkflow(workflow_id, message)
            
            # Extract information from analysis
            extracted_info = task_analysis.get("extracted_info", {})
            
            workflow.participants = extracted_info.get("participants", [])
            workflow.subject = extracted_info.get("subject", "Meeting")
            workflow.duration = extracted_info.get("duration", 60)
            workflow.preferred_dates = extracted_info.get("dates", [])
            workflow.preferred_times = extracted_info.get("times", [])
            workflow.location = extracted_info.get("location", "")
            
            # Store workflow
            self.active_workflows[workflow_id] = workflow
            
            # Check if we have enough info to proceed
            if not workflow.participants:
                return {
                    "success": False,
                    "message": "I need the participant's email address to schedule the meeting. Please provide it.",
                    "type": "missing_participant",
                    "workflow_id": workflow_id
                }
            
            # Parse specific date/time from user preferences
            selected_slot = await self._parse_and_create_time_slot(workflow)
            
            if not selected_slot:
                # Fallback to finding free slots
                workflow.state = WorkflowState.FINDING_SLOTS
                calendar_data = await self._get_calendar_data(7)
                calendar_data = self._ensure_timezone_aware(calendar_data)
                
                free_slots = self._identify_free_time_simple(
                    calendar_data,
                    duration_hours=workflow.duration / 60
                )
                
                if not free_slots:
                    return {
                        "success": False,
                        "message": "No available time slots found. Please suggest specific dates/times.",
                        "type": "no_availability",
                        "workflow_id": workflow_id
                    }
                
                # Select best slot (first available)
                selected_slot = free_slots[0]
            
            # Create calendar event
            workflow.state = WorkflowState.BOOKING
            event_created = self.onecom_tools.calendar_tool.create_event(
                summary=workflow.subject,
                start_time=selected_slot["start"],
                end_time=selected_slot["end"],
                description=f"Meeting scheduled via AI assistant\nOriginal request: {message}",
                location=workflow.location
            )
            
            if not event_created:
                return {
                    "success": False,
                    "message": "Failed to create calendar event. Please check calendar settings.",
                    "type": "calendar_error",
                    "workflow_id": workflow_id
                }
            
            # Send email confirmation
            workflow.state = WorkflowState.SENDING_PROPOSAL
            recipient = workflow.participants[0]
            
            email_subject = f"Meeting Scheduled: {workflow.subject}"
            email_body = f"""Hi,

I've scheduled our meeting about {workflow.subject.lower()}.

Meeting Details:
• Date & Time: {selected_slot["start"].strftime('%A, %B %d at %I:%M %p')}
• Duration: {workflow.duration} minutes
• Topic: {workflow.subject}"""
            
            if workflow.location:
                email_body += f"\n• Location: {workflow.location}"
            
            email_body += "\n\nLooking forward to our discussion!\n\nBest regards"
            
            # Send email
            email_sent = self.onecom_tools.email_tool.send_email(
                recipient,
                email_subject,
                email_body
            )
            
            if email_sent:
                workflow.state = WorkflowState.COMPLETED
                return {
                    "success": True,
                    "message": f"✅ Meeting scheduled successfully!\n\n📅 {workflow.subject}\n🕐 {selected_slot['start'].strftime('%A, %B %d at %I:%M %p')}\n📧 Confirmation sent to {recipient}",
                    "type": "meeting_scheduled",
                    "workflow_id": workflow_id,
                    "event_details": {
                        "subject": workflow.subject,
                        "start": selected_slot["start"].isoformat(),
                        "end": selected_slot["end"].isoformat(),
                        "participant": recipient,
                        "location": workflow.location
                    }
                }
            else:
                return {
                    "success": True,
                    "message": f"✅ Meeting created in calendar but failed to send email confirmation to {recipient}. Please send manually.",
                    "type": "partial_success",
                    "workflow_id": workflow_id,
                    "event_details": {
                        "subject": workflow.subject,
                        "start": selected_slot["start"].isoformat(),
                        "end": selected_slot["end"].isoformat(),
                        "participant": recipient,
                        "location": workflow.location
                    }
                }
                
        except Exception as e:
            print(f"Error in meeting scheduling: {e}")
            return {
                "success": False,
                "message": f"Error scheduling meeting: {str(e)}",
                "type": "error"
            }
    
    def _identify_free_time_simple(self, events: List[Dict], duration_hours: float = 1.0) -> List[Dict]:
        """Simple free time identification"""
        free_slots = []
        
        # Business hours: 9 AM to 5 PM, Monday to Friday
        business_start = 9
        business_end = 17
        working_days = [0, 1, 2, 3, 4]  # Monday to Friday
        
        # Get current date and next 7 days
        current_date = datetime.now(timezone.utc).date()
        
        for i in range(7):
            check_date = current_date + timedelta(days=i)
            
            # Skip weekends
            if check_date.weekday() not in working_days:
                continue
            
            # Get events for this day
            day_events = [
                e for e in events 
                if e.get('start') and e['start'].date() == check_date
            ]
            
            # Sort events by start time
            day_events.sort(key=lambda e: e['start'])
            
            # Check slots throughout the day
            for hour in range(business_start, business_end - int(duration_hours)):
                slot_start = datetime.combine(check_date, datetime.min.time()).replace(
                    hour=hour, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                )
                slot_end = slot_start + timedelta(hours=duration_hours)
                
                # Check if this slot conflicts with any event
                conflicts = False
                for event in day_events:
                    event_start = event['start']
                    event_end = event.get('end', event_start + timedelta(hours=1))
                    
                    # Check for overlap
                    if (slot_start < event_end and slot_end > event_start):
                        conflicts = True
                        break
                
                if not conflicts:
                    free_slots.append({
                        "start": slot_start,
                        "end": slot_end,
                        "duration_hours": duration_hours
                    })
                    
                    # Return first available slot for simplicity
                    if len(free_slots) >= 1:
                        return free_slots
        
        return free_slots
    
    async def _handle_calendar_view_with_context(self, message: str, task_analysis: Dict, 
                                               conversation_history: List[Dict], 
                                               user_context: Dict) -> Dict[str, Any]:
        """Handle calendar viewing with context"""
        try:
            calendar_days = self._extract_days_from_message(message)
            calendar_data = await self._get_calendar_data(calendar_days)
            calendar_data = self._ensure_timezone_aware(calendar_data)
            
            return await self.calendar_analyzer.analyze_schedule_with_context(
                calendar_data, message, task_analysis, conversation_history
            )
        except Exception as e:
            return {
                "success": False,
                "message": f"Error viewing calendar: {str(e)}",
                "type": "error"
            }
    
    async def _handle_find_free_time_with_context(self, message: str, task_analysis: Dict, 
                                                conversation_history: List[Dict], 
                                                user_context: Dict) -> Dict[str, Any]:
        """Handle finding free time with context"""
        try:
            calendar_data = await self._get_calendar_data(7)
            calendar_data = self._ensure_timezone_aware(calendar_data)
            
            extracted_info = task_analysis.get("extracted_info", {})
            duration = extracted_info.get("duration", 60) / 60  # Convert to hours
            
            free_slots = self._identify_free_time_simple(calendar_data, duration)
            
            if not free_slots:
                return {
                    "success": True,
                    "message": "No free time slots found matching your criteria. Would you like me to check different dates or times?",
                    "type": "no_free_slots"
                }
            
            # Format slots
            formatted_slots = []
            for i, slot in enumerate(free_slots[:5], 1):
                formatted = slot["start"].strftime("%A, %B %d at %I:%M %p")
                duration_text = f"({slot['duration_hours']:.1f} hours available)"
                formatted_slots.append(f"{i}. {formatted} {duration_text}")
            
            slots_text = "\n".join(formatted_slots)
            
            return {
                "success": True,
                "message": f"Here are your available free time slots:\n\n{slots_text}",
                "type": "free_slots_found",
                "free_slots": [
                    {
                        "start": slot["start"].isoformat(),
                        "end": slot["end"].isoformat(),
                        "duration_hours": slot["duration_hours"]
                    }
                    for slot in free_slots[:5]
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error finding free time: {str(e)}",
                "type": "error"
            }
    
    def _format_conversation_history(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for LLM"""
        if not conversation_history:
            return "No previous conversation."
        
        formatted = []
        for msg in conversation_history[-6:]:
            role = msg.get("role", "").title()
            content = msg.get("content", "")[:150]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def _extract_days_from_message(self, message: str) -> int:
        """Extract how many days to look ahead"""
        message_lower = message.lower()
        if "today" in message_lower:
            return 1
        elif "tomorrow" in message_lower:
            return 2
        elif "week" in message_lower:
            return 7
        elif "month" in message_lower:
            return 30
        else:
            return 7
    
    async def _get_calendar_data(self, days_ahead: int = 7) -> List[Dict]:
        """Get calendar data"""
        try:
            loop = asyncio.get_event_loop()
            events = await loop.run_in_executor(
                None,
                self.onecom_tools.calendar_tool.get_events,
                days_ahead
            )
            return events or []
        except Exception as e:
            print(f"Error getting calendar data: {e}")
            return []
    
    def _ensure_timezone_aware(self, events: List[Dict]) -> List[Dict]:
        """Ensure timezone awareness"""
        for event in events:
            if 'start' in event and event['start'] and not event['start'].tzinfo:
                event['start'] = event['start'].replace(tzinfo=timezone.utc)
            if 'end' in event and event['end'] and not event['end'].tzinfo:
                event['end'] = event['end'].replace(tzinfo=timezone.utc)
        return events
    
    async def _parse_and_create_time_slot(self, workflow: SchedulingWorkflow) -> Optional[Dict]:
        """Parse user preferences and create specific time slot"""
        try:
            # If we have specific dates and times, use them
            if workflow.preferred_dates and workflow.preferred_times:
                date_str = workflow.preferred_dates[0]
                time_str = workflow.preferred_times[0]
                
                print(f"Parsing date: {date_str}, time: {time_str}")
                
                # Parse the date and time
                parsed_datetime = self._parse_date_time_string(date_str, time_str)
                
                if parsed_datetime:
                    end_time = parsed_datetime + timedelta(minutes=workflow.duration)
                    
                    print(f"Parsed datetime: {parsed_datetime}")
                    print(f"End time: {end_time}")
                    
                    return {
                        "start": parsed_datetime,
                        "end": end_time,
                        "duration_hours": workflow.duration / 60
                    }
            
            # If we only have dates, use default business hours
            elif workflow.preferred_dates:
                date_str = workflow.preferred_dates[0]
                
                print(f"Parsing date only: {date_str}")
                
                # Default to 10 AM if no time specified
                parsed_datetime = self._parse_date_time_string(date_str, "10:00 AM")
                
                if parsed_datetime:
                    end_time = parsed_datetime + timedelta(minutes=workflow.duration)
                    
                    print(f"Parsed datetime with default time: {parsed_datetime}")
                    
                    return {
                        "start": parsed_datetime,
                        "end": end_time,
                        "duration_hours": workflow.duration / 60
                    }
            
            return None
            
        except Exception as e:
            print(f"Error parsing time slot: {e}")
            return None
    
    def _parse_date_time_string(self, date_str: str, time_str: str = None) -> Optional[datetime]:
        """Parse date and time strings into datetime object"""
        try:
            # Clean up the strings
            date_str = date_str.strip()
            if time_str:
                time_str = time_str.strip()
            
            # Handle specific formats
            current_year = datetime.now().year
            
            # Pattern 1: "Tuesday, May 27" or "May 27"
            month_day_pattern = r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s*(\w+\s+\d+)'
            match = re.search(month_day_pattern, date_str, re.IGNORECASE)
            if match:
                month_day = match.group(1)
                date_str = f"{month_day}, {current_year}"
            
            # Pattern 2: Just day name "Tuesday"
            elif date_str.lower() in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                # Find next occurrence of this day
                target_day = date_str.lower()
                days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                target_weekday = days.index(target_day)
                
                current_date = datetime.now().date()
                days_ahead = target_weekday - current_date.weekday()
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                
                target_date = current_date + timedelta(days=days_ahead)
                date_str = target_date.strftime("%Y-%m-%d")
            
            # Combine date and time
            if time_str:
                # Clean time string
                time_str = re.sub(r'(\d+:\d+)\s*-\s*\d+:\d+', r'\1', time_str)  # Remove time ranges
                datetime_str = f"{date_str} {time_str}"
            else:
                datetime_str = date_str
            
            print(f"Attempting to parse: '{datetime_str}'")
            
            # Parse using dateutil (import it locally to avoid dependency issues)
            try:
                from dateutil import parser
                parsed_dt = parser.parse(datetime_str)
            except ImportError:
                # Fallback to manual parsing if dateutil not available
                print("dateutil not available, using manual parsing")
                return self._manual_date_parse(date_str, time_str)
            
            # Ensure timezone awareness
            if not parsed_dt.tzinfo:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            
            return parsed_dt
            
        except Exception as e:
            print(f"Error parsing datetime '{date_str}' + '{time_str}': {e}")
            
            # Fallback: try basic parsing
            try:
                return self._manual_date_parse(date_str, time_str)
                
            except Exception as fallback_error:
                print(f"Fallback parsing also failed: {fallback_error}")
            
            return None
    
    def _manual_date_parse(self, date_str: str, time_str: str = None) -> Optional[datetime]:
        """Manual date parsing fallback"""
        try:
            # Simple fallback for basic formats
            if "tuesday" in date_str.lower():
                # Calculate next Tuesday
                current_date = datetime.now().date()
                days_ahead = 1 - current_date.weekday()  # Tuesday is weekday 1
                if days_ahead <= 0:
                    days_ahead += 7
                
                target_date = current_date + timedelta(days=days_ahead)
                
                # Default time
                hour = 10  # 10 AM default
                if time_str:
                    if "10" in time_str:
                        hour = 10
                    elif "11" in time_str:
                        hour = 11
                    elif "2" in time_str and ("pm" in time_str.lower()):
                        hour = 14
                
                return datetime.combine(target_date, datetime.min.time()).replace(
                    hour=hour, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                )
            
            elif "monday" in date_str.lower():
                # Calculate next Monday
                current_date = datetime.now().date()
                days_ahead = 0 - current_date.weekday()  # Monday is weekday 0
                if days_ahead <= 0:
                    days_ahead += 7
                
                target_date = current_date + timedelta(days=days_ahead)
                
                hour = 10  # Default time
                return datetime.combine(target_date, datetime.min.time()).replace(
                    hour=hour, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
                )
                
            return None
            
        except Exception as e:
            print(f"Manual parsing failed: {e}")
            return None

    def get_active_workflows(self) -> List[Dict]:
        """Get all active workflows"""
        return [workflow.to_dict() for workflow in self.active_workflows.values()]
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24):
        """Clean up old workflows"""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            to_remove = [
                wid for wid, workflow in self.active_workflows.items()
                if workflow.state in [WorkflowState.COMPLETED, WorkflowState.ERROR]
                and workflow.updated_at < cutoff
            ]
            for wid in to_remove:
                del self.active_workflows[wid]
            if to_remove:
                print(f"Cleaned up {len(to_remove)} old scheduling workflows")
        except Exception as e:
            print(f"Error during workflow cleanup: {e}")


class CalendarAnalyzer:
    """Enhanced calendar analyzer with conversation context"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def analyze_schedule_with_context(self, events: List[Dict], user_message: str, 
                                          context: Dict, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Analyze schedule with conversation context"""
        if not events:
            return {
                "success": True,
                "message": "You have no upcoming events in your calendar.",
                "type": "calendar_analysis",
                "events_count": 0
            }
        
        events_text = self._format_events_for_analysis(events)
        
        try:
            # Simple analysis without AI for reliability
            formatted_events = []
            for i, event in enumerate(events, 1):
                start = event.get('start', 'Unknown time')
                summary = event.get('summary', 'No title')
                location = event.get('location', '')
                
                event_text = f"{i}. **{summary}**"
                if start and hasattr(start, 'strftime'):
                    event_text += f" - {start.strftime('%A, %B %d at %I:%M %p')}"
                
                if location:
                    event_text += f" (Location: {location})"
                
                formatted_events.append(event_text)
            
            events_display = '\n'.join(formatted_events)
            
            return {
                "success": True,
                "message": f"Here are your upcoming {len(events)} events:\n\n{events_display}",
                "type": "calendar_analysis",
                "events_count": len(events)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error analyzing calendar: {str(e)}",
                "type": "error"
            }
    
    def _format_events_for_analysis(self, events: List[Dict]) -> str:
        """Format events for analysis"""
        formatted_events = []
        
        for event in events:
            start = event.get('start', 'Unknown time')
            summary = event.get('summary', 'No title')
            location = event.get('location', '')
            
            event_text = f"• {summary}"
            if start and hasattr(start, 'strftime'):
                event_text += f" - {start.strftime('%A, %B %d at %I:%M %p')}"
            
            if location:
                event_text += f" (Location: {location})"
            
            formatted_events.append(event_text)
        
        return '\n'.join(formatted_events)