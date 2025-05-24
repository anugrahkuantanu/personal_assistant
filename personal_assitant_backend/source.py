"""
Enhanced Meeting Scheduler Agent with Autonomous Workflow Management
Handles complex meeting scheduling flows with prompt chaining and decision-making
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass, asdict
from app.agents.base_agent import BaseAgent
from app.tools.onecom_tools import OneComTools
import uuid

class WorkflowState(Enum):
    """Workflow states for meeting scheduling"""
    INTENT_ANALYSIS = "intent_analysis"
    INFO_EXTRACTION = "info_extraction"
    CALENDAR_QUERY = "calendar_query"
    SLOT_PROPOSAL = "slot_proposal"
    CLIENT_RESPONSE = "client_response"
    AVAILABILITY_CHECK = "availability_check"
    BOOKING_CONFIRMATION = "booking_confirmation"
    COMPLETED = "completed"
    ERROR = "error"

class SchedulingIntent(Enum):
    """Types of scheduling intents"""
    SCHEDULE_NEW = "schedule_new"
    RESCHEDULE = "reschedule"
    CANCEL = "cancel"
    CONFIRM = "confirm"
    DECLINE = "decline"
    REQUEST_ALTERNATIVES = "request_alternatives"

@dataclass
class MeetingContext:
    """Context for meeting scheduling workflow"""
    workflow_id: str
    state: WorkflowState
    intent: Optional[SchedulingIntent]
    participants: List[str]
    subject: str
    duration: int  # in minutes
    preferred_dates: List[str]
    preferred_times: List[str]
    location: str
    description: str
    proposed_slots: List[Dict]
    selected_slot: Optional[Dict]
    client_email: str
    original_message: str
    workflow_history: List[Dict]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['state'] = self.state.value
        result['intent'] = self.intent.value if self.intent else None
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        return result

class MeetingSchedulerAgent(BaseAgent):
    """
    Advanced Meeting Scheduler Agent with Autonomous Workflow Management
    
    Features:
    - Intent classification and routing
    - Multi-step scheduling workflow
    - Prompt chaining for complex decisions
    - State management and recovery
    - Autonomous email handling
    """
    
    def __init__(self, onecom_tools: OneComTools):
        super().__init__()
        self.onecom_tools = onecom_tools
        self.active_workflows: Dict[str, MeetingContext] = {}
        self.workflow_templates = self._initialize_templates()
    
    async def process_request(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main entry point for processing scheduling requests"""
        try:
            # Step 1: Classify intent and determine if this is scheduling-related
            intent_analysis = await self._analyze_intent(user_message)
            
            if not intent_analysis["is_scheduling"]:
                # Route to general email agent
                return await self._route_to_email_agent(user_message, context)
            
            # Step 2: Check if this is part of an existing workflow
            workflow_id = context.get("workflow_id") if context else None
            existing_workflow = self.active_workflows.get(workflow_id) if workflow_id else None
            
            if existing_workflow:
                # Continue existing workflow
                return await self._continue_workflow(existing_workflow, user_message)
            else:
                # Start new scheduling workflow
                return await self._start_new_workflow(user_message, intent_analysis)
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error in meeting scheduler: {str(e)}",
                "type": "error"
            }
    
    async def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user intent with sophisticated classification"""
        system_prompt = """You are an expert at analyzing communication intent for meeting scheduling and email management.

Analyze the message and determine:
1. Is this scheduling-related? (meeting, appointment, call, etc.)
2. What specific scheduling intent? (schedule new, reschedule, cancel, confirm, decline)
3. What information can be extracted?

Return JSON:
{
    "is_scheduling": true/false,
    "scheduling_intent": "schedule_new|reschedule|cancel|confirm|decline|request_alternatives",
    "confidence": 0.0-1.0,
    "extracted_info": {
        "participants": ["email1", "email2"],
        "subject": "meeting topic",
        "duration": minutes,
        "preferred_dates": ["2025-05-25", "next Friday"],
        "preferred_times": ["2pm", "afternoon", "9-11am"],
        "location": "office/zoom/etc",
        "urgency": "high|medium|low"
    },
    "reasoning": "why you classified this way"
}

Examples:
- "Schedule a meeting with john@company.com about the project" → is_scheduling: true, intent: schedule_new
- "Can we reschedule our Friday meeting?" → is_scheduling: true, intent: reschedule
- "Please send an email to the team about the deadline" → is_scheduling: false
- "I'm available Tuesday afternoon for our call" → is_scheduling: true, intent: confirm"""

        try:
            response = await self._generate_ai_response(system_prompt, message, temperature=0.3)
            return json.loads(response)
        except Exception as e:
            return {
                "is_scheduling": False,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _start_new_workflow(self, message: str, intent_analysis: Dict) -> Dict[str, Any]:
        """Start a new scheduling workflow"""
        workflow_id = str(uuid.uuid4())
        
        # Create workflow context
        extracted_info = intent_analysis.get("extracted_info", {})
        context = MeetingContext(
            workflow_id=workflow_id,
            state=WorkflowState.INFO_EXTRACTION,
            intent=SchedulingIntent(intent_analysis.get("scheduling_intent", "schedule_new")),
            participants=extracted_info.get("participants", []),
            subject=extracted_info.get("subject", "Meeting"),
            duration=extracted_info.get("duration", 60),
            preferred_dates=extracted_info.get("preferred_dates", []),
            preferred_times=extracted_info.get("preferred_times", []),
            location=extracted_info.get("location", ""),
            description="",
            proposed_slots=[],
            selected_slot=None,
            client_email="",
            original_message=message,
            workflow_history=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        # Store workflow
        self.active_workflows[workflow_id] = context
        
        # Execute workflow steps
        return await self._execute_workflow_step(context)
    
    async def _continue_workflow(self, context: MeetingContext, message: str) -> Dict[str, Any]:
        """Continue an existing workflow with new input"""
        context.updated_at = datetime.now(timezone.utc)
        context.workflow_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": context.state.value,
            "input": message
        })
        
        # Process based on current state
        if context.state == WorkflowState.CLIENT_RESPONSE:
            return await self._handle_client_response(context, message)
        else:
            return await self._execute_workflow_step(context)
    
    async def _execute_workflow_step(self, context: MeetingContext) -> Dict[str, Any]:
        """Execute the current workflow step with prompt chaining"""
        try:
            if context.state == WorkflowState.INFO_EXTRACTION:
                return await self._extract_missing_info(context)
            elif context.state == WorkflowState.CALENDAR_QUERY:
                return await self._query_calendar_availability(context)
            elif context.state == WorkflowState.SLOT_PROPOSAL:
                return await self._propose_time_slots(context)
            elif context.state == WorkflowState.AVAILABILITY_CHECK:
                return await self._check_slot_availability(context)
            elif context.state == WorkflowState.BOOKING_CONFIRMATION:
                return await self._confirm_booking(context)
            else:
                context.state = WorkflowState.ERROR
                return {
                    "success": False,
                    "message": f"Unknown workflow state: {context.state}",
                    "type": "error"
                }
        except Exception as e:
            context.state = WorkflowState.ERROR
            return {
                "success": False,
                "message": f"Workflow error: {str(e)}",
                "type": "error"
            }
    
    async def _extract_missing_info(self, context: MeetingContext) -> Dict[str, Any]:
        """Extract or request missing information for scheduling"""
        system_prompt = """You are a meeting scheduler assistant. Analyze the current meeting context and determine what information is missing for scheduling.

Required information:
- At least one participant email
- Meeting subject/topic
- Preferred dates or timeframe
- Duration (default 60 minutes if not specified)

Current context: {context}

Determine:
1. Is information complete enough to proceed?
2. What specific information is missing?
3. How to request missing information naturally?

Return JSON:
{
    "information_complete": true/false,
    "missing_items": ["participant_email", "preferred_dates", "subject"],
    "request_message": "natural message to ask for missing info",
    "can_proceed": true/false,
    "next_action": "calendar_query|request_info|error"
}"""

        prompt = system_prompt.format(context=context.to_dict())
        response = await self._generate_ai_response(prompt, "Analyze meeting context", temperature=0.3)
        analysis = json.loads(response)
        
        if analysis["can_proceed"]:
            context.state = WorkflowState.CALENDAR_QUERY
            return await self._execute_workflow_step(context)
        else:
            return {
                "success": True,
                "message": analysis["request_message"],
                "type": "info_request",
                "workflow_id": context.workflow_id,
                "missing_items": analysis["missing_items"]
            }
    
    async def _query_calendar_availability(self, context: MeetingContext) -> Dict[str, Any]:
        """Query calendar and find available slots"""
        try:
            # Get calendar events
            calendar_data = await self._get_calendar_data(7)  # Next 7 days
            
            # Find free slots using existing calendar analyzer
            from app.agents.schedule_agent import CalendarAnalyzer
            analyzer = CalendarAnalyzer(self.openai_client)
            
            # Convert preferred dates/times to constraints
            date_constraint = context.preferred_dates[0] if context.preferred_dates else None
            time_constraint = context.preferred_times[0] if context.preferred_times else None
            
            free_slots = analyzer._identify_free_time(
                calendar_data, 
                date=date_constraint, 
                time_str=time_constraint, 
                duration=context.duration
            )
            
            if not free_slots:
                return {
                    "success": False,
                    "message": "No available time slots found for your preferences.",
                    "type": "no_availability",
                    "workflow_id": context.workflow_id
                }
            
            # Store proposed slots and move to next state
            context.proposed_slots = [
                {
                    "start": slot["start"].isoformat(),
                    "end": slot["end"].isoformat(),
                    "duration_hours": slot["duration_hours"]
                }
                for slot in free_slots[:3]  # Top 3 slots
            ]
            context.state = WorkflowState.SLOT_PROPOSAL
            
            return await self._execute_workflow_step(context)
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error querying calendar: {str(e)}",
                "type": "error"
            }
    
    async def _propose_time_slots(self, context: MeetingContext) -> Dict[str, Any]:
        """Compose and send email proposing time slots"""
        system_prompt = """You are composing a professional email to propose meeting times.

Meeting context: {context}

Write a professional email that:
1. Introduces the meeting purpose
2. Proposes the available time slots clearly
3. Asks for confirmation or alternative preferences
4. Includes relevant details

Return JSON:
{
    "subject": "email subject",
    "body": "email body text",
    "recipient": "primary recipient email",
    "tone": "professional|friendly|formal"
}

Make it natural and professional."""

        prompt = system_prompt.format(context=context.to_dict())
        response = await self._generate_ai_response(prompt, "Compose meeting proposal", temperature=0.7)
        email_content = json.loads(response)
        
        # Send the email
        if context.participants:
            recipient = context.participants[0]
            context.client_email = recipient
            
            success = await self._send_email(
                recipient,
                email_content["subject"],
                email_content["body"]
            )
            
            if success:
                context.state = WorkflowState.CLIENT_RESPONSE
                return {
                    "success": True,
                    "message": f"Meeting proposal sent to {recipient}",
                    "type": "email_sent",
                    "workflow_id": context.workflow_id,
                    "email_content": email_content,
                    "proposed_slots": context.proposed_slots
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to send meeting proposal email",
                    "type": "error"
                }
        else:
            return {
                "success": False,
                "message": "No recipient specified for meeting proposal",
                "type": "error"
            }
    
    async def _handle_client_response(self, context: MeetingContext, response_message: str) -> Dict[str, Any]:
        """Handle client response to meeting proposal"""
        system_prompt = """Analyze the client's response to a meeting proposal.

Original proposed slots: {proposed_slots}
Client response: "{response}"

Determine:
1. Did they accept one of the proposed slots?
2. Did they propose alternative times?
3. Did they decline or request to reschedule?
4. Extract any specific time preferences

Return JSON:
{
    "response_type": "accept|counter_propose|decline|unclear",
    "selected_slot_index": number or null,
    "alternative_times": ["list of suggested times"],
    "alternative_dates": ["list of suggested dates"],
    "confirmation_needed": true/false,
    "next_action": "book_meeting|propose_alternatives|request_clarification"
}"""

        prompt = system_prompt.format(
            proposed_slots=context.proposed_slots,
            response=response_message
        )
        response = await self._generate_ai_response(prompt, "Analyze client response", temperature=0.3)
        analysis = json.loads(response)
        
        if analysis["response_type"] == "accept" and analysis["selected_slot_index"] is not None:
            # Client accepted a proposed slot
            context.selected_slot = context.proposed_slots[analysis["selected_slot_index"]]
            context.state = WorkflowState.AVAILABILITY_CHECK
            return await self._execute_workflow_step(context)
            
        elif analysis["response_type"] == "counter_propose":
            # Client proposed alternative times
            context.preferred_dates = analysis.get("alternative_dates", [])
            context.preferred_times = analysis.get("alternative_times", [])
            context.state = WorkflowState.CALENDAR_QUERY
            return await self._execute_workflow_step(context)
            
        elif analysis["response_type"] == "decline":
            # Client declined - end workflow
            context.state = WorkflowState.COMPLETED
            return {
                "success": True,
                "message": "Meeting declined by client. Workflow completed.",
                "type": "meeting_declined",
                "workflow_id": context.workflow_id
            }
        else:
            # Unclear response - request clarification
            return {
                "success": True,
                "message": "Could you please clarify your preference for the meeting time? You can select one of the proposed slots or suggest alternative times.",
                "type": "clarification_request",
                "workflow_id": context.workflow_id
            }
    
    async def _check_slot_availability(self, context: MeetingContext) -> Dict[str, Any]:
        """Check if selected slot is still available"""
        try:
            # Re-query calendar to ensure slot is still free
            calendar_data = await self._get_calendar_data(7)
            
            selected_start = datetime.fromisoformat(context.selected_slot["start"])
            selected_end = datetime.fromisoformat(context.selected_slot["end"])
            
            # Check for conflicts
            for event in calendar_data:
                event_start = event.get("start")
                event_end = event.get("end")
                
                if event_start and event_end:
                    # Check for overlap
                    if (selected_start < event_end and selected_end > event_start):
                        # Conflict found
                        return {
                            "success": False,
                            "message": "The selected time slot is no longer available. Please choose another time.",
                            "type": "slot_conflict",
                            "workflow_id": context.workflow_id
                        }
            
            # Slot is available - proceed to booking
            context.state = WorkflowState.BOOKING_CONFIRMATION
            return await self._execute_workflow_step(context)
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error checking availability: {str(e)}",
                "type": "error"
            }
    
    async def _confirm_booking(self, context: MeetingContext) -> Dict[str, Any]:
        """Create calendar event and send confirmation"""
        try:
            selected_start = datetime.fromisoformat(context.selected_slot["start"])
            selected_end = datetime.fromisoformat(context.selected_slot["end"])
            
            # Create calendar event
            event_created = self.onecom_tools.calendar_tool.create_event(
                summary=context.subject,
                start_time=selected_start,
                end_time=selected_end,
                description=f"Meeting scheduled via AI assistant. Participants: {', '.join(context.participants)}",
                location=context.location
            )
            
            if event_created:
                # Send confirmation email
                confirmation_email = await self._compose_confirmation_email(context)
                email_sent = await self._send_email(
                    context.client_email,
                    confirmation_email["subject"],
                    confirmation_email["body"]
                )
                
                context.state = WorkflowState.COMPLETED
                
                return {
                    "success": True,
                    "message": f"Meeting scheduled successfully for {selected_start.strftime('%A, %B %d at %I:%M %p')}",
                    "type": "meeting_confirmed",
                    "workflow_id": context.workflow_id,
                    "meeting_details": {
                        "subject": context.subject,
                        "start": context.selected_slot["start"],
                        "end": context.selected_slot["end"],
                        "participants": context.participants,
                        "location": context.location
                    }
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to create calendar event",
                    "type": "error"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error confirming booking: {str(e)}",
                "type": "error"
            }
    
    async def _compose_confirmation_email(self, context: MeetingContext) -> Dict[str, str]:
        """Compose meeting confirmation email"""
        system_prompt = """Compose a professional meeting confirmation email.

Meeting details: {context}

Write a confirmation email that:
1. Confirms the meeting details
2. Includes date, time, duration
3. Lists participants
4. Includes location/meeting link if applicable
5. Provides contact information for changes

Return JSON:
{
    "subject": "confirmation subject",
    "body": "confirmation email body"
}"""

        prompt = system_prompt.format(context=context.to_dict())
        response = await self._generate_ai_response(prompt, "Compose confirmation", temperature=0.5)
        return json.loads(response)
    
    async def _route_to_email_agent(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Route to general email agent for non-scheduling tasks"""
        from app.agents.email_agent import EmailAgent
        email_agent = EmailAgent(self.onecom_tools)
        return await email_agent.process_request(message, context)
    
    async def _get_calendar_data(self, days_ahead: int = 7) -> List[Dict]:
        """Get calendar data from OneComTools"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.onecom_tools.calendar_tool.get_events,
            days_ahead
        )
    
    async def _send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email using OneComTools"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.onecom_tools.email_tool.send_email,
            to, subject, body
        )
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize email templates for different scenarios"""
        return {
            "meeting_proposal": """
Subject: Meeting Request: {subject}

Hi {name},

I'd like to schedule a meeting about {subject}. 

Here are some time slots that work on my end:
{time_slots}

Please let me know which time works best for you, or suggest alternative times if none of these work.

Best regards
            """,
            "meeting_confirmation": """
Subject: Meeting Confirmed: {subject}

Hi {name},

This confirms our meeting:

📅 Date: {date}
🕐 Time: {time}
⏱️ Duration: {duration}
📍 Location: {location}

Looking forward to our discussion.

Best regards
            """,
            "meeting_reschedule": """
Subject: Meeting Reschedule Request: {subject}

Hi {name},

I need to reschedule our meeting originally planned for {original_time}.

Here are some alternative times:
{alternative_slots}

Please let me know what works best for you.

Best regards
            """
        }
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get current status of a workflow"""
        workflow = self.active_workflows.get(workflow_id)
        return workflow.to_dict() if workflow else None
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24):
        """Clean up old completed workflows"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        to_remove = [
            wid for wid, workflow in self.active_workflows.items()
            if workflow.state in [WorkflowState.COMPLETED, WorkflowState.ERROR]
            and workflow.updated_at < cutoff
        ]
        for wid in to_remove:
            del self.active_workflows[wid]

