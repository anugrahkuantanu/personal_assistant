"""
Fixed Email Agent with Robust JSON Handling
File: app/agents/email_agent.py

Handles all email tasks with improved error handling and proper JSON parsing
"""

import json
import asyncio
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from app.agents.base_agent import BaseAgent
from app.tools.onecom_tools import OneComTools

class EmailAgent(BaseAgent):
    """
    Fixed Email Agent with Robust Error Handling
    """
    
    def __init__(self, onecom_tools: OneComTools):
        super().__init__()
        self.onecom_tools = onecom_tools
        self.email_analyzer = EmailAnalyzer(self.openai_client)
        self.email_composer = EmailComposer(self.openai_client)
    
    async def process_request(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process email request with improved error handling"""
        try:
            print(f"EmailAgent processing: {user_message}")
            
            # Extract conversation context
            conversation_history = context.get("conversation_history", []) if context else []
            user_context = context.get("user_context", {}) if context else {}
            routing_decision = context.get("routing_decision", {}) if context else {}
            
            print(f"User context: {user_context}")
            print(f"Conversation history length: {len(conversation_history)}")
            
            # Analyze email task with robust fallback
            email_task_analysis = await self._analyze_email_task_with_robust_fallback(
                user_message, conversation_history, user_context
            )
            print(f"Email task analysis: {email_task_analysis}")
            
            # Route based on task type with context
            task_type = email_task_analysis["task_type"]
            
            if task_type == "read_emails":
                return await self._handle_read_emails_with_context(
                    user_message, email_task_analysis, conversation_history
                )
            elif task_type == "analyze_emails":
                return await self._handle_analyze_emails_with_context(
                    user_message, email_task_analysis, conversation_history
                )
            elif task_type in ["compose_email", "send_email"]:
                return await self._handle_compose_and_send_with_context(
                    user_message, email_task_analysis, conversation_history, user_context
                )
            else:
                return {
                    "success": False,
                    "message": "I couldn't understand what you want to do with emails.",
                    "type": "error"
                }
                
        except Exception as e:
            print(f"Error in EmailAgent: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "message": f"Error processing email request: {str(e)}",
                "type": "error"
            }
    
    async def _analyze_email_task_with_robust_fallback(self, message: str, 
                                                      conversation_history: List[Dict], 
                                                      user_context: Dict) -> Dict[str, Any]:
        """Analyze email task with robust fallback logic"""
        
        # First try keyword-based analysis
        keyword_analysis = self._analyze_email_with_keywords(message, user_context)
        
        try:
            # Try AI analysis
            ai_analysis = await self._try_ai_email_analysis(message, conversation_history, user_context)
            
            # If AI analysis succeeds and makes sense, use it
            if ai_analysis and ai_analysis.get("task_type") and ai_analysis.get("confidence", 0) > 0.5:
                print("Using AI email analysis result")
                return ai_analysis
            else:
                print("AI email analysis failed or low confidence, using keyword fallback")
                return keyword_analysis
                
        except Exception as e:
            print(f"AI email analysis failed: {e}, using keyword fallback")
            return keyword_analysis
    
    def _analyze_email_with_keywords(self, message: str, user_context: Dict) -> Dict[str, Any]:
        """Robust keyword-based email analysis"""
        message_lower = message.lower()
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        
        # Get preferred emails from context
        preferred_emails = user_context.get("preferred_emails", [])
        
        # Determine task type
        if any(word in message_lower for word in ["send", "email", "ask", "tell", "message", "write"]):
            task_type = "send_email"
        elif any(word in message_lower for word in ["read", "check", "show", "view"]):
            task_type = "read_emails"
        elif any(word in message_lower for word in ["analyze", "summarize", "urgent"]):
            task_type = "analyze_emails"
        else:
            task_type = "send_email"  # Default for email agent
        
        # Determine recipient
        recipient = None
        if emails:
            recipient = emails[0]
        elif preferred_emails:
            recipient = preferred_emails[-1]  # Most recent
        
        # Determine email category
        email_category = "general"
        if any(word in message_lower for word in ["meeting", "schedule", "appointment", "time"]):
            email_category = "scheduling"
        elif any(word in message_lower for word in ["work", "project", "business"]):
            email_category = "work"
        
        # Extract subject hint
        subject_hint = "Follow-up"
        if "meeting" in message_lower:
            subject_hint = "Meeting Confirmation"
        elif "time" in message_lower or "available" in message_lower:
            subject_hint = "Time Confirmation"
        
        return {
            "task_type": task_type,
            "email_category": email_category,
            "confidence": 0.8,
            "is_follow_up": True,
            "extracted_info": {
                "recipient": recipient,
                "recipient_name": self._extract_name_from_recipient(recipient) if recipient else "",
                "subject_hint": subject_hint,
                "email_purpose": f"Follow-up communication based on: {message[:100]}",
                "tone": "professional",
                "urgency": "medium",
                "content_details": [message]
            },
            "context_info_used": {
                "recipients_from_context": preferred_emails,
                "topics_from_conversation": ["meeting discussion"],
                "preferences_applied": ["preferred email address"]
            },
            "needs_calendar_context": email_category == "scheduling",
            "follow_up_action": "Send follow-up email"
        }
    
    def _extract_name_from_recipient(self, email: str) -> str:
        """Extract name from email address"""
        if not email:
            return ""
        
        # Extract name from email (before @)
        name_part = email.split('@')[0]
        
        # Handle common patterns
        if '.' in name_part:
            name_parts = name_part.split('.')
            return ' '.join(part.capitalize() for part in name_parts)
        else:
            return name_part.capitalize()
    
    async def _try_ai_email_analysis(self, message: str, conversation_history: List[Dict], 
                                    user_context: Dict) -> Optional[Dict[str, Any]]:
        """Try AI analysis with proper error handling"""
        
        history_text = self._format_conversation_history(conversation_history)
        
        system_prompt = f"""Analyze this email request and return ONLY valid JSON.

CONVERSATION CONTEXT:
{history_text}

USER CONTEXT:
{json.dumps(user_context, indent=2)}

CURRENT MESSAGE: "{message}"

Return ONLY this JSON structure (no other text):
{{
    "task_type": "send_email|read_emails|analyze_emails|compose_email",
    "email_category": "scheduling|work|personal|general",
    "confidence": 0.0-1.0,
    "is_follow_up": true/false,
    "extracted_info": {{
        "recipient": "email address",
        "recipient_name": "recipient name",
        "subject_hint": "email subject hint",
        "email_purpose": "purpose of email",
        "tone": "professional|casual|friendly",
        "urgency": "high|medium|low"
    }},
    "needs_calendar_context": true/false
}}"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Analyze the email request."}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            response_content = response.choices[0].message.content.strip()
            print(f"AI Email Analysis Response: {response_content}")
            
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
            print(f"AI email analysis error: {e}")
            return None
    
    async def _handle_read_emails_with_context(self, message: str, task_analysis: Dict, 
                                              conversation_history: List[Dict]) -> Dict[str, Any]:
        """Handle reading emails with conversation context"""
        try:
            email_data = await self._get_email_data(10)
            
            if not email_data:
                return {
                    "success": True,
                    "message": "You have no recent emails in your inbox.",
                    "type": "no_emails",
                    "emails_count": 0
                }
            
            # Format emails for display
            formatted_emails = []
            for i, email in enumerate(email_data, 1):
                subject = email.get('subject', 'No subject')
                sender = email.get('from', 'Unknown sender')
                date = email.get('date', 'Unknown date')
                
                # Shorten long fields
                if len(subject) > 50:
                    subject = subject[:47] + "..."
                if len(sender) > 30:
                    sender = sender[:27] + "..."
                
                email_text = f"{i}. **{subject}**\n   From: {sender}\n   Date: {date}"
                formatted_emails.append(email_text)
            
            emails_display = '\n\n'.join(formatted_emails)
            
            return {
                "success": True,
                "message": f"Here are your {len(email_data)} most recent emails:\n\n{emails_display}",
                "type": "emails_displayed",
                "emails_count": len(email_data),
                "emails": email_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error reading emails: {str(e)}",
                "type": "error"
            }
    
    async def _handle_analyze_emails_with_context(self, message: str, task_analysis: Dict, 
                                                 conversation_history: List[Dict]) -> Dict[str, Any]:
        """Handle analyzing emails with conversation context"""
        try:
            email_data = await self._get_email_data(15)
            
            if not email_data:
                return {
                    "success": True,
                    "message": "You have no recent emails to analyze.",
                    "type": "no_emails",
                    "emails_count": 0
                }
            
            return await self.email_analyzer.analyze_emails_with_context(
                email_data, message, task_analysis, conversation_history
            )
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error analyzing emails: {str(e)}",
                "type": "error"
            }
    
    async def _handle_compose_and_send_with_context(self, message: str, task_analysis: Dict, 
                                                   conversation_history: List[Dict], 
                                                   user_context: Dict) -> Dict[str, Any]:
        """Handle composing and sending emails with full context"""
        try:
            extracted_info = task_analysis.get("extracted_info", {})
            
            # Determine recipient
            recipient = extracted_info.get("recipient")
            if not recipient:
                preferred_emails = user_context.get("preferred_emails", [])
                if preferred_emails:
                    recipient = preferred_emails[-1]  # Most recent
            
            if not recipient:
                return {
                    "success": False,
                    "message": "Please specify the recipient's email address.",
                    "type": "missing_recipient"
                }
            
            # Get calendar context if needed
            calendar_data = []
            if task_analysis.get("needs_calendar_context"):
                calendar_data = await self._get_calendar_context()
            
            # Compose email with robust error handling
            email_content = await self.email_composer.compose_email_with_robust_handling(
                message, task_analysis, conversation_history, user_context, calendar_data
            )
            
            if not email_content.get("success"):
                return email_content
            
            # Send email
            return await self._send_email_with_context(recipient, email_content["email_data"], task_analysis)
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error composing/sending email: {str(e)}",
                "type": "error"
            }
    
    async def _send_email_with_context(self, recipient: str, email_data: Dict, task_analysis: Dict) -> Dict[str, Any]:
        """Send email with context-aware confirmation message"""
        try:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                self.onecom_tools.email_tool.send_email,
                recipient,
                email_data["subject"],
                email_data["body"]
            )
            
            if success:
                success_message = f"✅ Email sent successfully to {recipient}"
                
                if task_analysis.get("is_follow_up"):
                    success_message += " (following up on our conversation)"
                
                return {
                    "success": True,
                    "message": success_message,
                    "type": "email_sent",
                    "recipient": recipient,
                    "subject": email_data["subject"],
                    "email_content": email_data
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to send email. Please check your email settings.",
                    "type": "email_send_failed",
                    "email_content": email_data
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error sending email: {str(e)}",
                "type": "error"
            }
    
    async def _get_email_data(self, limit: int = 10) -> List[Dict]:
        """Get email data from OneComTools"""
        try:
            loop = asyncio.get_event_loop()
            emails = await loop.run_in_executor(
                None,
                self.onecom_tools.email_tool.read_emails,
                limit
            )
            
            if not emails:
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
            print(f"Error reading emails: {str(e)}")
            return []
    
    async def _get_calendar_context(self) -> List[Dict]:
        """Get calendar context for scheduling-related emails"""
        try:
            from app.agents.scheduling_agent import SchedulingAgent
            scheduling_agent = SchedulingAgent(self.onecom_tools)
            return await scheduling_agent._get_calendar_data(7)
        except Exception as e:
            print(f"Error getting calendar context: {e}")
            return []
    
    def _format_conversation_history(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for LLM"""
        if not conversation_history:
            return "No previous conversation."
        
        formatted = []
        for msg in conversation_history[-8:]:
            role = msg.get("role", "").title()
            content = msg.get("content", "")[:200]
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)


class EmailAnalyzer:
    """Enhanced email analyzer with conversation context"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def analyze_emails_with_context(self, emails: List[Dict], user_message: str, 
                                        task_analysis: Dict, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Analyze emails with conversation context"""
        
        if not emails:
            return {
                "success": True,
                "message": "You have no recent emails to analyze.",
                "type": "email_analysis",
                "emails_count": 0
            }
        
        # Simple analysis without complex AI for reliability
        urgent_emails = self._identify_urgent_emails(emails)
        
        # Format emails for display
        formatted_emails = []
        for i, email in enumerate(emails[:5], 1):  # Show top 5
            subject = email.get('subject', 'No subject')
            sender = email.get('from', 'Unknown sender')
            date = email.get('date', 'Unknown date')
            
            if len(subject) > 50:
                subject = subject[:47] + "..."
            if len(sender) > 30:
                sender = sender[:27] + "..."
            
            urgency_marker = " ⚠️" if any(urgent['subject'] == email.get('subject') for urgent in urgent_emails) else ""
            
            email_text = f"{i}. **{subject}**{urgency_marker}\n   From: {sender}\n   Date: {date}"
            formatted_emails.append(email_text)
        
        emails_display = '\n\n'.join(formatted_emails)
        
        analysis_message = f"Email Analysis ({len(emails)} total emails):\n\n{emails_display}"
        
        if urgent_emails:
            urgent_list = '\n'.join([f"• {urgent['subject']} - {urgent['reason']}" for urgent in urgent_emails[:3]])
            analysis_message += f"\n\n🔔 **Urgent/Important emails:**\n{urgent_list}"
        
        return {
            "success": True,
            "message": analysis_message,
            "type": "email_analysis",
            "emails_count": len(emails),
            "urgent_emails": urgent_emails
        }
    
    def _identify_urgent_emails(self, emails: List[Dict]) -> List[Dict]:
        """Identify urgent emails using keywords"""
        urgent_keywords = ['urgent', 'asap', 'immediate', 'deadline', 'important', 'meeting', 'today', 'tomorrow']
        urgent_emails = []
        
        for email in emails:
            content = (email.get('subject', '') + ' ' + email.get('body', '')).lower()
            reasons = []
            
            for keyword in urgent_keywords:
                if keyword in content:
                    reasons.append(f"contains '{keyword}'")
            
            if reasons:
                urgent_emails.append({
                    'subject': email.get('subject'),
                    'from': email.get('from'),
                    'reason': ', '.join(reasons)
                })
        
        return urgent_emails


class EmailComposer:
    """Enhanced email composer with robust JSON handling"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    async def compose_email_with_robust_handling(self, user_message: str, task_analysis: Dict, 
                                               conversation_history: List[Dict], user_context: Dict,
                                               calendar_data: List[Dict] = None) -> Dict[str, Any]:
        """Compose email with robust error handling"""
        
        extracted_info = task_analysis.get("extracted_info", {})
        email_category = task_analysis.get("email_category", "general")
        
        # Try AI composition first, fall back to template-based
        try:
            return await self._try_ai_composition(
                user_message, task_analysis, conversation_history, user_context, calendar_data
            )
        except Exception as e:
            print(f"AI composition failed: {e}, using template fallback")
            return self._compose_with_template(
                user_message, extracted_info, conversation_history, user_context
            )
    
    async def _try_ai_composition(self, user_message: str, task_analysis: Dict, 
                                conversation_history: List[Dict], user_context: Dict,
                                calendar_data: List[Dict]) -> Dict[str, Any]:
        """Try AI composition with proper JSON handling"""
        
        extracted_info = task_analysis.get("extracted_info", {})
        history_text = self._format_conversation_for_composition(conversation_history)
        
        # Use simpler prompt to avoid JSON issues
        system_prompt = f"""Compose a professional email based on the conversation context.

CONVERSATION CONTEXT:
{history_text}

EMAIL REQUEST: "{user_message}"
RECIPIENT: {extracted_info.get('recipient', 'Unknown')}
SUBJECT HINT: {extracted_info.get('subject_hint', 'Follow-up')}
EMAIL PURPOSE: {extracted_info.get('email_purpose', 'Follow-up communication')}

Write a professional email that:
1. References the conversation naturally
2. Is clear and concise
3. Has an appropriate subject line
4. Uses professional tone

Respond ONLY with this exact format:
SUBJECT: [subject line here]
BODY: [email body here]"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Compose the email."}
                ],
                temperature=0.6,
                max_tokens=600
            )
            
            response_content = response.choices[0].message.content.strip()
            print(f"AI Email Composition Response: {response_content[:200]}...")
            
            # Parse the simple format
            subject_match = re.search(r'SUBJECT:\s*(.+)', response_content, re.IGNORECASE)
            body_match = re.search(r'BODY:\s*(.+)', response_content, re.IGNORECASE | re.DOTALL)
            
            if subject_match and body_match:
                subject = subject_match.group(1).strip()
                body = body_match.group(1).strip()
                
                return {
                    "success": True,
                    "message": "Email composed with AI assistance.",
                    "email_data": {
                        "subject": subject,
                        "body": body,
                        "tone": "professional",
                        "email_type": "ai_composed"
                    }
                }
            else:
                raise Exception("Could not parse AI response format")
            
        except Exception as e:
            print(f"AI composition error: {e}")
            raise e
    
    def _compose_with_template(self, user_message: str, extracted_info: Dict, 
                             conversation_history: List[Dict], user_context: Dict) -> Dict[str, Any]:
        """Compose email using template fallback"""
        
        recipient_name = extracted_info.get("recipient_name", "")
        subject_hint = extracted_info.get("subject_hint", "Follow-up")
        
        # Determine if this is about meeting confirmation
        message_lower = user_message.lower()
        is_meeting_confirmation = any(word in message_lower for word in ["meeting", "time", "available", "schedule"])
        
        if is_meeting_confirmation:
            subject = f"Meeting Time Confirmation - {subject_hint}"
            
            greeting = f"Hi {recipient_name}," if recipient_name else "Hi,"
            
            body = f"""{greeting}

I wanted to follow up regarding our meeting discussion about the latest AI technology.

Could you please confirm if you're available for the meeting at the proposed time? I want to make sure the timing works for you.

Looking forward to hearing from you.

Best regards"""
        
        else:
            subject = f"Follow-up - {subject_hint}"
            
            greeting = f"Hi {recipient_name}," if recipient_name else "Hi,"
            
            body = f"""{greeting}

I wanted to follow up on our recent conversation.

{user_message.capitalize()[:200]}

Please let me know your thoughts.

Best regards"""
        
        return {
            "success": True,
            "message": "Email composed using template.",
            "email_data": {
                "subject": subject,
                "body": body,
                "tone": "professional",
                "email_type": "template_composed"
            }
        }
    
    def _format_conversation_for_composition(self, conversation_history: List[Dict]) -> str:
        """Format conversation for email composition"""
        if not conversation_history:
            return "No previous conversation."
        
        formatted = []
        for msg in conversation_history[-4:]:  # Last 4 messages for context
            role = msg.get("role", "").title()
            content = msg.get("content", "")[:150]  # Shorter to avoid JSON issues
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)