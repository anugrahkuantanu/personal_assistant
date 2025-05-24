"""
OneCom Tools - Email and Calendar Integration for one.com
Provides tools for email and calendar operations
"""

import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from datetime import datetime, timedelta
import uuid

# Optional imports for calendar functionality
try:
    import caldav
    from icalendar import Calendar, Event
    CALENDAR_SUPPORT = True
except ImportError:
    CALENDAR_SUPPORT = False
    print("⚠️  Calendar libraries not installed. Run: pip install caldav icalendar")


class EmailTool:
    """Tool for handling email operations"""
    
    def __init__(self, email_address, password):
        self.email = email_address
        self.password = password
        self.imap_server = "imap.one.com"
        self.smtp_server = "send.one.com"
        
    def read_emails(self, limit=10):
        try:
            # Connect to IMAP server with explicit SSL
            mail = imaplib.IMAP4_SSL(self.imap_server, port=993)
            
            # Login with proper error handling
            try:
                mail.login(self.email, self.password)
            except imaplib.IMAP4.error as e:
                print(f"IMAP Login failed: {str(e)}")
                return []
            
            # Select inbox with error handling
            try:
                mail.select('inbox')
            except imaplib.IMAP4.error as e:
                print(f"Failed to select inbox: {str(e)}")
                return []
            
            # Search for emails
            status, data = mail.search(None, 'ALL')
            if status != 'OK':
                print(f"Search failed with status: {status}")
                return []
                
            email_ids = data[0].split()
            emails = []
            
            # Get last 'limit' emails
            for email_id in email_ids[-limit:]:
                try:
                    status, data = mail.fetch(email_id, '(RFC822)')
                    if status == 'OK' and data and data[0]:
                        raw_email = data[0][1]
                        if not raw_email:
                            continue
                            
                        email_message = email.message_from_bytes(raw_email)
                        if not email_message:
                            continue
                        
                        # Decode subject safely
                        subject = email_message.get('subject', '')
                        if subject:
                            try:
                                decoded_subject = decode_header(subject)[0]
                                if isinstance(decoded_subject[0], bytes):
                                    subject = decoded_subject[0].decode(decoded_subject[1] or 'utf-8')
                                else:
                                    subject = decoded_subject[0]
                            except:
                                subject = str(subject)
                        
                        # Get email body
                        body = self._get_email_body(email_message)
                        
                        # Get sender safely
                        sender = email_message.get('from', '')
                        if sender:
                            try:
                                decoded_sender = decode_header(sender)[0]
                                if isinstance(decoded_sender[0], bytes):
                                    sender = decoded_sender[0].decode(decoded_sender[1] or 'utf-8')
                                else:
                                    sender = decoded_sender[0]
                            except:
                                sender = str(sender)
                        
                        # Get date safely
                        date = email_message.get('date', '')
                        if date:
                            try:
                                # Try to parse the date
                                date_obj = email.utils.parsedate_to_datetime(date)
                                date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                date = str(date)
                        
                        emails.append({
                            'id': email_id.decode() if email_id else '',
                            'subject': subject or 'No Subject',
                            'from': sender or 'Unknown Sender',
                            'to': email_message.get('to', '') or '',
                            'date': date or '',
                            'body': body[:500] + '...' if len(body) > 500 else body
                        })
                except Exception as e:
                    print(f"Error processing email {email_id}: {e}")
                    continue
            
            mail.close()
            mail.logout()
            return emails
            
        except Exception as e:
            print(f"Error reading emails: {e}")
            return []
    
    def _get_email_body(self, email_message):
        """Extract email body handling multipart messages"""
        if not email_message:
            return ""
            
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                if not part:
                    continue
                    
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode('utf-8', errors='ignore')
                            break
                    except:
                        continue
        else:
            try:
                payload = email_message.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')
                else:
                    body = str(email_message.get_payload() or "")
            except:
                body = str(email_message.get_payload() or "")
        
        return body
    
    def send_email(self, to_address, subject, body, html_body=None):
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email
            msg['To'] = to_address
            msg['Subject'] = subject
            
            # Add plain text part
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML part if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Connect to SMTP server with explicit SSL
            server = smtplib.SMTP_SSL(self.smtp_server, port=465)
            
            # Login with proper error handling
            try:
                server.login(self.email, self.password)
            except smtplib.SMTPAuthenticationError as e:
                print(f"SMTP Login failed: {str(e)}")
                return False
            
            # Send email with error handling
            try:
                server.send_message(msg)
                server.quit()
                return True
            except smtplib.SMTPException as e:
                print(f"Failed to send email: {str(e)}")
                return False
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False


class CalendarTool:
    """Tool for handling calendar operations"""
    
    def __init__(self, email_address, password):
        self.username = email_address
        self.password = password
        # Set the CalDAV URL
        self.caldav_url = f"https://caldav.one.com/calendars/users/{email_address}/calendar"
        
        if not CALENDAR_SUPPORT:
            print("⚠️  Calendar libraries not available. Install with: pip install caldav icalendar")
        elif not self.caldav_url:
            print("⚠️  No CalDAV URL provided. Calendar features disabled.")
            print("   To enable: Get URL from webmail → Calendar → Show CalDAV url")
        
    def get_events(self, days_ahead=7):
        if not CALENDAR_SUPPORT:
            print("❌ Calendar libraries not installed.")
            return []
            
        if not self.caldav_url:
            print("❌ No CalDAV URL configured.")
            return []
            
        try:
            client = caldav.DAVClient(
                url=self.caldav_url,
                username=self.username, 
                password=self.password
            )
            
            principal = client.principal()
            calendars = principal.calendars()
            
            events = []
            if len(calendars) > 0:
                calendar = calendars[0]
                
                # Get events for specified period
                start_date = datetime.now()
                end_date = start_date + timedelta(days=days_ahead)
                
                calendar_events = calendar.date_search(start_date, end_date)
                
                for event in calendar_events:
                    try:
                        # Parse the iCalendar data properly
                        cal = Calendar.from_ical(event.data)
                        for component in cal.walk():
                            if component.name == "VEVENT":
                                events.append({
                                    'summary': str(component.get('summary', 'No Title')),
                                    'description': str(component.get('description', '')),
                                    'start': component.get('dtstart').dt if component.get('dtstart') else None,
                                    'end': component.get('dtend').dt if component.get('dtend') else None,
                                    'location': str(component.get('location', '')),
                                    'uid': str(component.get('uid', ''))
                                })
                    except Exception as e:
                        print(f"Error parsing event: {e}")
                        continue
                    
            return events
            
        except Exception as e:
            print(f"Error getting calendar events: {e}")
            return []
    
    def create_event(self, summary, start_time, end_time, description="", location=""):
        if not CALENDAR_SUPPORT:
            print("❌ Calendar libraries not installed.")
            return False
            
        if not self.caldav_url:
            print("❌ No CalDAV URL configured.")
            return False
            
        try:
            print(f"Creating calendar event: {summary}")
            print(f"Start time: {start_time}")
            print(f"End time: {end_time}")
            
            client = caldav.DAVClient(
                url=self.caldav_url,
                username=self.username, 
                password=self.password
            )
            
            print("Connected to CalDAV server")
            
            principal = client.principal()
            calendars = principal.calendars()
            
            if not calendars:
                print("No calendars found")
                return False
                
            calendar = calendars[0]
            print(f"Using calendar: {calendar.url}")
            
            # Create event with proper timezone handling
            event = Event()
            event.add('summary', summary)
            event.add('dtstart', start_time)
            event.add('dtend', end_time)
            if description:
                event.add('description', description)
            if location:
                event.add('location', location)
            event.add('status', 'CONFIRMED')
            event.add('uid', str(uuid.uuid4()) + '@onecom-tool')
            
            # Create calendar object
            cal = Calendar()
            cal.add('prodid', '-//OneComAI Tool//EN')
            cal.add('version', '2.0')
            cal.add_component(event)
            
            # Save the event
            try:
                calendar.save_event(cal)
                print("Event saved successfully")
                return True
            except Exception as e:
                print(f"Error saving event: {str(e)}")
                return False
            
        except Exception as e:
            print(f"Error creating calendar event: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False


class OneComTools:
    """Combined tools for email and calendar operations"""
    
    def __init__(self, email_address, password):
        """
        Initialize OneCom Tools
        
        Args:
            email_address: Your one.com email
            password: Your one.com password
        """
        self.email_tool = EmailTool(email_address, password)
        self.calendar_tool = CalendarTool(email_address, password)
        self.email_address = email_address
    
    def get_daily_summary(self):
        try:
            # Get recent emails
            emails = self.email_tool.read_emails(limit=5)
            
            # Get today's and tomorrow's calendar events
            events = self.calendar_tool.get_events(days_ahead=2)
            
            # Process and analyze
            summary = self.analyze_data(emails, events)
            return summary
            
        except Exception as e:
            print(f"Error getting daily summary: {e}")
            return "Error getting daily summary"
    
    def schedule_meeting(self, subject, start_time, end_time, attendee_email, location="", description=""):
        try:
            # Create calendar event
            event_created = self.calendar_tool.create_event(
                subject, start_time, end_time, description, location
            )
            
            # Send email invitation
            meeting_body = f"""Meeting Invitation: {subject}

Time: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}
Location: {location if location else 'To be determined'}

Description:
{description if description else 'No additional details provided.'}

This meeting has been added to the calendar."""

            email_sent = self.email_tool.send_email(
                attendee_email, 
                f"Meeting Invitation: {subject}", 
                meeting_body
            )
            
            return event_created and email_sent
            
        except Exception as e:
            print(f"Error scheduling meeting: {e}")
            return False
    
    def analyze_data(self, emails, events):
        try:
            email_summary = f"Recent emails: {len(emails)}"
            if emails:
                recent_subjects = [e['subject'][:50] + '...' if len(e['subject']) > 50 else e['subject'] for e in emails[:3]]
                email_summary += f"\nRecent subjects: {', '.join(recent_subjects)}"
            
            event_summary = f"Upcoming events: {len(events)}"
            if events:
                upcoming_events = [e['summary'][:30] + '...' if len(e['summary']) > 30 else e['summary'] for e in events[:3]]
                event_summary += f"\nUpcoming: {', '.join(upcoming_events)}"
            
            return f"{email_summary}\n\n{event_summary}"
            
        except Exception as e:
            print(f"Error analyzing data: {e}")
            return "Error analyzing data"
    

