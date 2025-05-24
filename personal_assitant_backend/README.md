# AI Personal Assistant Backend

A powerful AI-powered personal assistant backend that handles meeting scheduling and email management with autonomous workflow capabilities.

## Features

- **Intelligent Meeting Scheduling**
  - Autonomous workflow management
  - Calendar integration
  - Smart time slot detection
  - Email-based coordination

- **Email Management**
  - Email composition and sending
  - Email reading and summarization
  - Reply handling
  - Priority management

- **Workflow Orchestration**
  - Multi-agent coordination
  - Session management
  - Context preservation
  - Error handling and recovery

## Architecture

The system is built with a modular architecture:

1. **Base Agent**: Common functionality for all agents
2. **Meeting Scheduler Agent**: Handles complex meeting workflows
3. **Email Agent**: Manages email-related tasks
4. **Workflow Orchestrator**: Coordinates between agents and manages sessions

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd personal_assitant_backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Process Request
```http
POST /api/assistant/process
Content-Type: application/json

{
    "message": "Schedule a meeting with John tomorrow at 2 PM",
    "client_id": "user123",
    "context": {
        "preferences": {
            "timezone": "UTC",
            "working_hours": "9-17"
        }
    }
}
```

### Get Session Status
```http
GET /api/assistant/status/{client_id}
```

## Development

### Project Structure
```
personal_assitant_backend/
├── app/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── meeting_scheduler_agent.py
│   │   ├── email_agent.py
│   │   └── workflow_orchestrator.py
│   ├── tools/
│   │   └── onecom_tools.py
│   └── main.py
├── requirements.txt
└── README.md
```

### Adding New Features

1. Create new agent classes in `app/agents/`
2. Add new tools in `app/tools/`
3. Update the workflow orchestrator to handle new capabilities
4. Add new API endpoints in `main.py`

## Testing

Run tests with pytest:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 