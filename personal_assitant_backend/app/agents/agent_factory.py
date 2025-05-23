"""
Agent Factory - Creates and manages different types of agents
"""

from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.agents.email_agent import EmailAgent
from app.agents.schedule_agent import ScheduleAgent
from app.tools.onecom_tools import OneComTools

class AgentFactory:
    """Factory class for creating and managing agents"""
    
    def __init__(self, onecom_tools: OneComTools):
        self.onecom_tools = onecom_tools
        self._agents: Dict[str, BaseAgent] = {}
    
    def get_agent(self, agent_type: str) -> BaseAgent:
        """Get or create an agent of the specified type"""
        if agent_type not in self._agents:
            self._agents[agent_type] = self._create_agent(agent_type)
        return self._agents[agent_type]
    
    def _create_agent(self, agent_type: str) -> BaseAgent:
        """Create a new agent instance"""
        if agent_type == "email":
            return EmailAgent(self.onecom_tools)
        elif agent_type == "schedule":
            return ScheduleAgent(self.onecom_tools)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    async def process_request(self, user_message: str, agent_type: str, 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a request using the specified agent"""
        agent = self.get_agent(agent_type)
        return await agent.process_request(user_message, context) 