"""
Base Agent Class - Foundation for all AI agents
"""

from typing import Dict, Any
from openai import AsyncOpenAI
from app.core.config import settings

class BaseAgent:
    """Base class for all AI agents"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def process_request(self, user_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Base method for processing requests - to be implemented by child classes"""
        raise NotImplementedError("Child classes must implement process_request method")
    
    async def _generate_ai_response(self, system_prompt: str, user_prompt: str, 
                                  temperature: float = 0.5, max_tokens: int = 500) -> str:
        """Common method for generating AI responses"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating AI response: {str(e)}") 