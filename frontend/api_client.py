"""
API Client for Graph RAG Backend
"""
import requests
from typing import Optional


class GraphRAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    def check_health(self) -> bool:
        """Check if backend is reachable."""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def send_message(self, question: str) -> dict:
        """
        Send a question to the backend.
        
        Returns:
            {
                "answer": str,
                "is_department_related": bool,
                "guardrail_reason": str,
                "sources": list[str]
            }
        """
        response = requests.post(
            f"{self.api_base}/chat",
            json={"question": question},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
