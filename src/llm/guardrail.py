"""
LLM-based Guardrail for department-related questions.
Uses structured output to classify if a question is within scope.
"""
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from src.llm import get_llm_client


class GuardrailResult(BaseModel):
    """Structured output for guardrail check."""
    is_allowed: bool = Field(description="True if the question is about UET departments, programs, faculty, or admissions")
    reason: str = Field(description="Brief explanation of the classification")


GUARDRAIL_PROMPT = """You are a classifier for a UET Lahore Prospectus Q&A system.

Determine if the following question is about:
- UET Lahore departments
- Degree programs offered
- Faculty members
- Admission requirements or eligibility
- Campus locations or facilities - Department Related Only

If the question is about ANY of the above topics, it IS ALLOWED.
If the question is completely unrelated (e.g., weather, politics, other universities), it is NOT ALLOWED.

Question: {question}

Respond with your classification."""


class DepartmentGuardrail:
    """
    LLM-based guardrail that checks if a question is department-related.
    """
    
    def __init__(self, llm=None):
        """
        Initialize guardrail with optional custom LLM.
        If not provided, uses the default from get_llm_client().
        """
        self.llm = llm or get_llm_client()
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template=GUARDRAIL_PROMPT
        )
        
        # Create chain with structured output
        self.chain = self.prompt | self.llm.with_structured_output(GuardrailResult)
    
    def check(self, question: str) -> GuardrailResult:
        """
        Check if a question is allowed (department-related).
        
        Args:
            question: The user's question
            
        Returns:
            GuardrailResult with is_allowed and reason
        """
        try:
            result = self.chain.invoke({"question": question})
            return result
        except Exception as e:
            # On error, default to allowing (fail-open for better UX)
            print(f"Guardrail error: {e}")
            return GuardrailResult(
                is_allowed=True,
                reason="Guardrail check failed, allowing by default"
            )
