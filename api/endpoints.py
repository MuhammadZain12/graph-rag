"""
API endpoints for the Graph RAG system.
"""
from fastapi import APIRouter, HTTPException
from langchain_core.prompts import PromptTemplate

from .schemas import ChatRequest, ChatResponse, HealthResponse
from src.llm import get_llm_client
from src.llm.graph_client import GraphManager
from src.llm.hybrid_search import HybridRetriever
from src.llm.guardrail import DepartmentGuardrail, GuardrailResult
from config.settings import settings


router = APIRouter()

# Singleton components (initialized at startup for performance)
_llm = None
_retriever = None
_guardrail = None
_chain = None


def get_components():
    """Get initialized components (lazy init if not pre-warmed)."""
    global _llm, _retriever, _guardrail, _chain
    
    if _llm is None:
        init_components()
    
    return _llm, _retriever, _guardrail, _chain


def init_components():
    """Initialize all components. Call at startup for pre-warming."""
    global _llm, _retriever, _guardrail, _chain
    
    print("[API] Initializing LLM client...")
    _llm = get_llm_client()
    
    print("[API] Initializing Graph Manager & Hybrid Retriever...")
    graph_manager = GraphManager()
    _retriever = HybridRetriever(graph_manager)
    
    # Only init guardrail if enabled
    if settings.env.enable_guardrail:
        print("[API] Initializing Guardrail...")
        _guardrail = DepartmentGuardrail(llm=_llm)
    else:
        print("[API] Guardrail DISABLED")
        _guardrail = None
    
    print("[API] Building LLM chain...")
    template = settings.prompts.rag_prompt
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    _chain = prompt | _llm
    
    print("[API] All components initialized!")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for department-related questions.
    
    1. Guardrail checks if question is department-related (if enabled)
    2. Retrieves context via Hybrid RAG
    3. Generates answer using LLM
    """
    llm, retriever, guardrail, chain = get_components()
    
    # 1. Guardrail Check (if enabled)
    if guardrail is not None:
        guardrail_result = guardrail.check(request.question)
        
        if not guardrail_result.is_allowed:
            return ChatResponse(
                answer="I only answer questions about UET Lahore departments, programs, and faculty. " + guardrail_result.reason,
                is_department_related=False,
                guardrail_reason=guardrail_result.reason,
                sources=[]
            )
    else:
        # Guardrail disabled - always allow
        guardrail_result = GuardrailResult(is_allowed=True, reason="Guardrail disabled")
    
    # 2. Hybrid Retrieval
    try:
        context = retriever.search(request.question, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")
    
    if not context:
        return ChatResponse(
            answer="I couldn't find any relevant information in the documents.",
            is_department_related=True,
            guardrail_reason=guardrail_result.reason,
            sources=[]
        )
    
    # 3. Generate Answer
    try:
        response = chain.invoke({"context": context, "question": request.question})
        
        # Robustly extract content
        content = ""
        raw_content = response.content if hasattr(response, "content") else response
        
        if isinstance(raw_content, list):
            # Handle list of blocks (e.g. from Gemini: [{'type': 'text', 'text': '...'}])
            for block in raw_content:
                if isinstance(block, dict) and "text" in block:
                    content += block["text"]
                elif isinstance(block, str):
                    content += block
        else:
            content = str(raw_content)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    # Extract source chunk IDs from context
    sources = []
    for line in context.split("\n"):
        if "ID:" in line:
            try:
                chunk_id = line.split("ID: ")[1].split(",")[0]
                sources.append(chunk_id)
            except:
                pass
    
    return ChatResponse(
        answer=content,
        is_department_related=True,
        guardrail_reason=guardrail_result.reason,
        sources=sources
    )
