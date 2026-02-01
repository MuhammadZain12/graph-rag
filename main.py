"""
Main entry point for the Graph RAG API.
Run with: python main.py
Or: uvicorn main:app --reload
"""
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm components at startup."""
    from api.endpoints import init_components
    print("[Startup] Pre-warming components...")
    init_components()
    print("[Startup] Ready to serve requests!")
    yield
    print("[Shutdown] Cleaning up...")


# Import app after defining lifespan to avoid circular import
from api.endpoints import router

app = FastAPI(
    title="Graph RAG API",
    description="RAG system for UET Lahore Prospectus using Neo4j Knowledge Graph",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1", tags=["chat"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000
    )
