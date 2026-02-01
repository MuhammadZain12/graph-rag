"""
API package initialization.
"""
from fastapi import FastAPI
from .endpoints import router

app = FastAPI(
    title="Graph RAG API",
    description="RAG system for UET Lahore Prospectus using Neo4j Knowledge Graph",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1", tags=["chat"])
