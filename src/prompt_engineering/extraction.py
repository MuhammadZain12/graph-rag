from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Entity(BaseModel):
    id: str = Field(..., description="Unique identifier for the entity (e.g., 'department::computer_science', 'person::john_doe')")
    type: str = Field(..., description="Type of the entity (e.g., 'Department', 'DegreeProgram', 'Person', 'EligibilityCriteria')")
    name: str = Field(..., description="Human-readable name of the entity")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes and properties of the entity")

class Relationship(BaseModel):
    source: str = Field(..., description="ID of the source entity")
    target: str = Field(..., description="ID of the target entity")
    type: str = Field(..., description="Type of relationship (UPPER_SNAKE_CASE)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Attributes of the relationship")

class GraphData(BaseModel):
    nodes: List[Entity] = Field(..., description="List of nodes (entities) found in the text")
    edges: List[Relationship] = Field(..., description="List of edges (relationships) between nodes")
