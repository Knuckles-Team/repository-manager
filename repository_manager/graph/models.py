from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    type: str  # e.g. "Function", "Class", "Concept"
    name: str
    repo_path: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str  # e.g. "CALLS", "IMPORTS", "semantically_similar_to"
    provenance: str  # "EXTRACTED", "INFERRED", "AMBIGUOUS"
    confidence: float = 1.0
    rationale: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphGroup(BaseModel):
    name: str
    repositories: List[str] = Field(default_factory=list)


class GraphReport(BaseModel):
    nodes_processed: int
    edges_processed: int
    groups_analyzed: List[str]
    god_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    surprising_connections: List[Dict[str, Any]] = Field(default_factory=list)
    token_savings_estimate: Optional[str] = None
