"""
vision-link-ai-agent / workspace / state_schema.py
Pydantic v2 state schema shared across all agents and the LangGraph orchestrator.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

class PatientContext(BaseModel):
    patient_id: str
    age: int
    gender: str
    symptoms: List[str]
    medical_history: List[str] = Field(default_factory=list)
    location: Optional[str] = None          # ISO 3166-2 region code
    language: str = "en"


class ClinicalOutput(BaseModel):
    diagnosis_candidates: List[str]
    urgency_level: Literal["low", "medium", "high", "critical"]
    recommended_actions: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    raw_llm_response: str = ""


class LocalizationOutput(BaseModel):
    translated_diagnosis: List[str]
    translated_actions: List[str]
    locale_notes: str = ""
    target_language: str = "en"


class ValidationOutput(BaseModel):
    passed: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    retry_count: int = 0


class EvolutionMetrics(BaseModel):
    """Live fitness metrics tracked per pipeline run."""
    run_id: str
    latency_ms: float = 0.0
    error_rate: float = 0.0
    throughput_rps: float = 0.0
    fitness_score: float = 0.0          # composite: lower latency + lower error_rate + higher throughput
    variant_hash: str = ""              # SHA-256 of the winning code variant
    promoted_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Master pipeline state  (used as LangGraph TypedDict-compatible dict)
# ---------------------------------------------------------------------------

class PipelineState(BaseModel):
    """
    Single source of truth flowing through the LangGraph StateGraph.
    Every node reads from and writes back to this object.
    """
    # Input
    patient: PatientContext

    # Intermediate outputs
    clinical: Optional[ClinicalOutput] = None
    localization: Optional[LocalizationOutput] = None
    validation: Optional[ValidationOutput] = None

    # Orchestration control
    stage: Literal[
        "clinical", "localization", "validation",
        "self_healing", "evolve", "finalized", "failed"
    ] = "clinical"
    retry_count: int = 0
    max_retries: int = 3
    failed_agent: Optional[str] = None
    error_log: List[str] = Field(default_factory=list)

    # Evolution subsystem
    evolution: Optional[EvolutionMetrics] = None
    evolution_enabled: bool = True

    # Final result
    final_response: Optional[Dict[str, Any]] = None
    run_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"))

    class Config:
        use_enum_values = True
