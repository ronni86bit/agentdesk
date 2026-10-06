"""Retriever Agent: executes retrieval for a ResearchPlan.

Architecture note
-----------------
Actual searches are performed deterministically by the Python tools in
``tools/qdrant_tool.py`` and ``tools/tavily_tool.py``. The LLM is never asked
to search or to invent sources, so the evidence bundle cannot contain
fabricated results. The CrewAI agent definition below exists so the future
orchestration layer (Planner -> Retriever -> Writer -> Critic) can attach
these same Python functions as CrewAI tools without changing this interface.
"""

from crewai import Agent

from agents.schemas import EvidenceBundle, ResearchPlan
from config import get_config
from tools.retrieval import retrieve_evidence

RETRIEVER_ROLE = "Research Evidence Retrieval Specialist"
RETRIEVER_GOAL = (
    "Gather relevant, high-quality evidence for every research sub-question "
    "from the local knowledge base and live web search."
)
RETRIEVER_BACKSTORY = (
    "You are an expert research librarian who locates authoritative sources. "
    "You execute structured searches against a local Qdrant vector knowledge "
    "base and the Tavily web search API, then normalize and deduplicate what "
    "you find. You never fabricate sources or summarize findings; you only "
    "return retrieved evidence."
)


def build_retriever_agent() -> Agent:
    """CrewAI agent definition for future orchestration use."""
    from crewai import LLM

    cfg = get_config()
    llm = LLM(
        model=f"groq/{cfg.groq_model}",
        api_key=cfg.groq_api_key,
        temperature=0.2,
    )
    return Agent(
        role=RETRIEVER_ROLE,
        goal=RETRIEVER_GOAL,
        backstory=RETRIEVER_BACKSTORY,
        llm=llm,
        verbose=False,
    )


def retrieve_for_plan(plan: ResearchPlan, mode: str = "both") -> EvidenceBundle:
    """Execute retrieval for every sub-question in a research plan.

    `mode` selects the evidence sources: "both" (default), "web", or "kb".
    """
    return retrieve_evidence(
        queries=list(plan.sub_questions),
        original_query=plan.original_query,
        mode=mode,
    )
