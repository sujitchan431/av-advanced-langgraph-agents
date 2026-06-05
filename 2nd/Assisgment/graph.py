from langgraph.graph import StateGraph, END

from state import AgentState
from agents import (
    router_node,
    web_research_node,
    rag_node,
    llm_direct_node,
    summarization_node,
)


def _route_decision(state: AgentState) -> str:
    route = state.get("route", "llm")
    return {"web": "web_research", "rag": "rag", "llm": "llm_direct"}.get(route, "llm_direct")


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("web_research", web_research_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("llm_direct", llm_direct_node)
    workflow.add_node("summarization", summarization_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        _route_decision,
        {
            "web_research": "web_research",
            "rag": "rag",
            "llm_direct": "llm_direct",
        },
    )

    workflow.add_edge("web_research", "summarization")
    workflow.add_edge("rag", "summarization")
    workflow.add_edge("llm_direct", "summarization")
    workflow.add_edge("summarization", END)

    return workflow.compile()
