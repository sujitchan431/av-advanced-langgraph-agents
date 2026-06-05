from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)

_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a summarization agent. Produce a well-structured, concise response from the gathered information.

Use this format:

## Summary
A clear, direct answer to the query (2-4 sentences).

## Key Points
- Bullet each major finding or fact
- Keep each point concise

## Source Type
State how the information was retrieved (Web Search / Knowledge Base / LLM Reasoning) and list sources if available.

Stay factual. Do not hallucinate. If info is limited, say so."""),
    ("human", """Query: {query}

Route used: {route}

Gathered information:
{context}

Sources: {sources}"""),
])


def summarization_node(state: AgentState) -> AgentState:
    query = state["query"]
    route = state.get("route", "llm")
    sources = state.get("sources", [])

    if route == "web":
        context = state.get("web_results", "No web results.")
    elif route == "rag":
        context = state.get("rag_results", "No RAG results.")
    else:
        context = state.get("llm_response", "No LLM response.")

    print(f"\n[Summarization] Synthesizing response…")

    response = _llm.invoke(_prompt.format_messages(
        query=query,
        route=route.upper(),
        context=context,
        sources=", ".join(sources) if sources else "None",
    ))

    print("[Summarization] Done")
    return {**state, "final_summary": response.content}
