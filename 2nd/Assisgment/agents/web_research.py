import os

from state import AgentState


def _tavily_search(query: str, max_results: int = 4) -> tuple[str, list[str]]:
    from langchain_community.tools.tavily_search import TavilySearchResults
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke(query)
    formatted = "\n\n".join(
        f"Source: {r['url']}\n{r['content']}" for r in results
    )
    sources = [r["url"] for r in results]
    return formatted, sources


def _ddg_search(query: str, max_results: int = 4) -> tuple[str, list[str]]:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    formatted = "\n\n".join(
        f"Source: {r['href']}\n{r['title']}\n{r['body']}" for r in results
    )
    sources = [r["href"] for r in results]
    return formatted, sources


def web_research_node(state: AgentState) -> AgentState:
    query = state["query"]
    print(f"\n[Web Research] Searching: '{query[:60]}'")

    try:
        if os.getenv("TAVILY_API_KEY"):
            content, sources = _tavily_search(query)
            print(f"[Web Research] Tavily returned {len(sources)} results")
        else:
            content, sources = _ddg_search(query)
            print(f"[Web Research] DuckDuckGo returned {len(sources)} results")

        return {**state, "web_results": content, "sources": sources}

    except Exception as exc:
        print(f"[Web Research] ERROR: {exc}")
        return {**state, "web_results": f"Search failed: {exc}", "sources": []}
