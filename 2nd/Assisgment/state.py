from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    query: str
    route: Optional[str]          # "llm" | "web" | "rag"
    web_results: Optional[str]
    rag_results: Optional[str]
    llm_response: Optional[str]
    final_summary: Optional[str]
    sources: Optional[List[str]]
