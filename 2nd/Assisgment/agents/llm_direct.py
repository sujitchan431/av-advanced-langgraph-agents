from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)

_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable assistant. Answer clearly and accurately using your training knowledge."),
    ("human", "{query}"),
])


def llm_direct_node(state: AgentState) -> AgentState:
    query = state["query"]
    print(f"\n[LLM Direct] Answering: '{query[:60]}'")

    response = _llm.invoke(_prompt.format_messages(query=query))
    print("[LLM Direct] Response generated")
    return {**state, "llm_response": response.content, "sources": ["Claude (training knowledge)"]}
