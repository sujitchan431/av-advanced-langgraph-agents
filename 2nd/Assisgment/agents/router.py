from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query router for a research system. Classify the user query into exactly one of:

- "web"  → needs current/real-time info: news, prices, recent events, live data, today's info
- "rag"  → needs info from our knowledge base: ML algorithms, NLP, LangChain/LangGraph, Python data science libraries, deep learning, transformers, embeddings
- "llm"  → everything else: reasoning, math, creative writing, general knowledge, definitions

Reply with ONE word only: web, rag, or llm"""),
    ("human", "Query: {query}"),
])


def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    response = _llm.invoke(_prompt.format_messages(query=query))
    route = response.content.strip().lower().split()[0]

    if route not in ("web", "rag", "llm"):
        route = "llm"

    print(f"\n[Router] '{query[:60]}' → {route.upper()}")
    return {**state, "route": route}
