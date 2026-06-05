# 10th Assignment — Advanced AI Agents with LangGraph (2 Sub-Assignments)

## Project Overview

Two LangGraph-based agent assignments progressing from a simple tool-use loop to a full multi-agent routing system with web search, RAG, and LLM-direct paths.

## Sub-Assignment Structure

```
10th advanced agents Langraph/
├── 1st/   ← LangGraph Math Agent (tool-use loop)
└── 2nd/   ← Multi-Agent Research & Summarization System (router + 4 agents)
```

---

## Sub-Assignment 1 — LangGraph Math Agent

**File:** `math_agent.ipynb`

### What It Does
LangGraph agent that answers general questions via LLM and math queries via 4 custom tool functions: `plus`, `subtract`, `multiply`, `divide`.

### Graph Structure

```
User Query
    ↓
[agent node] ←─────────────────────────────┐
    │                                       │
    ├── tool_calls? ──YES──► [tools node] ──┘
    │                         (runs math tools)
    └── no tool_calls? ──► END
```

### Key Components

| Component | Description |
|-----------|-------------|
| `AgentState` | `TypedDict` with `messages: Annotated[list, add_messages]` |
| `@tool` decorators | 4 math functions auto-registered with LLM |
| `ToolNode` | Prebuilt LangGraph node that executes tool calls |
| `should_continue()` | Conditional router: `tool_calls` → tools, else END |
| LLM | Groq `llama-3.3-70b-versatile`, temperature=0 |

---

## Sub-Assignment 2 — Multi-Agent Research & Summarization System

**Files:** `graph.py`, `state.py`, `agents/` (5 files), `main.py`

### What It Does
Full multi-agent system where a router classifies queries and directs them to the right specialist agent, then summarizes all results uniformly.

### Graph Structure

```
User Query
    ↓
[router] ← Claude Haiku classifies: web / rag / llm
    ├── web  → [web_research] → [summarization] → END
    ├── rag  → [rag]          → [summarization] → END
    └── llm  → [llm_direct]   → [summarization] → END
```

### Agents

| Agent | File | LLM | Purpose |
|-------|------|-----|---------|
| **Router** | `agents/router.py` | Claude Haiku | Classify: web / rag / llm |
| **Web Research** | `agents/web_research.py` | — | Fetch real-time web info |
| **RAG** | `agents/rag_agent.py` | — | Query knowledge base (JSON) |
| **LLM Direct** | `agents/llm_direct.py` | — | General reasoning, no retrieval |
| **Summarization** | `agents/summarization.py` | — | Unified summary of any route's output |

### Router Classification Rules

- `web` → needs real-time info: news, prices, recent events
- `rag` → needs knowledge base: ML algorithms, NLP, LangChain/LangGraph, transformers
- `llm` → everything else: reasoning, math, creative writing, definitions

### AgentState

```python
class AgentState(TypedDict):
    query: str
    route: Optional[str]          # "llm" | "web" | "rag"
    web_results: Optional[str]
    rag_results: Optional[str]
    llm_response: Optional[str]
    final_summary: Optional[str]
    sources: Optional[List[str]]
```

### Data

`data/knowledge_base.json` — local JSON used by the RAG agent

---

## Tech Stack

| Technology | Used In |
|-----------|---------|
| LangGraph (StateGraph, ToolNode, conditional edges) | Both |
| LangChain (tool decorators, prompts) | Both |
| Groq API (llama-3.3-70b-versatile) | Sub-1 |
| Anthropic Claude (claude-haiku-4-5) | Sub-2 router |
| Python TypedDict | Both (state management) |

## Work Completed

- [x] LangGraph math agent with tool-use loop
- [x] 4 custom math tools with `@tool` decorator
- [x] Conditional edge routing (tool_calls check)
- [x] Graph visualization diagram documented
- [x] Multi-agent routing system with 5 nodes
- [x] Query classifier (web/rag/llm) via Claude Haiku
- [x] Modular `agents/` package structure
- [x] Typed `AgentState` with all result fields
- [x] Environment config via `.env.example`

## Complexity

**High** — LangGraph StateGraph with conditional routing, multi-agent coordination, typed state management, and integration of multiple LLMs and retrieval strategies.
