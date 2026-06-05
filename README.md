<div align="center">

# 🕸️ Advanced AI Agents with LangGraph — 2 Sub-Assignments

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.x-green)](https://langchain-ai.github.io/langgraph/)
[![Claude](https://img.shields.io/badge/Claude-Haiku_4.5-blueviolet)](https://anthropic.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-darkgreen)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> LangGraph `StateGraph` projects: a math tool-use agent and a full multi-agent routing system (web / RAG / LLM paths) with supervisor pattern.

**🎓 Part of the [Analytics Vidhya GenAI Pinnacle Plus Program](https://www.analyticsvidhya.com/)**

</div>

---

## 📋 Overview

Two LangGraph assignments progressing from a simple tool-use loop to a production-style multi-agent system with intelligent query routing, typed state management, and modular agent architecture.

---

## 📁 Sub-Assignments

```
10th advanced agents Langraph/
├── 1st/  ← LangGraph Math Agent (tool-use loop)
└── 2nd/
    └── Assisgment/
        ├── graph.py      ← StateGraph definition
        ├── state.py      ← AgentState TypedDict
        ├── agents/       ← 5 agent modules
        │   ├── router.py
        │   ├── web_research.py
        │   ├── rag_agent.py
        │   ├── llm_direct.py
        │   └── summarization.py
        └── main.py
```

---

## 🧮 Sub-1 — LangGraph Math Agent

**Graph flow:**
```
User query
    ↓
[agent node] ←────────────────────┐
    │                              │
    ├── has tool_calls? → [tools] ─┘
    │   (runs plus/subtract/multiply/divide)
    └── no tool_calls? → END
```

**Tools:** `@tool plus`, `@tool subtract`, `@tool multiply`, `@tool divide`  
**LLM:** Groq Llama-3.3-70B, temperature=0  
**State:** `MessagesState` with `add_messages` reducer

---

## 🔀 Sub-2 — Multi-Agent Research & Summarization System

**Graph flow:**
```
Query → [Router: Claude Haiku] → classify: web / rag / llm
    ├── web → [Web Research] → [Summarization] → END
    ├── rag → [RAG Agent]    → [Summarization] → END
    └── llm → [LLM Direct]   → [Summarization] → END
```

**Router classification rules:**
| Route | Condition |
|-------|-----------|
| `web` | Real-time info, news, prices, recent events |
| `rag` | ML algorithms, NLP, LangChain/LangGraph, transformers |
| `llm` | Reasoning, math, creative writing, general knowledge |

**AgentState:**
```python
class AgentState(TypedDict):
    query, route, web_results, rag_results,
    llm_response, final_summary, sources
```

---

## ⚙️ Setup

```bash
pip install langgraph langchain-groq langchain-anthropic langchain-core python-dotenv
```

```bash
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 💡 Key Learnings

- `StateGraph` + `TypedDict` state — typed, immutable state passing between nodes
- `add_messages` reducer — appends instead of overwriting message history
- `ToolNode` prebuilt — automatic tool call execution from LLM response
- `tools_condition` — prebuilt edge function checking for tool calls
- `MemorySaver` checkpointer — thread-scoped conversation persistence
- Conditional edges — routing based on state field values
- Modular multi-agent: each agent = one file + one node in graph

---

## 🎓 Program Context

This project is **Assignment 10** of the **Analytics Vidhya GenAI Pinnacle Plus Program** — Advanced AI Agents with LangGraph module.

---

## 📄 License

MIT © 2026 [sujitchan431](https://github.com/sujitchan431)
