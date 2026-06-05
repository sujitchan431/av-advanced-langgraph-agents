"""
Multi-Agent Research and Summarization System
Built with LangGraph + LangChain + Anthropic Claude
"""

import os
from dotenv import load_dotenv

load_dotenv()


def run_query(query: str, graph) -> dict:
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    initial_state = {
        "query": query,
        "route": None,
        "web_results": None,
        "rag_results": None,
        "llm_response": None,
        "final_summary": None,
        "sources": None,
    }

    result = graph.invoke(initial_state)

    print("\n" + "-" * 60)
    print("FINAL RESPONSE:")
    print("-" * 60)
    print(result["final_summary"])
    print("-" * 60)

    return result


def main():
    from graph import build_graph

    print("Building agent graph…")
    graph = build_graph()
    print("Graph ready.\n")

    test_queries = [
        # LLM route — general reasoning
        "What is the difference between supervised and unsupervised learning?",
        # RAG route — knowledge base hit
        "How does LangGraph work and what makes it different from simple chains?",
        # Web route — current events
        "What are the latest developments in AI regulation in 2024?",
        # RAG route — ML topic
        "Explain how Random Forest handles feature selection and overfitting.",
        # LLM route — reasoning/math
        "If I have a dataset with 80% class A and 20% class B, which metric should I prioritize?",
    ]

    for query in test_queries:
        run_query(query, graph)
        print()


if __name__ == "__main__":
    main()
