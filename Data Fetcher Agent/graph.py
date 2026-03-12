from langgraph.graph import StateGraph, END

from state import AgentState

from nodes.dataset_discovery import dataset_discovery
from nodes.data_fetch_node import data_fetch_node
from nodes.regulatory_metrics_node import regulatory_metrics_node
from nodes.report_generation_node import report_generation_node


def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("dataset_discovery", dataset_discovery)
    graph.add_node("data_fetch", data_fetch_node)
    graph.add_node("compute_metrics", regulatory_metrics_node)
    graph.add_node("generate_report", report_generation_node)

    graph.set_entry_point("dataset_discovery")

    graph.add_edge("dataset_discovery", "data_fetch")
    graph.add_edge("data_fetch", "compute_metrics")
    graph.add_edge("compute_metrics", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()