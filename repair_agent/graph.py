from langgraph.graph import StateGraph, START, END
from repair_agent.state import RepairState


def build_graph(
        repository_context_node,
        failure_analysis_node,
        repair_node,
        grouping_node,
        patch_and_validation_node,
        pull_request_node,
):

    builder = StateGraph(RepairState)

    builder.add_node(
        "repository_context",
        repository_context_node,
    )

    builder.add_node(
        "failure_analysis",
        failure_analysis_node,
    )

    builder.add_node(
        "repair",
        repair_node,
    )

    builder.add_node(
        "grouping",
        grouping_node,
    )

    builder.add_node(
        "patch_and_validation",
        patch_and_validation_node,
    )

    builder.add_node(
        "pull_request",
        pull_request_node,
    )

    builder.add_edge(
        START,
        "repository_context",
    )

    builder.add_edge(
        "repository_context",
        "failure_analysis",
    )

    builder.add_edge(
        "failure_analysis",
        "repair",
    )

    builder.add_edge(
        "repair",
        "grouping",
    )

    builder.add_edge(
        "grouping",
        "patch_and_validation"
    )

    builder.add_edge(
        "patch_and_validation",
        "pull_request",
    )

    builder.add_edge(
        "pull_request",
        END,
    )

    return builder.compile()