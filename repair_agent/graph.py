from langgraph.graph import StateGraph, START, END

from repair_agent.state import RepairState


def failure_router(state: RepairState):

    if state["is_infrastructure"]:
        return END

    if not state["is_reproducible"]:
        return END

    return "repair_generation"


def validation_router(state: RepairState):

    if state["validation_passed"]:
        return "pull_request"

    if state["retry_count"] < state["max_retries"]:
        return "retry"

    return END


def retry_node(state: RepairState):

    state["retry_count"] += 1

    return state


def build_graph(
        repository_context_node,
        failure_analysis_node,
        repair_generation_node,
        validation_node,
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
        "repair_generation",
        repair_generation_node,
    )

    builder.add_node(
        "validation",
        validation_node,
    )

    builder.add_node(
        "retry",
        retry_node,
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

    builder.add_conditional_edges(
        "failure_analysis",
        failure_router,
        {
            "repair_generation": "repair_generation",
            END: END,
        },
    )

    builder.add_edge(
        "repair_generation",
        "validation",
    )

    builder.add_conditional_edges(
        "validation",
        validation_router,
        {
            "pull_request": "pull_request",
            "retry": "retry",
            END: END,
        },
    )

    builder.add_edge(
        "retry",
        "repair_generation",
    )

    builder.add_edge(
        "pull_request",
        END,
    )

    return builder.compile()