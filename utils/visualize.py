import re
from IPython.display import clear_output 
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from agent_builder.base.state import get_initial_state

def _slug(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', s)

def _strip_mermaid_header_and_classes(mermaid: str) -> list[str]:
    """
    Remove front matter, 'graph ...;' header and classDefs from a Mermaid string,
    returning only node/edge lines.
    """
    lines = mermaid.splitlines()
    body = []
    seen_graph_line = False
    for ln in lines:
        s = ln.strip()
        if not seen_graph_line:
            if s.startswith("graph "):
                seen_graph_line = True
            continue
        if not s or s.startswith("classDef"):
            continue
        body.append(ln)
    return body

def _prefix_internal_ids(lines: list[str], prefix: str) -> list[str]:
    """
    Prefix node IDs in node declarations and edge endpoints.
    Keeps the rest of the line (labels, HTML, etc.) intact.
    """
    p = _slug(prefix)

    def pref_id(x: str) -> str:
        return f"{p}__{x}"

    out = []
    for line in lines:
        s = line

        # 1) Node declarations at line start: <id>(, <id>[, <id>{, <id>[[, etc.
        #    Example:   __start__(<p>__start__</p>)
        s = re.sub(
            r'^(\s*)([A-Za-z0-9_]+)(\s*[\(\[\{])',
            lambda m: f"{m.group(1)}{pref_id(m.group(2))}{m.group(3)}",
            s,
        )

        # 2) Edge left endpoints at line start: <id> -->, -.->, ===>, etc.
        s = re.sub(
            r'^(\s*)([A-Za-z0-9_]+)(\s*)([-=.]+>|==>|-->)',
            lambda m: f"{m.group(1)}{pref_id(m.group(2))}{m.group(3)}{m.group(4)}",
            s,
        )

        # 3) Edge right endpoints after an arrow (with optional label): -->|lbl| <id>
        #    Works for multiple arrows per line.
        def repl_right(match):
            arrow = match.group(1)
            label = match.group(2) or ""  # "|...|" or empty
            ident = match.group(3)
            return f"{arrow}{label}{pref_id(ident)}"

        s = re.sub(
            r'([-=.]*>|==>|-->)\s*(\|[^|]*\|\s*)?([A-Za-z0-9_]+)',
            repl_right,
            s,
        )

        out.append(s)
    return out

def draw_agent_with_subgraphs(
    agent,
    expand_tasks=None,
    include_router=True,
    flow_config="curve: linear",
    internal_entry_node="chatbot",  # adjust if your entry is different
):
    """
    Render a Mermaid diagram that includes:
      - The parent Agent graph
      - A subgraph per Task with its internal LangGraph (IDs prefixed)
      - A connection: ParentTaskNode --> <TaskName>__<internal_entry_node>

    Params:
      agent: your Agent instance
      expand_tasks: optional set/list of task names to expand. If None, expand all.
      include_router: include 'default_router_task' expansion (default True)
      flow_config: mermaid flowchart config snippet (e.g., 'curve: linear')
      internal_entry_node: inner graph entry node (defaults to 'chatbot')
    """
    parent = agent.graph.get_graph().draw_mermaid()

    # Mermaid header
    out = []
    out.append("---")
    out.append("config:")
    out.append("  flowchart:")
    for ln in (flow_config or "").splitlines():
        if ln.strip():
            out.append(f"    {ln}")
    out.append("---")
    out.append("graph TD;")

    # Parent graph body
    parent_body = _strip_mermaid_header_and_classes(parent)
    out.extend(parent_body)

    # Subgraphs
    for task in agent.tasks:
        tname = task.task_config.name
        if expand_tasks is not None and tname not in set(expand_tasks):
            continue
        if not include_router and tname == "default_router_task":
            continue
        if not hasattr(task, "graph") or task.graph is None:
            continue

        # Inner graph
        try:
            inner = task.graph.get_graph().draw_mermaid()
        except Exception:
            continue

        inner_body = _strip_mermaid_header_and_classes(inner)
        inner_prefixed = _prefix_internal_ids(inner_body, tname)

        # Subgraph box (do NOT redeclare the parent node ID here to avoid duplicates)
        out.append(f"subgraph cluster_{_slug(tname)}[\"{tname}\"]")
        out.append("    %% Internal task graph (IDs are prefixed to avoid collisions)")
        for ln in inner_prefixed:
            out.append("    " + ln)
        out.append("end")

        # Connect parent Task node (outside) to internal entry (inside subgraph)
        parent_task_node_id = _slug(tname)  # this is how LangGraph names the parent node
        inner_entry_id = f"{_slug(tname)}__{internal_entry_node}"
        out.append(f"{parent_task_node_id} --> {inner_entry_id};")

    # Optional styling (keep minimal)
    out.append("classDef default fill:#f2f0ff,line-height:1.2;")
    out.append("classDef first fill-opacity:0;")
    out.append("classDef last fill:#bfb6fc;")

    return "\n".join(out)





def _render_ai_message(msg: AIMessage, label:str = "Assistant"):
    if msg.content:
        print(f"\n🤖 {label}:\n{msg.content}")
    for call in msg.additional_kwargs.get("tool_calls", []):
        name = call['function'].get("name")
        args = call['function'].get("arguments")
        print(f"\n🤖 {label} ➜ tool call: {name}({json.dumps(args, indent=0)})")
        

def _render_tool_message(msg: ToolMessage):
    """
    Show the result returned by the tool function.
    """
    print(f"\n🔧 Tool result ({msg.name}):\n{msg.content}")



def print_history(messages, label="You",clear_screen = True):
    if clear_screen:
        clear_output(wait=True)
    for m in messages:
        if isinstance(m, HumanMessage):
            print(f"\n👤 {label}-input:\n{m.content}")
        elif isinstance(m, AIMessage):
            _render_ai_message(m, label)
        elif isinstance(m, ToolMessage):
            _render_tool_message(m)

# ------------------------------------------------------------------------------------
# 2. Async chat loop
# ------------------------------------------------------------------------------------
async def chat_loop(agent):
    state = get_initial_state(session_id=str(uuid.uuid4()))

    while True:
        # --- 2.1 get user input ------------------------------------------------------
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit", "q"}:
            print("👋 conversation finished.")
            return state

        # --- 2.2 append HumanMessage ------------------------------------------------
        state["messages"].append(HumanMessage(content=user_text))

        # --- 2.3 call agent ----------------------------------------------------------
        state = await agent.ainvoke(state)

        # --- 2.4 display whole history ----------------------------------------------
        print_history(state["messages"])


async def agents_chat_loop(
        agent_1,
        agent_2,
        *,
        start_prompt: str = "Hello! What do you want?",
        stop_keyword: str = "STOP",
        max_turns: int = 20
    ):

    state_1 = get_initial_state(session_id=str(uuid.uuid4()))
    state_2 = get_initial_state(session_id=str(uuid.uuid4()))

    # kick-off
    state_1["messages"].append(HumanMessage(content=start_prompt))

    for _ in range(max_turns):

        # ---------------- Agent-1 ---------------------------------------------------
        old_len = len(state_1["messages"])
        state_1 = await agent_1.ainvoke(state_1)

        new_msgs_1 = state_1["messages"][old_len:]          # everything just added
        print_history(new_msgs_1, agent_1.agent_config.name, clear_screen=False)

        # pick the last AIMessage in that slice (there might be several)
        ai_msg_1 = next(
            (m for m in reversed(new_msgs_1) if isinstance(m, AIMessage)), None
        )
        if not ai_msg_1:
            raise RuntimeError("Agent-1 produced no AIMessage!")

        if stop_keyword.lower() in (ai_msg_1.content or "").lower():
            print(f"\n🔚 '{stop_keyword}' detected – conversation finished.")
            break

        state_2["messages"].append(HumanMessage(content=ai_msg_1.content))

        # ---------------- Agent-2 ---------------------------------------------------
        old_len = len(state_2["messages"])
        state_2 = await agent_2.ainvoke(state_2)

        new_msgs_2 = state_2["messages"][old_len:]
        print_history(new_msgs_2, agent_2.agent_config.name, clear_screen=False)

        ai_msg_2 = next(
            (m for m in reversed(new_msgs_2) if isinstance(m, AIMessage)), None
        )
        if not ai_msg_2:
            raise RuntimeError("Agent-2 produced no AIMessage!")

        if stop_keyword.lower() in (ai_msg_2.content or "").lower():
            print(f"\n🔚 '{stop_keyword}' detected – conversation finished.")
            break

        state_1["messages"].append(HumanMessage(content=ai_msg_2.content))

    else:
        print(f"\n⚠️  Reached max_turns={max_turns}. Terminating.")

    return state_1, state_2

