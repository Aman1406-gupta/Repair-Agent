import os
from copy import deepcopy
from typing import List, Any, Dict
from agent_builder.base.state import State
from agent_builder.base.configs import TaskConfig
from agent_builder.llm_client.sprinklr_chat_model import SprinklrChatModel
from langchain_core.messages import AIMessage, HumanMessage, convert_to_openai_messages
from agent_builder.base.task import Task

import json


class ThinkActTask(Task):
    def __init__(self, task_config: TaskConfig, tools, handoffs, memory, callbacks=None):
        thinker_llm_config = deepcopy(task_config.llm_config)
        # predetermined / or fetched from mongo
        thinker_llm_config.model = 'gpt-4.1-mini'
        # thinker_llm_config.model = 'gpt-5-nano'

        self.thinker_llm = SprinklrChatModel(llm_config=thinker_llm_config, disable_streaming=True)
        self.actor_llm = SprinklrChatModel(llm_config=task_config.llm_config).bind_tools(tools or [])

        current_dir = os.path.dirname(os.path.abspath(__file__))
        thinker_template_path = os.path.join(current_dir, 'prompts', 'think_act_thinker_template.txt')
        self.thinker_system_template = open(thinker_template_path).read()

        task_config.system_template = f"{task_config.system_template}\n\n[IMPORTANT] Make sure to adhere to the plans provided in between <plan> </plan> tags!"
        task_config.task_type = "think_act"
        super().__init__(task_config, tools, handoffs, memory, callbacks=callbacks)

    def get_default_chatbot_node(self):
        async def chatbot(state: State, **kwargs):
            transcript = render_history(state["messages"])
            thinking_prompt = self.thinker_system_template.format(
                think_act_system_template=self.task_config.system_template, conv_history=transcript)
            thinking_message_full = (await self.thinker_llm.ainvoke([HumanMessage(content=thinking_prompt)], **kwargs))
            thinking_message = thinking_message_full.content
            thinking_spending = thinking_message_full.response_metadata['spending']
            input_messages = deepcopy(state['messages'])
            input_messages += [AIMessage(content=f"<plan>{thinking_message}</plan>")]
            actor_msg = await self.actor_llm.ainvoke(input_messages, **kwargs)

            ##Spending is not returned by llm router in case of streaming
            ##TODO: token usage from streamed data
            actor_msg.response_metadata['spending'] = actor_msg.response_metadata.get('spending', 0) + thinking_spending
            state["messages"] += [actor_msg]
            return state

        return chatbot


def render_history(history: List[Dict[str, Any]]) -> str:
    def _stringify(obj: Any) -> str:
        if isinstance(obj, str):
            return obj
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    ROLE_LABELS = {
        "user": "USER",
        "assistant": "ASSISTANT",
        "system": "SYSTEM",
        "tool": "TOOL",
        "function": "FUNCTION",
    }

    lines = []

    for msg in convert_to_openai_messages(history):
        role = msg.get("role", "unknown")
        if role == 'system':
            continue

        label = ROLE_LABELS.get(role, role.upper())
        content = msg.get("content", "")

        if role == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                fn = call.get("function", {})
                name = fn.get("name", "unknown_function")
                raw_args = fn.get("arguments", {})
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                lines.append(f"ASSISTANT TOOL CALL: name={name} args={args}\n")

        if role in {"tool", "function"}:
            meta: List[str] = []
            tool_name = msg.get("name") or msg.get("tool")
            if tool_name:
                meta.append(f"name={tool_name}")
            if meta:
                label += f" ({', '.join(meta)})"

            content = _stringify(content)

        if content:
            lines.append(f"{label}: {content}\n")

    return "".join(lines)
