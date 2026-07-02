import logging
from typing import Any

from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


class StreamableMixin:
    """Provides graph-based ``astream`` for any class that has a ``self.graph``."""

    async def astream(self, state: dict[str, Any], stream_mode=['messages', 'values'], **kwargs):
        allowed = {'messages', 'values'}
        assert not set(stream_mode) - allowed, f"stream_mode must be a subset of {allowed}"

        async for _path, mode, chunk in self.graph.astream(
            state,
            stream_mode=list(stream_mode),
            subgraphs=True,
            **kwargs,
        ):
            if mode == 'values':
                yield {'stream_mode': 'values', 'value': chunk}
                continue

            msg, metadata = chunk
            node = metadata.get('langgraph_node')

            if node in ('chatbot', 'model') and metadata.get('ls_model_type') == 'chat':
                if msg.additional_kwargs.get('disable_streaming', False):
                    continue
                yield {'stream_mode': 'messages', 'message': msg}

            elif node == 'tools' and isinstance(msg, ToolMessage):
                yield {'stream_mode': 'messages', 'message': msg}