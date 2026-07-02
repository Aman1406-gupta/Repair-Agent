import os

registry = None

if os.environ.get('ENABLE_AGENT_BUILDER_REGISTRY', 'false').lower() == 'true':
    from .storage.registry import InMemoryRegistry
    registry = InMemoryRegistry()


