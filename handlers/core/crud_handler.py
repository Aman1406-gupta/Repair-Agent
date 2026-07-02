from typing import Any, Dict

from agent_builder.handlers.core.base_handler import BaseBuilderHandler


class CrudHandler(BaseBuilderHandler):
    """Declarative base for simple CRUD endpoints.

    Subclasses set class attributes instead of writing boilerplate:
        request_model  — Pydantic model to validate the payload
        response_model — Pydantic model for the response (used with **result)
        mongo_method   — method name on self.mongo_client

    Override validate_payload() or build_response() for edge cases.
    """
    request_model = None
    response_model = None
    mongo_method: str = None

    def validate_payload(self, payload: Dict[str, Any]):
        if self.request_model is None:
            raise NotImplementedError("Set request_model or override validate_payload()")
        return self.request_model(**payload)

    async def process(self, request) -> Dict[str, Any]:
        method = getattr(self.mongo_client, self.mongo_method)
        result = await method(request)
        return self.build_response(request, result)

    def build_response(self, request, result) -> Dict[str, Any]:
        if self.response_model is None:
            raise NotImplementedError("Set response_model or override build_response()")
        return self.response_model(**result).model_dump()
