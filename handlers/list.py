"""
List handlers — one base class, one-liner subclasses per collection.
"""
import logging
from typing import Any, Dict, List, Optional

import tornado.web

from agent_builder.handlers.core.base_handler import BaseBuilderHandler
from agent_builder.utils.constants import (
    AGENT_COLLECTION,
    AGENT_ID,
    PARTNER_ID,
    VERSION,
)
from agent_builder.storage.utils.builders import build_agent_from_doc
from agent_builder.utils.visualize import draw_agent_with_subgraphs

logger = logging.getLogger(__name__)


class ListHandler(BaseBuilderHandler):
    collection: str = None
    include_mermaid: bool = False

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        doc_ids = payload.get("ids")

        if doc_ids:
            if isinstance(doc_ids, str):
                doc_ids = [doc_ids]

            response = await self.mongo_client._resolve_refs_cached(self.collection, doc_ids)

            if self.include_mermaid:
                response = [await self._enrich_with_mermaid(doc) for doc in response]

            return {"success": True, "response": response}

        response = await self.mongo_client.get_all_docs(collection=self.collection)

        if self.include_mermaid:
            return await self._process_with_mermaid(response)

        return {"success": True, "response": response}

    async def _enrich_with_mermaid(self, agent_doc: Dict[str, Any]) -> Dict[str, Any]:
        try:
            agent_obj = build_agent_from_doc(agent_doc, memory=None)
            mermaid_code = draw_agent_with_subgraphs(agent_obj, include_router=True)
            agent_doc["mermaid_code"] = mermaid_code
        except Exception as e:
            logger.warning("Failed to process agent %s: %s", agent_doc.get('name', 'unknown'), e)
        return agent_doc

    async def _process_with_mermaid(self, agent_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_docs = []
        for agent_doc in agent_docs:
            try:
                agent_obj = build_agent_from_doc(agent_doc, memory=None)
                mermaid_code = draw_agent_with_subgraphs(agent_obj, include_router=True)
                agent_doc["mermaid_code"] = mermaid_code
                processed_docs.append(agent_doc)
            except Exception as e:
                logger.warning("Failed to process agent %s: %s", agent_doc.get('name', 'unknown'), e)
                processed_docs.append(agent_doc)
                continue

        return {"success": True, "response": processed_docs}


# ── One-liner subclasses ─────────────────────────────────────────────────

class ListAgentsHandler(ListHandler):
    collection = AGENT_COLLECTION
    include_mermaid = True

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Support composite lookup by logical agent id + partner + version.

        Query params (GET ``/agents``):

        - ``agentId`` + ``partnerId`` + ``versionId`` (or ``version``) — single matching
          agent document (same triple used by sync / invoke resolution).
        - ``ids`` — Mongo ``_id`` strings (existing behaviour).
        - neither — list all agents.
        """
        agent_id = self._first_query_value(payload, ("agentId", "agent_id"))
        partner_raw = self._first_query_value(payload, ("partnerId", "partner_id"))
        version_raw = self._first_query_value(
            payload, ("versionId", "version", "version_id"),
        )

        if partner_raw is None:
            raise tornado.web.HTTPError(400, reason="partnerId is required")

        try:
            partner_id = int(partner_raw)
        except (TypeError, ValueError) as exc:
            raise tornado.web.HTTPError(
                400, reason="partnerId must be an integer",
            ) from exc

        if agent_id is not None and version_raw is not None:
            aid = str(agent_id).strip()
            if not aid:
                raise tornado.web.HTTPError(400, reason="agentId must be non-empty")
            try:
                version = int(version_raw)
            except (TypeError, ValueError) as exc:
                raise tornado.web.HTTPError(
                    400, reason="versionId/version must be an integer",
                ) from exc

            doc = await self.mongo_client._find_one_cached(
                self.collection,
                {AGENT_ID: aid, PARTNER_ID: partner_id, VERSION: version},
                partner_id=partner_id,
            )
            if not doc:
                return {"success": True, "response": []}

            response: List[Dict[str, Any]] = [doc]
            if self.include_mermaid:
                response = [await self._enrich_with_mermaid(d) for d in response]
            return {"success": True, "response": response}

        doc_ids = payload.get("ids")
        if doc_ids:
            if isinstance(doc_ids, str):
                doc_ids = [doc_ids]

            response = await self.mongo_client._resolve_refs_cached(
                self.collection, doc_ids, partner_id=partner_id,
            )

            if self.include_mermaid:
                response = [await self._enrich_with_mermaid(doc) for doc in response]

            return {"success": True, "response": response}

        response = await self.mongo_client.get_all_docs(
            collection=self.collection, partner_id=partner_id,
        )

        if self.include_mermaid:
            return await self._process_with_mermaid(response)

        return {"success": True, "response": response}

    @staticmethod
    def _first_query_value(payload: Dict[str, Any], keys: tuple) -> Optional[Any]:
        for k in keys:
            if k in payload and payload[k] is not None:
                v = payload[k]
                if isinstance(v, list):
                    v = v[0] if v else None
                if v is not None and str(v).strip() != "":
                    return v
        return None


