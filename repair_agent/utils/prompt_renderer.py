from __future__ import annotations

import json
import re
from typing import Any

_TEMPLATE_PATTERN = re.compile(r"\{\{([^}]+)\}\}")


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return str(value)


def render_prompt_template(template: str, variables: dict[str, Any]) -> str:
    """Replace ``{{key}}`` placeholders in a prompt template."""

    def replacer(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in variables:
            return match.group(0)
        return _format_value(variables[key])

    return _TEMPLATE_PATTERN.sub(replacer, template)
