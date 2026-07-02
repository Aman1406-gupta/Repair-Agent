from typing import Any, Dict, List, Optional, Tuple
from pydantic import create_model
import re

# -------------------------------------------------------------------------
# 1) JSON-schema → Python type
# -------------------------------------------------------------------------
def _python_type_from_schema(schema: Dict[str, Any]) -> type:
    return {
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(schema.get("type", "string"), str)


def _extract_properties_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    # Direct properties
    if "properties" in schema:
        return schema["properties"]
    
    # allOf composition - merge properties from all schemas
    if "allOf" in schema:
        merged_props = {}
        for sub_schema in schema["allOf"]:
            if isinstance(sub_schema, dict):
                merged_props.update(_extract_properties_from_schema(sub_schema))
        return merged_props
    
    # oneOf/anyOf - take properties from first schema that has them
    for keyword in ("oneOf", "anyOf"):
        if keyword in schema:
            for sub_schema in schema[keyword]:
                if isinstance(sub_schema, dict):
                    props = _extract_properties_from_schema(sub_schema)
                    if props:
                        return props
    
    return {}


def _extract_required_from_schema(schema: Dict[str, Any]) -> set:
    """
    Extract required field names from a JSON schema, handling compositions.
    """
    required = set(schema.get("required", []))
    
    # allOf composition - merge required from all schemas
    if "allOf" in schema:
        for sub_schema in schema["allOf"]:
            if isinstance(sub_schema, dict):
                required.update(_extract_required_from_schema(sub_schema))
    
    return required

# -------------------------------------------------------------------------
# 2) Build a Pydantic Args model (OpenAPI parameters + requestBody)
# -------------------------------------------------------------------------
def _build_arg_model(
    op_id: str,
    params: List[Dict[str, Any]],
    body_schema: Optional[Dict[str, Any]] = None
) -> Tuple[Any, List[str]]:
    """
    Build a Pydantic model for the operation's arguments.
    
    Returns:
        Tuple of (ArgsModel, body_field_names):
        - ArgsModel: The Pydantic model for the arguments
        - body_field_names: List of field names that belong in the HTTP request body
    """
    fields: Dict[str, Tuple[type, Any]] = {}
    body_field_names: List[str] = []

    # Add path/query parameters
    for p in params:
        name = p["name"]
        required = p.get("required", False)
        typ = _python_type_from_schema(p.get("schema", {}))
        fields[name] = (typ, ... if required else None)

    # Flatten requestBody properties into individual parameters
    if body_schema:
        # Try to extract properties - handles direct properties and allOf compositions
        props = _extract_properties_from_schema(body_schema)
        required_fields = _extract_required_from_schema(body_schema)
        
        if props:
            # Flatten each property from the body schema
            for prop_name, prop_schema in props.items():
                typ = _python_type_from_schema(prop_schema)
                is_required = prop_name in required_fields
                fields[prop_name] = (typ, ... if is_required else None)
                body_field_names.append(prop_name)
        else:
            # Fallback for non-object schemas (arrays, primitives, $ref, etc.)
            # Use a generic 'body' param
            fields["body"] = (dict, ...)
            body_field_names.append("body")

    model = create_model(f"{op_id.title()}Args", **fields)  # type: ignore[arg-type]
    return model, body_field_names



def sanitize_tool_name(name: str) -> str:

    original_name = name

    name = name.replace('{', '').replace('}', '')
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)    
    name = re.sub(r'_+', '_', name)
    name = name.strip('_-')
    if not name or name[0].isdigit():
        name = 'tool_' + name
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError(
            f"Failed to sanitize tool name. Original: '{original_name}' -> "
            f"Sanitized: '{name}' does not match pattern ^[a-zA-Z0-9_-]+$"
        )
    
    return name

