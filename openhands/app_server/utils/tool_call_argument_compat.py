"""Compatibility parsing for provider-emitted tool call arguments."""

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from openhands.sdk.agent import agent as sdk_agent_module
from openhands.sdk.agent import utils as sdk_agent_utils

_logger = logging.getLogger(__name__)
_original_parse_tool_call_arguments: Callable[[str], dict[str, Any]] = (
    sdk_agent_utils.parse_tool_call_arguments
)


def _parse_json_object(raw_arguments: str) -> dict[str, Any] | None:
    """Parse a JSON object from an argument string if possible."""
    try:
        parsed = json.loads(raw_arguments, strict=False)
    except json.JSONDecodeError:
        try:
            parsed, _ = json.JSONDecoder(strict=False).raw_decode(raw_arguments)
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def _drop_dangling_json_tail(value: str) -> str:
    """Remove tails that cannot be made valid by adding closing delimiters."""
    trimmed = re.sub(r',\s*$', '', value)
    trimmed = re.sub(r':\s*$', ': ""', trimmed)
    return re.sub(r',\s*([}\]])', r'\1', trimmed)


def _close_truncated_json_object(raw_arguments: str) -> str | None:
    """Best-effort repair for truncated object literals from tool calls.

    SIRB-hosted Qwen sometimes returns native function-call arguments such as
    ``{"command": "echo hello``. The SDK's normal fallback only sanitizes raw
    control characters; this closes open strings and delimiters so validation can
    continue with the command the model actually intended to run.
    """
    start = raw_arguments.find('{')
    if start == -1:
        return None

    value = raw_arguments[start:].strip()
    stack: list[str] = []
    in_string = False
    escaped = False

    for char in value:
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in '{[':
            stack.append('}' if char == '{' else ']')
        elif char in '}]' and stack and char == stack[-1]:
            stack.pop()

    if escaped:
        value += '\\'
    if in_string:
        value += '"'

    value = _drop_dangling_json_tail(value)
    value += ''.join(reversed(stack))
    return value


def parse_tool_call_arguments_compat(raw_arguments: str) -> dict[str, Any]:
    """Parse SDK tool-call arguments with a conservative truncation repair."""
    try:
        return _original_parse_tool_call_arguments(raw_arguments)
    except json.JSONDecodeError:
        closed = _close_truncated_json_object(raw_arguments)
        if closed is not None:
            parsed = _parse_json_object(closed)
            if parsed is not None:
                _logger.warning('Repaired malformed tool call arguments for execution')
                return sdk_agent_utils._normalize_arguments(parsed)

        raise


def install_tool_call_argument_compat() -> None:
    """Install the parser shim into SDK modules imported by the app server."""
    sdk_agent_utils.parse_tool_call_arguments = parse_tool_call_arguments_compat
    sdk_agent_module.parse_tool_call_arguments = parse_tool_call_arguments_compat

