"""Parse cachewise_policy from the client JSON and compute priorities for KV cache."""

from __future__ import annotations
import json
from typing import Any
_V = 1

def parse_cachewise_policy_body(raw: Any) -> dict[str, Any] | None:
    if raw is None or not isinstance(raw, dict):
        return None
    if raw.get("version", _V) != _V:
        return None
    scope = raw.get("scope")
    if scope is not None and not isinstance(scope, dict):
        return None
    hints = raw.get("hints")
    if hints is not None and not isinstance(hints, dict):
        return None
    return {"version": _V, "scope": dict(scope) if scope else {}, "hints": dict(hints) if hints else {}}


def cachewise_eviction_score(policy: dict[str, Any] | None) -> float:
    """Higher => keep longer when reordering eviction among cached+free blocks."""

    if not policy:
        return 0.0
    hints = policy.get("hints") or {}
    oracle = hints.get("oracle")
    if isinstance(oracle, dict):
        try:
            return float(oracle.get("ground_truth_idle_s", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    next_tool = hints.get("next_tool")
    if isinstance(next_tool, dict):
        name = str(next_tool.get("name") or "")
        args = next_tool.get("args")
        tail = json.dumps(args, sort_keys=True) if args is not None else ""
        return float(hash(name + "\0" + tail) % (2**31))
    return 0.0