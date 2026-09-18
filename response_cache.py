"""Small, immutable cache for exact model responses.

The cache does not understand DentVLM tasks or experiments. A caller supplies a
namespace describing the model/runtime and the exact request. Only an exact digest
match is reused; malformed artifacts fail loudly instead of being silently ignored.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


SCHEMA_VERSION = 1
REPLY_FIELDS = (
    "text",
    "finish_reason",
    "truncated",
    "prompt_tokens",
    "completion_tokens",
    "latency_seconds",
)


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class ResponseCache:
    """Content-addressed JSON replies under one model/runtime namespace."""

    def __init__(self, root: str | Path, namespace: dict) -> None:
        self.root = Path(root)
        self.namespace = namespace
        self.namespace_sha256 = hashlib.sha256(_json_bytes(namespace)).hexdigest()

    def key(self, request: dict) -> str:
        payload = {"schema_version": SCHEMA_VERSION, "namespace": self.namespace, "request": request}
        return hashlib.sha256(_json_bytes(payload)).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid response-cache artifact {path}: {exc}") from exc
        if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"invalid response-cache artifact {path}: unsupported schema")
        if artifact.get("key") != key or artifact.get("namespace_sha256") != self.namespace_sha256:
            raise ValueError(f"invalid response-cache artifact {path}: identity mismatch")
        reply = artifact.get("reply")
        if not isinstance(reply, dict) or set(reply) != set(REPLY_FIELDS):
            raise ValueError(f"invalid response-cache artifact {path}: incomplete reply")
        if not isinstance(reply["text"], str) or not isinstance(reply["truncated"], bool):
            raise ValueError(f"invalid response-cache artifact {path}: invalid reply types")
        return dict(reply)

    def put(self, key: str, reply: dict) -> Path:
        """Atomically store the normalized reply. Existing immutable entries win."""
        path = self._path(key)
        if path.is_file():
            self.get(key)
            return path
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "key": key,
            "namespace_sha256": self.namespace_sha256,
            "reply": {field: reply.get(field) for field in REPLY_FIELDS},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(artifact, indent=1, ensure_ascii=False), encoding="utf-8")
            if path.is_file():
                self.get(key)
            else:
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path
