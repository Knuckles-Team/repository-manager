"""Canonical JSON and digest helpers for repository-development contracts."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _sort_key(value: Any) -> str:
    """Return a canonical comparable representation for unordered values."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonicalize(value: Any) -> Any:
    """Convert a supported value into deterministic JSON-compatible data.

    Mapping keys are sorted by their string representation, while sequences
    retain their declared order.  Sets are sorted by their canonical JSON
    representation so callers cannot produce different digests by changing
    insertion order.
    """

    if isinstance(value, BaseModel):
        return canonicalize(value.model_dump(mode="json", exclude_none=False))
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, Mapping):
        items = ((str(key), canonicalize(item)) for key, item in value.items())
        return {key: item for key, item in sorted(items, key=lambda pair: pair[0])}
    if isinstance(value, (set, frozenset)):
        return sorted((canonicalize(item) for item in value), key=_sort_key)
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("canonical datetimes must be timezone-aware")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported value for canonical JSON: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize *value* with stable key ordering and separators."""

    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_digest(value: Any, *, algorithm: str = "sha256") -> str:
    """Return a full hexadecimal digest of a canonical JSON value."""

    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"unsupported digest algorithm: {algorithm!r}") from exc
    digest.update(canonical_json(value).encode("utf-8"))
    return digest.hexdigest()


def serialize_contract(model: BaseModel) -> str:
    """Serialize one typed contract to canonical JSON."""

    return canonical_json(model)


def deserialize_contract(model_type: type[T], payload: str | bytes | bytearray) -> T:
    """Deserialize canonical JSON into the requested typed contract."""

    if not isinstance(payload, (str, bytes, bytearray)):
        raise TypeError("contract payload must be JSON text or bytes")
    return model_type.model_validate_json(payload)


def contract_schema(model_type: type[BaseModel]) -> dict[str, Any]:
    """Return the machine-readable serialization schema for one contract."""

    return model_type.model_json_schema(mode="serialization")
