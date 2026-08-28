"""Canonical JSON and digest helpers for repository-development contracts."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _sort_key(value: Any) -> str:
    """Return a canonical comparable representation for unordered values."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonicalize_model(value: BaseModel) -> Any:
    return canonicalize(value.model_dump(mode="json", exclude_none=False))


def _canonicalize_enum(value: Enum) -> Any:
    return canonicalize(value.value)


def _canonicalize_mapping(value: Mapping[Any, Any]) -> Any:
    items = ((str(key), canonicalize(item)) for key, item in value.items())
    return {key: item for key, item in sorted(items, key=lambda pair: pair[0])}


def _canonicalize_unordered(value: set[Any] | frozenset[Any]) -> Any:
    return sorted((canonicalize(item) for item in value), key=_sort_key)


def _canonicalize_sequence(value: list[Any] | tuple[Any, ...]) -> Any:
    return [canonicalize(item) for item in value]


def _canonicalize_datetime(value: datetime) -> Any:
    if value.tzinfo is None:
        raise ValueError("canonical datetimes must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonicalize_date(value: date) -> Any:
    return value.isoformat()


def _canonicalize_path(value: Path) -> Any:
    return str(value)


def _canonicalize_bytes(value: bytes) -> Any:
    return base64.b64encode(value).decode("ascii")


# Ordered (type, handler) pairs, checked with isinstance in this exact
# order. The order is load-bearing: e.g. datetime is a subclass of date and
# must be matched first, and this mirrors the original if/elif chain
# precisely so subclass resolution is unchanged.
_CANONICALIZERS: tuple[tuple[type | tuple[type, ...], Callable[[Any], Any]], ...] = (
    (BaseModel, _canonicalize_model),
    (Enum, _canonicalize_enum),
    (Mapping, _canonicalize_mapping),
    ((set, frozenset), _canonicalize_unordered),
    ((list, tuple), _canonicalize_sequence),
    (datetime, _canonicalize_datetime),
    (date, _canonicalize_date),
    (Path, _canonicalize_path),
    (bytes, _canonicalize_bytes),
)


def canonicalize(value: Any) -> Any:
    """Convert a supported value into deterministic JSON-compatible data.

    Mapping keys are sorted by their string representation, while sequences
    retain their declared order.  Sets are sorted by their canonical JSON
    representation so callers cannot produce different digests by changing
    insertion order.
    """

    for types, handler in _CANONICALIZERS:
        if isinstance(value, types):
            return handler(value)
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
