"""Machine-readable bundle for the repository-development v1 models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .version import CONTRACT_NAME, CONTRACT_VERSION


def contract_schema_bundle(
    model_types: tuple[type[BaseModel], ...],
) -> dict[str, Any]:
    """Build a versioned JSON-schema bundle for the supplied public models.

    Keeping the bundle function-based avoids committing generated schema churn
    while still giving package consumers a deterministic, machine-readable
    schema source.  Callers should pass an explicit ordered tuple so additions
    are reviewable and do not silently alter the public inventory.
    """

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "contract_name": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "models": {
            model_type.__name__: model_type.model_json_schema(mode="serialization")
            for model_type in model_types
        },
    }


__all__ = ["contract_schema_bundle"]
