"""Thin typed seam onto epistemic-graph's served work-coordination methods.

Work items, development-lane state, claim leases and change cursors are
epistemic-graph records. Repository-manager reaches them only through the EG
client (``epistemic_graph.client``) that the caller injects: ``client.work_items``
(``SubmitWorkItem`` / ``ClaimWorkItem`` / ``GetWorkItem`` / ``ListWorkItems`` /
``CancelWorkItem``) and ``client.development_lanes`` (``ReserveDevelopmentLane``
… ``DevelopmentLaneStatus``). Nothing here stores, plans or decides; a client
that does not serve a method fails closed with :class:`EgWorkCoordinationUnavailable`.

The repository work-item projection (repository requests onto the generic
WorkItem rows) is not served by EG yet, so :data:`repository_work_items` refuses
every call until it is (handoff ``EG-WORK-COORDINATION``).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Final

__all__ = [
    "DEVELOPMENT_LANE_METHODS",
    "EgDevelopmentLaneTransport",
    "EgWorkCoordinationUnavailable",
    "EgWorkItemClaimer",
    "repository_work_items",
]


class EgWorkCoordinationUnavailable(RuntimeError):
    """The connected EG client does not serve the requested coordination method."""

    code: Final[str] = "eg_work_coordination_unavailable"

    def __init__(self, method: str) -> None:
        super().__init__(f"{self.code}: {method}")
        self.method = method


#: Wire method name -> attribute on ``client.development_lanes``.
DEVELOPMENT_LANE_METHODS: Final[Mapping[str, str]] = {
    "ReserveDevelopmentLane": "reserve",
    "RenewDevelopmentLane": "renew",
    "ObserveDevelopmentLane": "observe",
    "FinishDevelopmentLane": "finish",
    "CleanupDevelopmentLane": "cleanup_complete",
    "QueryDevelopmentLane": "query",
    "DevelopmentLaneStatus": "status",
    "UpdateDevelopmentLaneQuota": "update_quota",
}


def _namespace(client: Any, name: str, method: str) -> Any:
    namespace = getattr(client, name, None)
    if namespace is None:
        raise EgWorkCoordinationUnavailable(method)
    return namespace


def _sync_result(value: Any, method: str) -> dict[str, Any]:
    if inspect.isawaitable(value):
        close = getattr(value, "close", None)
        if callable(close):
            close()
        raise TypeError(f"{method} requires the synchronous EG client")
    if not isinstance(value, Mapping):
        raise TypeError(f"{method} returned a non-mapping result")
    return dict(value)


class EgDevelopmentLaneTransport:
    """Call EG's served development-lane methods; capability-checked, fail closed."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._lanes = _namespace(client, "development_lanes", "development_lanes")

    def _require_capability(self, method: str) -> None:
        supports = getattr(self._client, "supports", None)
        if not callable(supports):
            raise EgWorkCoordinationUnavailable(method)
        try:
            advertised = supports(method)
        except Exception as exc:  # noqa: BLE001 - malformed/old Health fails closed
            raise EgWorkCoordinationUnavailable(method) from exc
        if advertised is not True:
            raise EgWorkCoordinationUnavailable(method)

    def call(self, method: str, request: Mapping[str, Any]) -> dict[str, Any]:
        attribute = DEVELOPMENT_LANE_METHODS.get(method)
        if attribute is None:
            raise ValueError(f"unknown development-lane method {method!r}")
        self._require_capability(method)
        operation = getattr(self._lanes, attribute, None)
        if not callable(operation):
            raise EgWorkCoordinationUnavailable(method)
        return _sync_result(operation(request=dict(request)), method)

    def reserve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.call("ReserveDevelopmentLane", request)

    def renew(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.call("RenewDevelopmentLane", request)

    def observe(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.call("ObserveDevelopmentLane", request)

    def finish(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.call("FinishDevelopmentLane", request)

    def cleanup_complete(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.call("CleanupDevelopmentLane", request)

    def query(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.call("QueryDevelopmentLane", request)

    def status(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.call("DevelopmentLaneStatus", request)

    def update_quota(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self.call("UpdateDevelopmentLaneQuota", request)


class EgWorkItemClaimer:
    """Submit-if-absent then claim one ``lane.lifecycle`` WorkItem through EG."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def claim_lifecycle(
        self,
        *,
        work_item_id: str,
        kind: str,
        tenant_ref: str,
        payload_ref: str,
        idempotency_key: str,
        created_by: str,
        metadata: Mapping[str, Any],
        now_ms: int,
    ) -> dict[str, Any] | None:
        work_items = _namespace(self._client, "work_items", "SubmitWorkItem")
        submit = getattr(work_items, "submit", None)
        claim = getattr(work_items, "claim", None)
        if not callable(submit) or not callable(claim):
            raise EgWorkCoordinationUnavailable("SubmitWorkItem/ClaimWorkItem")
        _sync_result(
            submit(
                {
                    "tenant": tenant_ref,
                    "work_item_id": work_item_id,
                    "kind": kind,
                    "payload_ref": payload_ref,
                    "idempotency_key": idempotency_key,
                    "created_by": created_by,
                    "metadata": dict(metadata),
                    "now_ms": now_ms,
                }
            ),
            "SubmitWorkItem",
        )
        claimed = _sync_result(
            claim(
                {
                    "tenant": tenant_ref,
                    "work_item_id": work_item_id,
                    "limit": 1,
                    "now_ms": now_ms,
                },
                idempotency_key=f"{idempotency_key}:claim",
            ),
            "ClaimWorkItem",
        )
        if claimed.get("claimed") is False:
            return None
        return claimed


class _UnservedRepositoryWorkItems:
    """Refuses every repository work-item projection call until EG serves it."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise EgWorkCoordinationUnavailable(f"repository_work_item.{name}")


repository_work_items: Final[Any] = _UnservedRepositoryWorkItems()
