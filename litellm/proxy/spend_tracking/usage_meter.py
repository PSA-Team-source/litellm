from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from itertools import accumulate, chain, count, takewhile
from threading import Lock
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.accounting import (
    DeliveryTarget,
    Dimension,
    DimensionName,
    InboundUsageEvent,
    Metric,
    MetricName,
    OutboundUsageEvent,
    Outcome,
)
from litellm.constants import BILLED_RETRY_ATTEMPT_METADATA_KEY, INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.types.utils import CallTypes, CustomPricingLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.db.accounting_ledger import AccountingConflictError, UsageWrite


class UsageCaptureSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(default=1000, ge=2, le=1000)
    flush_interval_seconds: float = Field(default=0.2, gt=0, le=10)
    max_pending_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    max_pending_records: int = Field(default=10000, ge=2)


class UsageCaptureError(RuntimeError):
    pass


class UsageAcknowledgementError(HTTPException):
    def __init__(self) -> None:
        super().__init__(503, "Usage persistence did not acknowledge commit")


@dataclass(frozen=True, slots=True)
class _Pending:
    writes: tuple[UsageWrite, ...]
    size: int
    committed: Future[None]
    finished: asyncio.Event


async def _wait_for_batch(wakeup: asyncio.Event, timeout: float) -> None:
    try:
        await asyncio.wait_for(wakeup.wait(), timeout)
    except TimeoutError:
        return


def _report_failure(receipt: Future[None]) -> None:
    if not receipt.cancelled() and receipt.exception() is not None:
        verbose_proxy_logger.error("Usage capture did not acknowledge commit", exc_info=receipt.exception())


def _retryable_persistence_failure(error: Exception) -> bool:
    from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler, exception_chain

    outage: Final = PrismaDBExceptionHandler.find_database_service_unavailable_error_in_chain(error)
    return (outage is not None and not PrismaDBExceptionHandler.is_permanent_database_fault(outage)) or any(
        isinstance(cause, Exception)
        and (
            PrismaDBExceptionHandler.is_deadlock_error(cause)
            or PrismaDBExceptionHandler.is_lock_timeout_error(cause)
            or PrismaDBExceptionHandler.postgres_sqlstate(cause) == "40001"
        )
        for cause in exception_chain(error)
    )


class DurableUsageMeter:
    def __init__(
        self,
        write_batch: Callable[[tuple[UsageWrite, ...]], Awaitable[None]],
        settings: UsageCaptureSettings,
        *,
        wait_for_batch: Callable[[asyncio.Event, float], Awaitable[None]] = _wait_for_batch,
        retry_delay: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._write_batch: Final = write_batch
        self.settings: Final = settings
        self._wait_for_batch: Final = wait_for_batch
        self._retry_delay: Final = retry_delay
        self._loop: Final = asyncio.get_running_loop()
        self._admission: Final = Lock()
        self._queue: Final[deque[_Pending]] = deque()  # mutable-ok: one bounded producer/consumer queue
        self._wakeup: Final = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self.pending_bytes = 0
        self.pending_records = 0

    def submit(self, writes: tuple[UsageWrite, ...]) -> asyncio.Future[None]:
        size: Final = sum(len(write.envelope.encode("utf-8")) for write in writes)
        with self._admission:
            if (
                self._closed
                or not writes
                or len(writes) > self.settings.batch_size
                or self.pending_bytes + size > self.settings.max_pending_bytes
                or self.pending_records + len(writes) > self.settings.max_pending_records
            ):
                raise UsageCaptureError("Usage capture admission rejected: closed, empty or capacity exhausted")
            receipt: Final = Future[None]()
            receipt.set_running_or_notify_cancel()
            receipt.add_done_callback(_report_failure)
            self._queue.append(_Pending(writes, size, receipt, asyncio.Event()))
            self.pending_bytes += size
            self.pending_records += len(writes)
            self._loop.call_soon_threadsafe(self._ensure_worker)
        wrapped: Final = asyncio.wrap_future(receipt, loop=self._loop)
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return wrapped

    def _ensure_worker(self) -> None:
        if self.pending_records >= self.settings.batch_size:
            self._wakeup.set()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="accounting usage capture")

    async def record(self, event: InboundUsageEvent | OutboundUsageEvent) -> None:
        envelope: Final = '{"event":' + event.model_dump_json() + "}"
        await asyncio.shield(self.submit((UsageWrite(event, envelope, _targets(event.dimensions)),)))

    async def flush(self) -> None:
        self._ensure_worker()
        self._wakeup.set()
        if self._task is not None:
            await asyncio.shield(self._task)

    async def close(self) -> None:
        with self._admission:
            self._closed = True
        await self.flush()

    async def stop(self, timeout: float) -> None:
        with self._admission:
            self._closed = True
        try:
            await asyncio.wait_for(self.close(), timeout)
        except TimeoutError:
            if self._task is not None:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            while self._queue:
                self._finish(
                    self._take_batch(), UsageCaptureError("Usage capture stopped before commit acknowledgement")
                )
            verbose_proxy_logger.error("Usage capture shutdown deadline exhausted; commit acknowledgement incomplete")

    def _take_batch(self) -> tuple[_Pending, ...]:
        with self._admission:
            sizes: Final = accumulate(len(item.writes) for item in self._queue)
            count: Final = sum(1 for _ in takewhile(lambda size: size <= self.settings.batch_size, sizes))
            return tuple(self._queue.popleft() for _ in range(count))

    async def _persist(self, writes: tuple[UsageWrite, ...]) -> None:
        for attempt in count():
            try:
                await self._write_batch(writes)
            except Exception as error:  # noqa: BLE001  # retry only classified database failures, preserving frozen writes
                if not _retryable_persistence_failure(error):
                    raise
                verbose_proxy_logger.warning("Usage persistence unavailable; retaining admitted batch for retry")
                await self._retry_delay(min(10.0, 0.2 * 2 ** min(attempt, 6)))
            else:
                return

    async def _commit(self, batch: tuple[_Pending, ...]) -> None:
        writes: Final = tuple(chain.from_iterable(item.writes for item in batch))
        try:
            await self._persist(writes)
        except asyncio.CancelledError:
            self._finish(batch, UsageCaptureError("Usage commit acknowledgement interrupted"))
            raise
        except AccountingConflictError:
            if len(batch) > 1:
                try:
                    for item in batch:
                        await self._commit((item,))
                except asyncio.CancelledError:
                    self._finish(batch, UsageCaptureError("Usage commit acknowledgement interrupted"))
                    raise
                return
            self._finish(batch, UsageCaptureError("Usage identity conflicts with accepted content"))
        except Exception as error:  # noqa: BLE001  # persistence failures must reach every producer receipt
            self._finish(batch, error)
        else:
            self._finish(batch, None)

    def _finish(self, batch: tuple[_Pending, ...], error: Exception | None) -> None:
        for item in batch:
            if item.finished.is_set():
                continue
            item.finished.set()
            with self._admission:
                self.pending_bytes -= item.size
                self.pending_records -= len(item.writes)
            if item.committed.done():
                continue
            if error is None:
                item.committed.set_result(None)
            else:
                item.committed.set_exception(error)

    async def _run(self) -> None:
        try:
            while self._queue:
                await self._wait_for_batch(self._wakeup, self.settings.flush_interval_seconds)
                self._wakeup.clear()
                await self._commit(self._take_batch())
                if self._closed:
                    self._wakeup.set()
        except (Exception, asyncio.CancelledError):  # noqa: BLE001  # reject queued receipts if the writer exits abnormally
            while self._queue:
                self._finish(
                    self._take_batch(), UsageCaptureError("Usage capture stopped before commit acknowledgement")
                )
            raise


@dataclass(frozen=True, slots=True)
class CapturedUsage:
    payload: SpendLogsPayload
    writes: tuple[UsageWrite, ...]
    receipt: asyncio.Future[None]


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_FIELDS: Final = TypeAdapter(dict[str, object])
_INSTANT: Final = TypeAdapter(datetime)
_DIMENSIONS: Final = (
    ("api_key", DimensionName.KEY),
    ("user", DimensionName.USER),
    ("team_id", DimensionName.TEAM),
    ("organization_id", DimensionName.ORGANIZATION),
    ("end_user", DimensionName.END_USER),
    ("model", DimensionName.MODEL),
    ("model_group", DimensionName.MODEL_GROUP),
    ("model_id", DimensionName.DEPLOYMENT),
    ("custom_llm_provider", DimensionName.PROVIDER),
    ("mcp_namespaced_tool_name", DimensionName.TOOL),
    ("agent_id", DimensionName.AGENT),
)
_USAGE_METRICS: Final = tuple(
    metric
    for metric in MetricName
    if metric not in (MetricName.SPEND, MetricName.REQUESTS, MetricName.SUCCESSFUL_REQUESTS, MetricName.FAILED_REQUESTS)
)
_DAILY_TARGETS: Final = (
    (DimensionName.KEY, DeliveryTarget.DAILY_USER_SPEND),
    (DimensionName.USER, DeliveryTarget.DAILY_USER_SPEND),
    (DimensionName.TEAM, DeliveryTarget.DAILY_TEAM_SPEND),
    (DimensionName.ORGANIZATION, DeliveryTarget.DAILY_ORGANIZATION_SPEND),
    (DimensionName.END_USER, DeliveryTarget.DAILY_END_USER_SPEND),
    (DimensionName.AGENT, DeliveryTarget.DAILY_AGENT_SPEND),
    (DimensionName.TAG, DeliveryTarget.DAILY_TAG_SPEND),
    (DimensionName.MODEL, DeliveryTarget.DAILY_MODEL_USAGE),
    (DimensionName.TOOL, DeliveryTarget.DAILY_TOOL_SPEND),
)


def _fields(value: object) -> Mapping[str, object]:
    return _FIELDS.validate_python(value) if isinstance(value, Mapping) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _utc_time(value: datetime | str) -> datetime:
    parsed: Final = _INSTANT.validate_python(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _usage_metrics(usage: Mapping[str, JsonValue] | None) -> tuple[Metric, ...] | None:
    if usage is None:
        return None
    return tuple(
        Metric(name=metric, value=Decimal(str(value)))
        for metric in _USAGE_METRICS
        if isinstance(value := usage.get(metric.value), (int, float)) and not isinstance(value, bool)
    )


def _targets(dimensions: tuple[Dimension, ...]) -> tuple[DeliveryTarget, ...]:
    names: Final = frozenset(dimension.name for dimension in dimensions)
    return tuple(dict.fromkeys(target for dimension, target in _DAILY_TARGETS if dimension in names))


@dataclass(frozen=True, slots=True)
class UsageCaptureContext:
    caller_id: str | None
    attempt_id: str
    parent_request_id: str | None
    retry_of_attempt_id: str | None
    job_id: str | None
    charge_eligible: bool
    request_count_eligible: bool
    ownership: str
    pricing: str
    finalized_cost: str | None = None
    allocation: str | None = None
    outcome: Outcome | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    phase: str = "execution"
    extra_dimensions: tuple[Dimension, ...] = ()
    provider_timestamps: str = "{}"
    usage_source: str = "measured"
    included_attempt_ids: tuple[str, ...] = ()


def usage_writes(
    payload: SpendLogsPayload, usage_json: str | None, context: UsageCaptureContext
) -> tuple[UsageWrite, ...]:
    fields: Final = _FIELDS.validate_python(payload)
    tags: Final = TypeAdapter(tuple[str, ...]).validate_json(payload["request_tags"])
    dimensions: Final = (
        *(Dimension(name=name, value=value) for field, name in _DIMENSIONS if (value := _text(fields.get(field)))),
        *(Dimension(name=DimensionName.TAG, value=tag) for tag in dict.fromkeys(tags)),
        *context.extra_dimensions,
    )
    outcome: Final = context.outcome or (Outcome.FAILED if payload["status"] == "failure" else Outcome.SUCCEEDED)
    started: Final = context.started_at or _utc_time(payload["startTime"])
    ended: Final = context.ended_at or _utc_time(payload["endTime"])
    usage: Final = None if usage_json is None else _JSON_OBJECT.validate_json(usage_json)
    outbound: Final = OutboundUsageEvent(
        event_id=context.attempt_id,
        attempt_id=context.attempt_id,
        request_id=context.caller_id,
        retry_of_attempt_id=context.retry_of_attempt_id,
        call_type=CallTypes(payload["call_type"]),
        started_at=started,
        ended_at=ended,
        outcome=outcome,
        dimensions=dimensions,
        metrics=_usage_metrics(None if context.usage_source == "unknown" else usage),
    )
    envelope: Final = _json(
        {
            "event": outbound.model_dump(mode="json"),
            "usage": usage,
            "usage_source": "unknown" if usage is None else context.usage_source,
            "included_attempt_ids": context.included_attempt_ids,
            "pricing": _JSON_OBJECT.validate_json(context.pricing),
            "charge_eligible": context.charge_eligible,
            "request_count_eligible": context.request_count_eligible,
            "ownership": context.ownership,
            "parent_request_id": context.parent_request_id,
            "job_id": context.job_id,
            "phase": context.phase,
            "provider_timestamps": _JSON_OBJECT.validate_json(context.provider_timestamps),
            "finalized_cost": None
            if context.finalized_cost is None
            else _JSON_OBJECT.validate_json(context.finalized_cost),
            "allocation": None if context.allocation is None else _JSON_OBJECT.validate_json(context.allocation),
        }
    )
    outbound_write: Final = UsageWrite(outbound, envelope, _targets(dimensions))
    if not context.request_count_eligible or context.caller_id is None:
        return (outbound_write,)
    inbound: Final = InboundUsageEvent(
        event_id=f"{context.caller_id}:inbound",
        request_id=context.caller_id,
        call_type=outbound.call_type,
        started_at=started,
        ended_at=ended,
        outcome=outcome,
        dimensions=dimensions,
        metrics=(
            Metric(name=MetricName.REQUESTS, value=Decimal(1)),
            Metric(name=MetricName.SUCCESSFUL_REQUESTS, value=Decimal(int(outcome is Outcome.SUCCEEDED))),
            Metric(name=MetricName.FAILED_REQUESTS, value=Decimal(int(outcome is not Outcome.SUCCEEDED))),
        ),
    )
    return (
        UsageWrite(inbound, _json({"event": inbound.model_dump(mode="json")}), _targets(dimensions)),
        outbound_write,
    )


_INCLUDED_IN_PARENT: Final = object()

_PRICING_FIELDS: Final = frozenset(
    {
        "service_tier",
        "data_residency",
        "vertex_location",
        "region_name",
        "base_model",
        "custom_pricing",
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_second",
        "output_cost_per_second",
        "input_cost_per_character",
        "output_cost_per_character",
        "size",
        "quality",
        "n",
        "duration",
        "seconds",
        "search_context_size",
        "additional_response_cost",
    }
)


def _selected_pricing(fields: Mapping[str, object]) -> Mapping[str, JsonValue]:
    return _JSON_OBJECT.validate_python(
        {
            key: value
            for key, value in fields.items()
            if key in _PRICING_FIELDS or key in CustomPricingLiteLLMParams.model_fields
        }
    )


def _response_fields(response: object) -> Mapping[str, object]:
    if isinstance(response, BaseModel):
        return _FIELDS.validate_python(response.model_dump())
    return _fields(response)


def _job_time(response: Mapping[str, object], key: str) -> datetime | None:
    stamp: Final = response.get(key)
    if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
        return datetime.fromtimestamp(stamp, tz=timezone.utc)
    return None


def capture_context(payload: SpendLogsPayload, kwargs: Mapping[str, object], response: object) -> UsageCaptureContext:
    from dataclasses import asdict

    from litellm.litellm_core_utils.hidden_params import get_hidden_params
    from litellm.litellm_core_utils.internal_call_metadata import is_background_response, is_unbilled_non_inference_call
    from litellm.litellm_core_utils.litellm_logging import Logging, UsageObservation, mask_api_base_credentials
    from litellm.litellm_core_utils.llm_cost_calc.finalization import FinalizedCost
    from litellm.proxy.hooks.search_free_tier import SearchAllowanceAllocation
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_request_model_access_groups

    metadata: Final = _JSON_OBJECT.validate_json(payload["metadata"])
    params: Final = _fields(kwargs.get("litellm_params"))
    request_metadata: Final = _fields(params.get("litellm_metadata") or params.get("metadata"))
    standard: Final = _fields(kwargs.get("standard_logging_object"))
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    raw_response: Final = _response_fields(response)
    hidden: Final = _fields(get_hidden_params(response))
    tool: Final = _fields(kwargs.get("mcp_tool_call_metadata"))
    tool_pricing: Final = _fields(tool.get("mcp_server_cost_info"))
    caller: Final = payload["litellm_call_id"]
    cached: Final = payload["cache_hit"].lower() == "true"
    included_in_parent: Final = kwargs.get("_accounting_internal_aggregate") is _INCLUDED_IN_PARENT
    billed_id: Final = payload["request_id"]
    terminal: Final = raw_response.get("status") in {
        "completed",
        "complete",
        "failed",
        "cancelled",
        "expired",
        "incomplete",
    }
    deferred: Final = billed_id.endswith(("_batch_cost", "_background_cost")) and terminal
    management: Final = (
        (billed_id.endswith(("_batch_cost", "_background_cost")) and not terminal)
        or (is_background_response(response) and not terminal)
        or is_unbilled_non_inference_call(payload["call_type"], request_metadata, response)
    )
    attempt: Final = (
        f"{caller}_cache_hit"
        if cached and caller
        else f"{caller}:status"
        if management
        else f"{billed_id}:included"
        if included_in_parent
        else billed_id
    )
    if not attempt or attempt == "None" or (cached and not caller):
        raise UsageCaptureError("Usage capture requires a stable execution identity")
    retry: Final = metadata.get(BILLED_RETRY_ATTEMPT_METADATA_KEY) is True
    origin: Final = metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY)
    parent: Final = _text(kwargs.get("parent_request_id")) or _text(request_metadata.get("parent_request_id"))
    finalized_value: Final = (
        logging_obj.finalized_cost if isinstance(logging_obj, Logging) else kwargs.get("_accounting_finalized_cost")
    )
    finalized: Final = finalized_value if isinstance(finalized_value, FinalizedCost) else None
    allocation_value: Final = (
        logging_obj.cost_allowance_allocation
        if isinstance(logging_obj, Logging)
        else kwargs.get("_accounting_allocation")
    )
    allocation: Final = allocation_value if isinstance(allocation_value, SearchAllowanceAllocation) else None
    pricing_snapshot: Final = kwargs.get("_accounting_pricing_snapshot")
    captured_pricing: Final = {
        "model_map_information": (
            _JSON_OBJECT.validate_json(pricing_snapshot.model_map_information)
            if isinstance(pricing_snapshot, PricingCapture)
            else metadata.get("model_map_information")
        ),
        "cost_breakdown": metadata.get("cost_breakdown"),
        **(
            _JSON_OBJECT.validate_json(pricing_snapshot.inputs)
            if isinstance(pricing_snapshot, PricingCapture)
            else _pricing_inputs(kwargs, logging_obj, response)
        ),
        "tool": {
            "name": tool.get("name"),
            "cost_per_query": _fields(tool_pricing.get("tool_name_to_cost_per_query")).get(
                _text(tool.get("name")) or "", tool_pricing.get("default_cost_per_query")
            ),
        }
        if tool
        else None,
        "response_inputs": {
            "audio_transcription_duration": hidden.get("audio_transcription_duration", raw_response.get("duration")),
            "billed_units": _fields(raw_response.get("meta")).get("billed_units"),
            "batch_models": hidden.get("batch_models"),
            "batch_successful_requests": hidden.get("batch_successful_requests"),
            "batch_failed_requests": hidden.get("batch_failed_requests"),
            "service_tier": raw_response.get("service_tier"),
            "region_name": hidden.get("region_name"),
        },
        "reported_cost": standard.get(
            "response_cost", kwargs.get("response_cost", _fields(get_hidden_params(response)).get("response_cost"))
        ),
        "adjustments": (
            _JSON_OBJECT.validate_json(pricing_snapshot.adjustments)
            if isinstance(pricing_snapshot, PricingCapture)
            else None
        ),
    }
    return UsageCaptureContext(
        caller_id=None if deferred else parent if retry or included_in_parent else caller,
        attempt_id=attempt,
        parent_request_id=parent,
        retry_of_attempt_id=_text(request_metadata.get("retry_of_attempt_id")),
        job_id=_text(raw_response.get("id")) if deferred or management else None,
        charge_eligible=not cached and not included_in_parent and not management,
        request_count_eligible=bool(caller) and not origin and not retry and not deferred and not included_in_parent,
        ownership="included_in_parent"
        if included_in_parent
        else "aggregate"
        if kwargs.get("additional_response_cost") is not None
        else "attempt",
        pricing=_json(captured_pricing),
        finalized_cost=_json(
            {
                "response_cost": finalized.amount,
                "cost_breakdown": finalized.logging_breakdown(),
                "allowance_discount": finalized.allowance_discount,
            }
        )
        if finalized is not None
        else (
            _json({"response_cost": standard.get("response_cost"), "cost_breakdown": standard.get("cost_breakdown")})
            if standard.get("cost_finalized") is True
            else None
        ),
        allocation=_json(asdict(allocation)) if allocation is not None else None,
        started_at=_job_time(raw_response, "created_at") if deferred else None,
        ended_at=(
            _job_time(raw_response, "completed_at")
            or _job_time(raw_response, "failed_at")
            or _job_time(raw_response, "cancelled_at")
            or _job_time(raw_response, "expired_at")
        )
        if deferred
        else None,
        outcome=(
            Outcome.CANCELLED
            if raw_response.get("status") == "cancelled"
            or isinstance(response, asyncio.CancelledError)
            or getattr(response, "status_code", None) == 499
            else Outcome.FAILED
            if raw_response.get("status") in {"failed", "expired", "incomplete"} or isinstance(response, BaseException)
            else None
        ),
        phase="terminal" if deferred else "management" if management else "execution",
        usage_source="estimated"
        if kwargs.get("_accounting_usage_observed") is UsageObservation.ESTIMATED
        else "measured",
        included_attempt_ids=TypeAdapter(tuple[str, ...]).validate_python(
            kwargs.get("_accounting_included_attempt_ids", ())
        ),
        provider_timestamps=_json(
            {
                key: raw_response.get(key)
                for key in ("created_at", "completed_at", "failed_at", "cancelled_at", "expired_at")
                if raw_response.get(key) is not None
            }
        )
        if deferred
        else "{}",
        extra_dimensions=(
            *(
                (Dimension(name=DimensionName.TOOL, value=tool_name),)
                if not payload.get("mcp_namespaced_tool_name")
                and (tool_name := _text(tool.get("namespaced_tool_name")))
                else ()
            ),
            *(
                (Dimension(name=DimensionName.API_BASE, value=mask_api_base_credentials(api_base)),)
                if (api_base := _text(payload["api_base"]))
                else ()
            ),
            *(
                Dimension(name=DimensionName.MODEL_ACCESS_GROUP, value=group)
                for group in get_request_model_access_groups(kwargs)
            ),
            *(
                (Dimension(name=DimensionName.PROJECT, value=project),)
                if (project := _text(request_metadata.get("user_api_key_project_id")))
                else ()
            ),
            *(
                (Dimension(name=DimensionName.SERVICE_TIER, value=tier),)
                if (tier := _text(raw_response.get("service_tier") or kwargs.get("service_tier")))
                else ()
            ),
        ),
    )


class _CaptureBuilder:
    def __init__(self, kwargs: Mapping[str, object], response: object) -> None:
        self.kwargs: Final = kwargs
        self.response: Final = response
        self.writes: tuple[UsageWrite, ...] = ()

    def observe(self, payload: SpendLogsPayload, usage: str | None) -> None:
        parsed: Final = _JSON_OBJECT.validate_json(usage) if usage is not None else {}
        batch: Final = _fields(parsed.get("batch_usage_by_model"))
        unknown: Final = (bool(batch) or bool(parsed.get("batch_unknown_usage_requests"))) and all(
            value is None for value in batch.values()
        )
        observed: Final = (
            _json(
                {
                    "batch_usage_by_model": batch,
                    "batch_unknown_usage_requests": parsed.get("batch_unknown_usage_requests"),
                }
            )
            if unknown
            else usage
        )
        context: Final = capture_context(payload, self.kwargs, self.response)
        self.writes = usage_writes(payload, observed, replace(context, usage_source="unknown") if unknown else context)


def capture_callback(
    meter: DurableUsageMeter,
    kwargs: Mapping[str, object],
    response: object,
    start_time: datetime,
    end_time: datetime,
) -> CapturedUsage:
    from litellm.proxy.db.db_spend_update_writer import get_llm_router
    from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload

    existing: Final = kwargs.get("_accounting_capture")
    if isinstance(existing, CapturedUsage):
        return existing
    builder: Final = _CaptureBuilder(kwargs, response)
    payload: Final = get_logging_payload(
        dict(kwargs), response, start_time, end_time, llm_router=get_llm_router(), capture_usage=builder.observe
    )
    return CapturedUsage(payload, builder.writes, meter.submit(builder.writes))


def captured_callback_kwargs(
    kwargs: Mapping[str, object], response: object, start_time: datetime, end_time: datetime
) -> Mapping[str, object]:
    from litellm.proxy.proxy_server import usage_capture

    usage_meter: Final = usage_capture.meter

    if usage_meter is None:
        return kwargs
    try:
        return {**kwargs, "_accounting_capture": capture_callback(usage_meter, kwargs, response, start_time, end_time)}
    except Exception:  # noqa: BLE001  # capture failure must not suppress legacy billing
        verbose_proxy_logger.exception("Usage capture failed before legacy cost tracking")
        return kwargs


def validate_sync_ack_request(
    settings: Mapping[str, object], call_type: str, streaming: bool, background: bool = False
) -> None:
    if settings.get("accounting_sync_ack") is not True:
        return
    if streaming or background or call_type not in {"acompletion", "atext_completion", "aembedding", "aresponses"}:
        raise HTTPException(
            400, "accounting_sync_ack supports foreground non-stream chat, text, embedding and Responses requests"
        )


async def capture_proxy_response(logging_obj: object, response: object) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.proxy_server import general_settings, usage_capture

    settings: Final = _FIELDS.validate_python(general_settings)
    usage_meter: Final = usage_capture.meter

    if usage_meter is None or not isinstance(logging_obj, Logging):
        return
    try:
        capture_logging_usage(usage_meter, logging_obj, response, logging_obj.start_time, datetime.now(timezone.utc))
        captured: Final = logging_obj.model_call_details.get("_accounting_capture")
        if settings.get("accounting_sync_ack") is True:
            if not isinstance(captured, CapturedUsage):
                raise UsageCaptureError("Usage capture was not admitted")
            timeout: Final = TypeAdapter(float).validate_python(
                settings.get("accounting_sync_ack_timeout_seconds", 5.0)
            )
            await asyncio.wait_for(asyncio.shield(captured.receipt), timeout=timeout)
    except Exception as error:  # noqa: BLE001  # the configured response policy owns every capture failure
        verbose_proxy_logger.exception("Usage capture failed before the response boundary")
        if (
            settings.get("accounting_sync_ack") is True
            and settings.get("accounting_sync_ack_failure_policy", "fail_closed") == "fail_closed"
        ):
            raise UsageAcknowledgementError() from error


def capture_logging_usage(
    meter: DurableUsageMeter,
    logging_obj: object,
    response: object,
    start_time: datetime,
    end_time: datetime,
    freeze_only: bool = False,
) -> None:
    from litellm._internal_context import internal_parent_request_id, is_internal_call
    from litellm.batches.batch_utils import batch_cost_is_final
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import LiteLLMBatch

    if not isinstance(logging_obj, Logging):
        return
    internal: Final = is_internal_call.get()
    if isinstance(response, LiteLLMBatch) and batch_cost_is_final(response) and response.usage is None:
        return
    try:
        if not internal and isinstance(logging_obj.model_call_details.get("_accounting_capture"), CapturedUsage):
            logging_obj.record_usage_capture(
                capture_callback(meter, logging_obj.model_call_details, response, start_time, end_time)
            )
            return
        normalized_result: Final = logging_obj.usage_capture_response(response)
        prepared: Final = logging_obj.prepared_cost
        normalized: Final = (
            {**_response_fields(normalized_result), "usage": prepared.usage.model_dump()}
            if prepared is not None
            else normalized_result
        )
        if normalized is None:
            return
        details: Final = _FIELDS.validate_python(logging_obj.model_call_details)
        params: Final = _fields(details.get("litellm_params"))
        metadata: Final = _fields(params.get("litellm_metadata") or params.get("metadata"))
        if metadata.get("batch_ignore_default_logging") is True:
            return
        existing_pricing: Final = details.get("_accounting_pricing_snapshot")
        pricing: Final = (
            existing_pricing
            if isinstance(existing_pricing, PricingCapture) and not internal
            else _capture_pricing(logging_obj, normalized, details)
        )
        if freeze_only:
            if not internal:
                logging_obj.record_usage_pricing(pricing)
            return
        captured: Final = capture_callback(
            meter,
            {
                **details,
                "litellm_logging_obj": logging_obj,
                "_accounting_capture": None if internal else details.get("_accounting_capture"),
                "_accounting_pricing_snapshot": pricing,
                "_accounting_internal_aggregate": _INCLUDED_IN_PARENT if internal else None,
                "parent_request_id": internal_parent_request_id.get() if internal else details.get("parent_request_id"),
            },
            normalized,
            start_time,
            end_time,
        )
        if not internal:
            logging_obj.record_usage_capture(captured)
    except Exception:  # noqa: BLE001  # failed capture must not prevent normal logging and legacy charging
        verbose_proxy_logger.exception("Usage capture failed at the execution boundary")


@dataclass(frozen=True, slots=True)
class PricingCapture:
    model_map_information: str
    adjustments: str
    inputs: str


def _capture_pricing(logging_obj: Logging, normalized: object, details: Mapping[str, object]) -> PricingCapture:
    import litellm
    from litellm.litellm_core_utils.cost_discount import resolve_cost_discount
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    params: Final = _fields(details.get("litellm_params"))
    prepared: Final = logging_obj.prepared_cost
    provider: Final = _text(details.get("custom_llm_provider"))
    builtins: Final = logging_obj.standard_built_in_tools_params
    pricing: Final = (
        prepared.payload["model_map_information"]
        if prepared is not None
        else StandardLoggingPayloadSetup.get_model_cost_information(
            base_model=logging_obj.get_deployment_model_for_cost(),
            custom_pricing=bool(params.get("custom_pricing")),
            custom_llm_provider=_text(details.get("custom_llm_provider")),
            init_response_obj=normalized,
            api_base=_text(params.get("api_base")),
            router_model_id=logging_obj.get_router_model_id(),
        )
    )
    adjustments: Final = _json(
        {
            "discount": resolve_cost_discount(
                litellm.cost_discount_config,
                provider,
                logging_obj.get_deployment_model_for_cost(),
                _text(params.get("region_name") or params.get("vertex_location")),
            ),
            "margin": litellm.cost_margin_config.get(provider or "", litellm.cost_margin_config.get("global")),
            "web_search_options": (
                _selected_pricing(_fields(builtins.get("web_search_options")))
                if builtins.get("web_search_options") is not None
                else None
            ),
            "file_search": builtins.get("file_search") is not None,
        }
    )
    batch_usage: Final = _fields(_fields(_response_fields(normalized).get("usage")).get("batch_usage_by_model"))
    return PricingCapture(
        _json(pricing),
        adjustments,
        _json(
            {
                **_pricing_inputs(details, logging_obj, normalized),
                "batch_models": {
                    model: StandardLoggingPayloadSetup.get_model_cost_information(
                        base_model=model,
                        custom_pricing=bool(params.get("custom_pricing")),
                        custom_llm_provider=provider,
                        init_response_obj={"model": model},
                        api_base=_text(params.get("api_base")),
                        router_model_id=logging_obj.get_router_model_id(),
                    )
                    for model in batch_usage
                },
            }
        ),
    )


def _pricing_inputs(kwargs: Mapping[str, object], logging_obj: object, response: object) -> Mapping[str, object]:
    from litellm.litellm_core_utils.litellm_logging import Logging

    return {
        "request": _selected_pricing(kwargs),
        "deployment": _selected_pricing(_fields(kwargs.get("litellm_params"))),
        "optional_params": _selected_pricing(_fields(kwargs.get("optional_params"))),
        "logging": logging_obj.usage_capture_pricing(response) if isinstance(logging_obj, Logging) else None,
        "model_info": _selected_pricing(
            _fields(
                _fields(
                    _fields(kwargs.get("litellm_params")).get("litellm_metadata")
                    or _fields(kwargs.get("litellm_params")).get("metadata")
                ).get("model_info")
            )
        ),
    }


def _terminal_facts(write: UsageWrite) -> Mapping[str, JsonValue] | None:
    payload: Final = _JSON_OBJECT.validate_json(write.envelope)
    if payload.get("phase") != "terminal":
        return None
    event: Final = _JSON_OBJECT.validate_python(payload["event"])
    return {
        **{
            key: value
            for key, value in payload.items()
            if key not in {"event", "pricing", "finalized_cost", "allocation"}
        },
        "event": {key: value for key, value in event.items() if key not in {"started_at", "ended_at"}},
    }


def _reuse_terminal_snapshot(candidate: UsageWrite, accepted: UsageWrite) -> UsageWrite:
    facts: Final = _terminal_facts(candidate)
    if facts is None or facts != _terminal_facts(accepted) or set(candidate.targets) != set(accepted.targets):
        raise AccountingConflictError("Repeated terminal job disagrees with its accepted usage or attribution")
    return accepted


class UsageBatchWriter:
    def __init__(
        self,
        insert: Callable[[tuple[UsageWrite, ...]], Awaitable[None]],
        load: Callable[[tuple[str, ...]], Awaitable[tuple[UsageWrite, ...]]],
    ) -> None:
        self._insert: Final = insert
        self._load: Final = load

    async def _restore(self, writes: tuple[UsageWrite, ...]) -> tuple[UsageWrite, ...]:
        terminal: Final = tuple(write for write in writes if _terminal_facts(write) is not None)
        if not terminal:
            return writes
        identities: Final = tuple(dict.fromkeys(write.event.event_id for write in terminal))
        stored: Final = await self._load(identities)
        # Re-read the first accepted observation and pricing before the ledger's unchanged id/hash check.
        snapshots: Final = {write.event.event_id: write for write in (*reversed(terminal), *stored)}
        return tuple(
            _reuse_terminal_snapshot(write, snapshots[write.event.event_id])
            if write.event.event_id in snapshots
            else write
            for write in writes
        )

    async def write(self, writes: tuple[UsageWrite, ...]) -> None:
        restored: Final = await self._restore(writes)
        try:
            await self._insert(restored)
        except AccountingConflictError:
            if not any(_terminal_facts(write) is not None for write in writes):
                raise
            await self._insert(await self._restore(writes))


class UsageCaptureRuntime:
    def __init__(self) -> None:
        self.meter: DurableUsageMeter | None = None

    def start(
        self, write_batch: Callable[[tuple[UsageWrite, ...]], Awaitable[None]], settings: UsageCaptureSettings
    ) -> None:
        from litellm.litellm_core_utils.litellm_logging import Logging

        if self.meter is not None:
            if self.meter.settings != settings:
                raise ValueError("Changing accounting_capture requires a worker restart")
            return
        meter: Final = DurableUsageMeter(write_batch, settings)
        self.meter = meter
        Logging.usage_capture_callback = lambda logger, response, start, end, freeze: capture_logging_usage(
            meter, logger, response, start, end, freeze
        )

    async def stop(self, timeout: float) -> None:
        from litellm.litellm_core_utils.litellm_logging import Logging

        if self.meter is None:
            return
        try:
            await self.meter.stop(timeout)
        finally:
            Logging.usage_capture_callback = None
            self.meter = None
