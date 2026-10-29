"""
``RustCore``: the router core served by litellm-rust/crates/router-core through ``_native``.

The Rust core owns the registry, selection, retries, fallbacks and cooldowns for the operations in
``RUST_SERVED_OPS``. It never awaits Python: ``RouterCall.next`` returns the step the call loop is
blocked on (run an attempt on a deployment, sleep, check a fallback target, run a plugin) and this
driver answers it. Each attempt runs through the Python core with that one deployment, so request
preparation, inference, logging and cost tracking are unchanged.

Everything else (the views, runtime writes, every other endpoint) is served by the ``PythonCore``
the Rust core wraps, built from the same config. A config the Rust core cannot serve fails at
construction and ``make_core`` builds a ``PythonCore`` instead.
"""

import asyncio
import inspect
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_router_logger
from litellm.router_config import RouterConfig
from litellm.router_python_core import PythonCore
from litellm.router_python_host import PythonHost

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.rust_bridge._native import RouterCall, RouterCoreHandle

RustServedOp: TypeAlias = Literal["acompletion", "aanthropic_messages", "aembedding", "aresponses"]
RUST_SERVED_OPS: Final[frozenset[str]] = frozenset({"acompletion", "aanthropic_messages", "aembedding", "aresponses"})

_FAILURE_KINDS: Final = (
    "RateLimitError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "Timeout",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "GuardrailIntervention",
    "CallerCredentialError",
)
_DECLINED: Final = frozenset({"MalformedRequestFieldError", "UnsupportedEndpointError"})
_STEP: Final = TypeAdapter(Mapping[str, object])
_REQUEST_FIELDS: Final = ("specific_deployment", "num_retries", "disable_fallbacks")
_PYTHON_ONLY_REQUEST_KEYS: Final = frozenset(
    {
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "default_fallbacks",
        "max_fallbacks",
        "model_group_retry_policy",
        "retry_policy",
    }
)
_STATIC_FIELDS: Final = (
    "routing_strategy",
    "cooldown_time",
    "disable_cooldowns",
    "allowed_fails_policy",
    "retry_after",
    "timeout",
)
_HANDLED_FIELDS: Final = frozenset(
    {
        *_STATIC_FIELDS,
        "model_list",
        "model_group_alias",
        "fallbacks",
        "default_fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "max_fallbacks",
        "allowed_fails",
        "set_verbose",
        "debug_level",
    }
)
_PYTHON_DEFAULTS: Final = MappingProxyType(
    {
        name: parameter.default
        for name, parameter in inspect.signature(PythonCore.__init__).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
)


def failure_kind(error: BaseException) -> str:
    names: Final = tuple(cls.__name__ for cls in type(error).__mro__)
    if not _DECLINED.isdisjoint(names):
        return "Declined"
    return next((name for name in names if name in _FAILURE_KINDS), "APIError")


def _retry_after_seconds(error: BaseException) -> float | None:
    response: Final = getattr(error, "response", None)
    headers: Final = getattr(error, "litellm_response_headers", None) or getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    milliseconds: Final = headers.get("retry-after-ms")
    seconds: Final = headers.get("retry-after")
    try:
        value: Final = float(milliseconds) / 1000 if milliseconds is not None else float(seconds)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def attempt_failure(error: BaseException) -> Mapping[str, object]:
    status: Final = getattr(error, "status_code", None)
    num_retries: Final = getattr(error, "num_retries", None)
    return {
        "ok": False,
        "kind": failure_kind(error),
        "status": status if isinstance(status, int) and 100 <= status <= 599 else None,
        "retry_after": _retry_after_seconds(error),
        "num_retries": num_retries if isinstance(num_retries, int) and num_retries >= 0 else None,
        "detail": str(error)[:500],
    }


def total_tokens(response: object) -> int:
    usage: Final = getattr(response, "usage", None)
    tokens: Final = getattr(usage, "total_tokens", None)
    return tokens if isinstance(tokens, int) else 0


def _json_default(value: object) -> object:
    return str(value)


def _plain(value: object) -> object:
    """``value`` as plain JSON values, the shape the Rust core reads."""
    return json.loads(json.dumps(value, default=_json_default))


def static_document(config: RouterConfig, python: PythonCore) -> Mapping[str, object]:
    """The Python core's own normalized deployments (with their ids) and the static settings."""
    return {
        "model_list": _plain(python._core_model_list),
        **{name: _plain(config.arguments[name]) for name in _STATIC_FIELDS if name in config.arguments},
    }


def dynamic_document(python: PythonCore) -> Mapping[str, object]:
    """The values the Python core resolved from its arguments and the process settings."""
    settings: Final = python.settings()
    resolved: Final = {
        "version": python._registry_version,
        "model_group_alias": python._core_model_group_alias,
        "fallbacks": settings.fallbacks,
        "context_window_fallbacks": settings.context_window_fallbacks,
        "content_policy_fallbacks": settings.content_policy_fallbacks,
        "num_retries": python.num_retries,
        "max_fallbacks": python.max_fallbacks,
        "allowed_fails": python.explicit_allowed_fails,
    }
    return {name: _plain(value) for name, value in resolved.items() if value is not None}


def unsupported_arguments(config: RouterConfig) -> frozenset[str]:
    """Arguments the Rust core does not serve; any of them keeps the router on Python."""
    changed: Final = frozenset(
        name
        for name, value in config.arguments.items()
        if name not in _HANDLED_FIELDS and value != _PYTHON_DEFAULTS.get(name, None)
    )
    strategy: Final = config.arguments.get("routing_strategy", "simple-shuffle")
    return changed | ({"routing_strategy"} if strategy != "simple-shuffle" else frozenset())


def runs_on_python(request: Mapping[str, object]) -> bool:
    """Per-request routing overrides the Rust core does not read yet."""
    return any(key in _PYTHON_ONLY_REQUEST_KEYS or key.startswith("mock_testing_") for key in request)


def request_team_id(request: Mapping[str, object]) -> str | None:
    metadata: Final = request.get("metadata") or request.get("litellm_metadata")
    team_id: Final = metadata.get("user_api_key_team_id") if isinstance(metadata, Mapping) else None
    return team_id if isinstance(team_id, str) else None


_OWN_ATTRIBUTES: Final = frozenset({"_handle", "_config", "_python", "_host", "_built_for"})


class RustCore:
    """A ``RouterCore`` whose routing for ``RUST_SERVED_OPS`` runs in Rust, over a ``PythonCore``."""

    def __init__(self, handle: "RouterCoreHandle", config: RouterConfig, python: PythonCore, host: PythonHost) -> None:
        self._handle: RouterCoreHandle | None = handle
        self._config: Final = config
        self._python: Final = python
        self._host: Final = host
        self._built_for: tuple[int, object] = self._state()

    @classmethod
    def build(cls, config: RouterConfig, host: PythonHost, python_class: type[PythonCore] = PythonCore) -> "RustCore":
        from litellm.rust_bridge._native import RouterCoreHandle, RustBridgeDeclined

        extra: Final = unsupported_arguments(config)
        if extra:
            raise RustBridgeDeclined(f"router arguments not served by the Rust core: {sorted(extra)}")
        python: Final = python_class(**config.arguments, _host=host)  # pyright: ignore[reportArgumentType]  # the arguments were bound against PythonCore's own signature
        handle: Final = RouterCoreHandle(static_document(config, python), dynamic_document(python))
        return cls(handle, config, python, host)

    def __getattr__(self, name: str) -> object:
        return getattr(self._python, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in _OWN_ATTRIBUTES:
            object.__setattr__(self, name, value)
            return
        setattr(self._python, name, value)

    def __delattr__(self, name: str) -> None:
        if name in _OWN_ATTRIBUTES:
            object.__delattr__(self, name)
            return
        delattr(self._python, name)

    def _state(self) -> tuple[int, object]:
        # The snapshot itself, not its id: a dropped snapshot's id can be reused by the next one.
        return (self._python._registry_version, self._python.settings())

    def _current_handle(self) -> "RouterCoreHandle | None":
        """The Rust handle, rebuilt after any deployment or settings change; ``None`` once Rust cannot serve it."""
        version, snapshot = self._state()
        if self._handle is None or (version == self._built_for[0] and snapshot is self._built_for[1]):
            return self._handle
        from litellm.rust_bridge._native import RouterCoreHandle

        self._handle.retire()
        try:
            self._handle = RouterCoreHandle(static_document(self._config, self._python), dynamic_document(self._python))
        except Exception as declined:  # noqa: BLE001  # a change the Rust core cannot take moves the router to Python for good
            verbose_router_logger.info("Router moved to the Python core after a change: %s", declined)
            self._handle = None
        self._built_for = self._state()
        return self._handle

    async def call(self, op: str, request: Mapping[str, object]) -> object:
        if op not in RUST_SERVED_OPS:
            return await self._python.call(op, request)
        return await self._drive(op, request)

    async def acompletion(
        self, model: str, messages: object, **kwargs: object
    ) -> object:  # kwargs-ok: the endpoint's own keyword payload
        return await self._drive("acompletion", {"model": model, "messages": messages, **kwargs})

    async def aanthropic_messages(self, **kwargs: object) -> object:  # kwargs-ok: the endpoint's own keyword payload
        return await self._drive("aanthropic_messages", kwargs)

    async def aembedding(
        self, model: str, input: object, **kwargs: object
    ) -> object:  # kwargs-ok: the endpoint's own keyword payload
        return await self._drive("aembedding", {"model": model, "input": input, **kwargs})

    async def aresponses(self, **kwargs: object) -> object:  # kwargs-ok: the endpoint's own keyword payload
        return await self._drive("aresponses", kwargs)

    def retire(self) -> None:
        if self._handle is not None:
            self._handle.retire()
        self._python.retire()

    async def _drive(self, op: str, request: Mapping[str, object]) -> object:
        handle: Final = self._current_handle()
        if handle is None or runs_on_python(request):
            return await self._python.call(op, request)
        fields: Final = {key: request[key] for key in _REQUEST_FIELDS if request.get(key) is not None}
        team_id: Final = request_team_id(request)
        try:
            call: Final = handle.start(
                op, _plain({"model": request["model"], **fields, **({"team_id": team_id} if team_id else {})})
            )
        except ValueError:
            return await self._python.call(op, request)
        return await _Driver(call, self._python, op, request).run()


def _ttl(step: Mapping[str, object]) -> Mapping[str, object]:
    ttl: Final = step.get("ttl")
    return {"ttl": ttl} if isinstance(ttl, int | float) else {}


def _store_for(python: PythonCore, key: str) -> "DualCache":
    """Cooldowns live in the router's cooldown store; every other routing key in its main DualCache."""
    health: Final = python.health()
    return health.cooldowns.cooldown_store if key.endswith(":cooldown") else health.store


async def answer_cache_step(python: PythonCore, step: Mapping[str, object]) -> object:
    """Run one cache op the Rust core asked for against the router's own DualCaches (memory plus Redis)."""
    key: Final = str(step.get("key"))
    match step.get("op"):
        case "get":
            return _plain(await _store_for(python, key).async_get_cache(key))
        case "batch_get":
            keys: Final = [str(k) for k in step.get("keys") or ()]
            values: Final = [await _store_for(python, k).async_get_cache(k) for k in keys]
            return _plain(values)
        case "set":
            await _store_for(python, key).async_set_cache(key, step.get("value"), **_ttl(step))
            return None
        case "increment":
            amount: Final = step.get("value")
            by: Final = amount if isinstance(amount, int | float) and not isinstance(amount, bool) else 1
            total: Final = await _store_for(python, key).async_increment_cache(key, by, **_ttl(step))
            return total if isinstance(total, int | float) else by
        case "delete":
            await _store_for(python, key).async_delete_cache(key)
            return None
        case other:
            raise RuntimeError(f"router-core asked for an unknown cache op {other!r}")


class _Driver:
    """Answers the steps one Rust call loop blocks on until it is done."""

    def __init__(self, call: "RouterCall", python: PythonCore, op: str, request: Mapping[str, object]) -> None:
        self._call: Final = call
        self._python: Final = python
        self._op: Final = op
        self._request: Final = request
        self._last_response: object = None
        self._last_error: BaseException | None = None
        self._primary_error: BaseException | None = None
        self._attempted: bool = False
        self._primary_budget: int | None = None

    async def run(self) -> object:
        outcome: object = None
        while True:
            step: Final = _STEP.validate_python(self._call.next(outcome))
            match step["kind"]:
                case "attempt":
                    outcome = await self._attempt(step)  # rebind-ok: the outcome answers this step
                case "sleep":
                    await asyncio.sleep(float(str(step["seconds"])))
                    outcome = None  # rebind-ok: a sleep has no outcome
                case "allow_fallback":
                    outcome = await self._allow_fallback(
                        str(step["target"])
                    )  # rebind-ok: the outcome answers this step
                case "cache":
                    outcome = await answer_cache_step(self._python, step)  # rebind-ok: the outcome answers this step
                case "select":
                    outcome = step["candidates"]  # rebind-ok: plugins are declined at build, so this is never a pick
                case "done":
                    return self._last_response
                case "failed":
                    if self._last_error is not None:
                        raise self._with_retry_counts(self._primary_error or self._last_error)
                    if not self._attempted:
                        # No deployment was tried (unknown model, nothing healthy): Python raises its own error.
                        return await self._python.call(self._op, self._request)
                    raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    def _with_retry_counts(self, error: BaseException) -> BaseException:
        """The final error carrying the retry budget Python stamps on it (``num_retries``/``max_retries``)."""
        requested: Final = self._request.get("num_retries")
        deployment_budget: Final = self._primary_budget
        budget: Final = (
            deployment_budget
            if deployment_budget is not None
            else requested
            if isinstance(requested, int)
            else self._python.num_retries
        )
        if not isinstance(budget, int) or budget <= 0:
            return error
        error.num_retries = budget  # pyright: ignore[reportAttributeAccessIssue]  # Python's router stamps the same attributes
        error.max_retries = budget  # pyright: ignore[reportAttributeAccessIssue]  # Python's router stamps the same attributes
        return error

    def _deployment_num_retries(self, deployment_id: str) -> int | None:
        deployment: Final = self._python.registry().deployment(deployment_id)
        value: Final = getattr(deployment.litellm_params, "num_retries", None) if deployment is not None else None
        return value if isinstance(value, int) and value >= 0 else None

    def _pinned_request(self, deployment_id: str, model_group: str) -> Mapping[str, object]:
        """The request pinned to one deployment, keeping its model group so logs and headers name the group."""
        group_ids: Final = tuple(self._python.registry().names(model_group, kind="ids") or ())
        if deployment_id not in group_ids or self._request.get("model") == deployment_id:
            return {**self._request, "model": deployment_id}
        return {
            **self._request,
            "model": model_group,
            "_retry_skipped_deployment_ids": tuple(i for i in group_ids if i != deployment_id),
        }

    async def _attempt(self, step: Mapping[str, object]) -> Mapping[str, object]:
        self._attempted = True
        request: Final = self._pinned_request(str(step["deployment_id"]), str(step["model_group"]))
        try:
            response: Final = await self._python.call(
                self._op, {**request, "num_retries": 0, "disable_fallbacks": True}
            )
        except Exception as error:  # noqa: BLE001  # every attempt failure is classified and handed to the Rust call loop
            self._last_error = error
            if step.get("fallback_depth") == 0:
                self._primary_error = error
            deployment_retries: Final = self._deployment_num_retries(str(step["deployment_id"]))
            if step.get("fallback_depth") == 0:
                self._primary_budget = deployment_retries
            failure: Final = attempt_failure(error)
            return failure if deployment_retries is None else {**failure, "num_retries": deployment_retries}
        self._last_response = response
        return {"ok": True, "total_tokens": total_tokens(response)}

    async def _allow_fallback(self, target: str) -> bool:
        ctx: Final = self._python._core_ctx
        request: Final = dict(self._request)
        return await ctx.fallback_access_allowed(
            model=target, request_kwargs=request
        ) and await ctx.fallback_budget_allowed(model=target, request_kwargs=request)
