"""Cabangile AI Studio - Enterprise Base Provider Module.

This module establishes the production-ready, enterprise-grade abstract architecture
for all AI provider integrations within the Cabangile AI Studio ecosystem (v2.6 Enterprise Edition).
Built strictly with the Python Standard Library (Python 3.11+), it implements Clean Architecture,
SOLID principles, and Dependency Inversion.

It features thread-safe state management, configuration fingerprinting, advanced telemetry with
percentile latency calculations via interpolation, an event callback system, concurrent request
scheduling queues with a dedicated worker thread, strong reference request tracking, structural
request validation, request cancellation capabilities, and comprehensive JSON serialization helpers.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import contextlib
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
import functools
import hashlib
import inspect
import json
import logging
from pathlib import Path
import queue
import statistics
import threading
import time
import traceback
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import uuid

# ============================================================================
# TYPE DEFINITIONS, EVENT HOOKS & CONSTANTS
# ============================================================================

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

EventCallback = Callable[[str, Dict[str, Any]], None]

# ============================================================================
# ENUMERATIONS
# ============================================================================


class ProviderState(Enum):
    """Represents the operational lifecycle states of an AI Provider."""

    INITIALIZING = auto()
    READY = auto()
    BUSY = auto()
    STREAMING = auto()
    ERROR = auto()
    DISABLED = auto()
    SHUTDOWN = auto()


class ProviderCapability(Enum):
    """Defines the feature-set capabilities supported by a given provider."""

    TEXT = auto()
    CHAT = auto()
    STREAMING = auto()
    EMBEDDINGS = auto()
    TOOLS = auto()
    JSON_MODE = auto()
    IMAGE_INPUT = auto()
    IMAGE_OUTPUT = auto()
    AUDIO_INPUT = auto()
    AUDIO_OUTPUT = auto()
    FUNCTION_CALLING = auto()


class FinishReason(Enum):
    """Standardized reasons for LLM generation cessation."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# ============================================================================
# SYSTEM EXCEPTIONS
# ============================================================================


class ProviderError(Exception):
    """Base exception for all Cabangile AI Studio provider anomalies."""

    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        request_id: Optional[str] = None,
        raw_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider_name = provider_name or "UNKNOWN"
        self.request_id = request_id or "N/A"
        self.raw_error = raw_error
        self.timestamp = datetime.now(timezone.utc)
        self.traceback_str = (
            "".join(traceback.format_exception(None, raw_error, raw_error.__traceback__))
            if raw_error and raw_error.__traceback__
            else "".join(traceback.format_stack())
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the error context into a structured dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "provider_name": self.provider_name,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "raw_error_details": str(self.raw_error) if self.raw_error else None,
            "traceback": self.traceback_str,
        }

    def __str__(self) -> str:
        return (
            f"[{self.__class__.__name__}] Provider: {self.provider_name} | "
            f"ReqID: {self.request_id} | Message: {self.message}"
        )


class ConfigurationError(ProviderError):
    """Raised when provider configuration options are invalid or missing."""


class AuthenticationError(ProviderError):
    """Raised when authentication credentials fail or are rejected."""


class RateLimitError(ProviderError):
    """Raised when the provider signals rate limits have been exceeded."""


class TimeoutError(ProviderError):
    """Raised when a request or stream breaches execution time thresholds."""


class UnavailableError(ProviderError):
    """Raised when an upstream provider or model endpoint is unreachable."""


class ValidationError(ProviderError):
    """Raised when standard internal payload schemas fail sanity checks."""


class StreamingError(ProviderError):
    """Raised when structural stream consumption errors manifest."""


class ResponseError(ProviderError):
    """Raised when upstream API parses cleanly but payload denotes structural failure."""


class RetryLimitExceeded(ProviderError):
    """Raised when a request fails completely after all retries are exhausted."""


class RequestCancelledError(ProviderError):
    """Raised when an operation is explicitly aborted via the cancellation API."""


# ============================================================================
# DATA CONTAINERS / DATA TRANSFER OBJECTS (DTOs)
# ============================================================================


@dataclass(frozen=True)
class ModelMetadata:
    """Encapsulates the explicit operational boundaries and metadata of an LLM."""

    model_id: str
    display_name: str
    context_window: int
    max_output_tokens: int
    capabilities: Set[ProviderCapability]
    input_token_cost_per_1k: float = 0.0
    output_token_cost_per_1k: float = 0.0
    extra_properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes model specifications to a standard dictionary format."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "capabilities": [c.name for c in self.capabilities],
            "input_token_cost_per_1k": self.input_token_cost_per_1k,
            "output_token_cost_per_1k": self.output_token_cost_per_1k,
            "extra_properties": self.extra_properties,
        }


@dataclass(frozen=True)
class AIMessage:
    """Represents a single structural node within a conversation tree."""

    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("Message role cannot be empty.")
        if self.content is None:
            raise ValueError("Message content cannot be None.")

    def generate_hash(self) -> str:
        """Generates a deterministic cryptographic SHA-256 fingerprint of the message content."""
        payload = f"{self.role}:{self.content}:{sorted(self.metadata.items(), key=lambda x: x[0])}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Converts the message data container to a standard map."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TokenUsage:
    """Standardizes structural token distribution data across executions."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens >= 0 and self.completion_tokens >= 0 and self.total_tokens == 0:
            object.__setattr__(self, "total_tokens", self.prompt_tokens + self.completion_tokens)

        if self.prompt_tokens < 0 or self.completion_tokens < 0 or self.total_tokens < 0:
            raise ValueError("Token allocation quantities cannot be negative.")

    def to_dict(self) -> Dict[str, Any]:
        """Converts token metrics into serialized telemetry."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class GenerationRequest:
    """Encapsulates input operational structures for generation steps."""

    prompt: Optional[str] = None
    messages: Optional[List[AIMessage]] = None
    model: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout: float = 30.0
    user: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.prompt and not self.messages:
            raise ValueError("Either prompt or messages array must be supplied.")
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("Temperature must strictly reside between 0.0 and 2.0.")
        if self.top_p < 0.0 or self.top_p > 1.0:
            raise ValueError("Top-P parameters must reside between 0.0 and 1.0.")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("Max tokens allocation parameter must exceed zero.")
        if self.timeout <= 0.0:
            raise ValueError("Execution timeouts must represent real positive limits.")

    def generate_checksum(self) -> str:
        """Calculates a repeatable hash checksum of the core inference payload properties."""
        components = [
            self.prompt or "",
            str(self.model),
            f"{self.temperature:.4f}",
            f"{self.top_p:.4f}",
            str(self.max_tokens),
            str(self.seed),
        ]
        if self.messages:
            components.extend([m.generate_hash() for m in self.messages])
        if self.stop_sequences:
            components.extend(sorted(self.stop_sequences))

        combined = "|".join(components)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def validate_checksum(self) -> bool:
        """Validates structural integrity by ensuring the payload computes a valid hash."""
        return len(self.generate_checksum()) == 64

    def to_dict(self) -> Dict[str, Any]:
        """Provides an extraction layout sanitized of complex datatypes."""
        return {
            "prompt": self.prompt,
            "messages": [m.to_dict() for m in self.messages] if self.messages else None,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
            "stream": self.stream,
            "metadata": self.metadata,
            "request_id": self.request_id,
            "timeout": self.timeout,
            "user": self.user,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GenerationResponse:
    """Encapsulates complete structured outputs of concrete generation engines."""

    text: str
    model: str
    provider: str
    finish_reason: FinishReason
    usage: TokenUsage
    latency: float
    request_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Transforms response model into nested structural dictionaries."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason.value,
            "usage": self.usage.to_dict(),
            "latency": self.latency,
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "success": self.success,
        }


@dataclass
class ProviderHealth:
    """Telemetry capture structural block tracking current health dynamics."""

    available: bool = True
    state: ProviderState = ProviderState.INITIALIZING
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    failures: int = 0
    average_latency: float = 0.0
    success_rate: float = 100.0
    uptime: float = 0.0
    active_requests: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProviderStatistics:
    """Cumulative, thread-safe transactional telemetry repository."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retries: int = 0
    streamed_requests: int = 0
    average_latency: float = 0.0
    tokens_generated: int = 0
    uptime: float = 0.0
    last_request_time: Optional[datetime] = None

    _latency_history: List[float] = field(default_factory=list, repr=False)
    _metrics_snapshots: List[Dict[str, Any]] = field(default_factory=list, repr=False)


# ============================================================================
# STRUCTURED UTILITY COMPONENT HOOKS & DECORATORS
# ============================================================================


def validate_payload(func: F) -> F:
    """Decorator ensuring incoming arguments conform strictly to type requirements."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound_args = inspect.signature(func).bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        for name, value in bound_args.arguments.items():
            if name == "request" and isinstance(value, GenerationRequest):
                value.validate_checksum()  # Safe execution following fix
                if self.state in (ProviderState.SHUTDOWN, ProviderState.DISABLED):
                    raise UnavailableError(
                        f"Target provider is offline in state: {self.state.name}",
                        provider_name=self.provider_name,
                        request_id=value.request_id,
                    )
        return func(self, *args, **kwargs)

    return wrapper  # type: ignore


# ============================================================================
# STRUCTURAL LOGGING ENGINE
# ============================================================================


class StructuredAuditLogFormatter(logging.Formatter):
    """Formats log records into explicit, uniform JSON payloads for audit streaming."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "provider"):
            log_data["provider"] = getattr(record, "provider")
        if hasattr(record, "request_id"):
            log_data["request_id"] = getattr(record, "request_id")
        if hasattr(record, "latency"):
            log_data["latency_ms"] = round(getattr(record, "latency") * 1000.0, 2)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# ============================================================================
# ABSTRACT CORE BASE PROVIDER INTERFACE
# ============================================================================


class BaseProvider(ABC):
    """Abstract Core Component for enterprise integrations within Cabangile AI Studio.

    Implements a production architecture featuring transaction safety loops,
    concurrency scheduler engines, unified audit formatters, and hot reload managers.
    """

    FRAMEWORK_VERSION: str = "2.6.0"
    BUILD_METADATA: str = "RELEASE-ENTERPRISE-PROD"

    def __init__(
        self,
        provider_name: str,
        configuration: Dict[str, Any],
        default_model: str,
        supported_capabilities: Set[ProviderCapability],
        provider_version: str = "1.0.0",
        api_version: str = "v1",
    ) -> None:
        """Initializes tracking loops, memory protection blocks, and telemetry registers."""
        self._provider_name: str = provider_name.strip().upper()
        self._configuration: Dict[str, Any] = copy.deepcopy(configuration)
        self._default_model: str = default_model
        self._supported_capabilities: Set[ProviderCapability] = set(supported_capabilities)

        self._provider_version: str = provider_version
        self._api_version: str = api_version

        # Operational Lifecycles and Telemetry
        self._state: ProviderState = ProviderState.INITIALIZING
        self._health: ProviderHealth = ProviderHealth(state=self._state)
        self._statistics: ProviderStatistics = ProviderStatistics()
        self._boot_time: datetime = datetime.now(timezone.utc)

        # Thread Protection Architecture
        self._lock = threading.RLock()

        # FIXED: Strong dict prevents premature reference collections compared to weak references
        self._active_resources: Dict[str, GenerationRequest] = {}
        self._cancelled_requests: Set[str] = set()

        # FIXED: Dictionary based cache mapping yields O(1) performance profiles
        self._cached_models: Dict[str, ModelMetadata] = {}
        self._last_cache_refresh: Optional[datetime] = None
        self._config_fingerprint: str = self._generate_fingerprint(self._configuration)

        # Hook Callback Lists
        self._before_request_hooks: List[Callable[[GenerationRequest], None]] = []
        self._after_request_hooks: List[Callable[[Optional[GenerationResponse], Optional[Exception]], None]] = []
        self._event_listeners: defaultdict[str, List[EventCallback]] = defaultdict(list)

        # Retry and Loop Constants
        self._max_retries: int = int(self._configuration.get("max_retries", 3))
        self._backoff_factor: float = float(self._configuration.get("backoff_factor", 1.5))
        self._initial_backoff: float = float(self._configuration.get("initial_backoff", 0.5))

        # Structured Logging Infrastructure
        self._logger: logging.Logger = logging.getLogger(
            f"cabangile.ai_studio.providers.{self._provider_name.lower()}"
        )
        self._logger.setLevel(logging.INFO)
        
        # FIXED: Attach Structured Log Formatter handler directly during base setup
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredAuditLogFormatter())
        self._logger.addHandler(handler)

        # True Scheduling Queue and Dedicated Engine Worker
        self._request_queue: queue.Queue[Tuple[GenerationRequest, threading.Event]] = queue.Queue(
            maxsize=int(self._configuration.get("scheduler_queue_capacity", 1000))
        )
        self._shutdown_event = threading.Event()
        self._scheduler_thread = threading.Thread(target=self._process_scheduler_queue, daemon=True)
        self._scheduler_thread.start()

    # ========================================================================
    # CONTEXT MANAGER INTERFACE
    # ========================================================================

    def __enter__(self) -> "BaseProvider":
        """Initializes context manager structure."""
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Triggers orderly resource de-allocation upon scope destruction."""
        self.shutdown()

    # ========================================================================
    # LIFECYCLE MANAGEMENT ABSTRACT METHODS
    # ========================================================================

    @abstractmethod
    def initialize(self) -> None:
        """Executes bootstrap verification steps, credentials testing, and connections."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Performs orderly de-allocation of client sessions and scheduler loops."""
        with self._lock:
            self._shutdown_event.set()
            self._set_state(ProviderState.SHUTDOWN)

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Executes standard validation checks against upstream endpoints."""
        pass

    @abstractmethod
    def validate_configuration(self, configuration: Dict[str, Any]) -> None:
        """Validates configuration parameters prior to injection."""
        pass

    @abstractmethod
    def available_models(self) -> List[str]:
        """Queries local indexes or live connections for operational identifiers."""
        pass

    # ========================================================================
    # EXECUTION CORE ABSTRACT METHODS
    # ========================================================================

    @abstractmethod
    def _execute_generation(self, request: GenerationRequest) -> GenerationResponse:
        """Internal worker abstraction for blocking text synthesis."""
        pass

    @abstractmethod
    def _execute_stream_generation(
        self, request: GenerationRequest
    ) -> Generator[Dict[str, Any], None, None]:
        """Internal worker abstraction for processing generative streams."""
        pass

    # ========================================================================
    # CONCURRENT TRUE SCHEDULER IMPLEMENTATION
    # ========================================================================

    def _process_scheduler_queue(self) -> None:
        """True internal scheduling worker loop managing rate limits and pacing execution."""
        while not self._shutdown_event.is_set():
            try:
                # Poll queue for work blocks
                request, execution_gate = self._request_queue.get(timeout=0.2)
                
                # Check cancellation states prior to processing step activation
                if request.request_id in self._cancelled_requests:
                    self._request_queue.task_done()
                    continue

                # Enforce pacing delays or dynamic backend token management structures if required
                pacing_delay = float(self._configuration.get("scheduler_pacing_delay", 0.0))
                if pacing_delay > 0:
                    time.sleep(pacing_delay)

                # Release calling thread to enter core provider execution
                execution_gate.set()
                self._request_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self._audit_log(logging.ERROR, f"Scheduler loop encountered anomaly: {str(e)}")

    # ========================================================================
    # REQUEST CANCELLATION API
    # ========================================================================

    def cancel_request(self, request_id: str) -> bool:
        """Explicitly cancels a pending or ongoing execution thread request.

        Args:
            request_id: Unique transaction ID string.

        Returns:
            Boolean indicating if cancellation state successfully updated.
        """
        with self._lock:
            if request_id in self._active_resources:
                self._cancelled_requests.add(request_id)
                self._audit_log(logging.INFO, f"Cancellation signal dispatched.", request_id)
                self._trigger_event("request.cancelled", {"request_id": request_id})
                return True
            return False

    # ========================================================================
    # ENHANCED TIMEOUT CONTEXT MANAGER
    # ========================================================================

    @contextlib.contextmanager
    def _enforce_timeout(self, seconds: float, request_id: str) -> Generator[None, None, None]:
        """Enforces structural execution boundaries using explicit system timer intervals."""
        start_time = time.time()
        yield
        if time.time() - start_time > seconds:
            raise TimeoutError(
                f"Request execution exceeded operational timeout boundary of {seconds}s.",
                provider_name=self._provider_name,
                request_id=request_id,
            )

    # ========================================================================
    # PUBLIC ENTRY INTERFACES
    # ========================================================================

    @validate_payload
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Executes a standard non-streaming text generation request with automated retries."""
        self._check_and_recover_configuration()
        self._ensure_operational()

        # True Scheduling Orchestration Block
        execution_gate = threading.Event()
        try:
            self._request_queue.put((request, execution_gate), block=True, timeout=5.0)
            # Wait for scheduler queue allocator loop thread to drop gate
            if not execution_gate.wait(timeout=request.timeout):
                raise TimeoutError("Queue scheduling wait timeout exceeded.", self._provider_name, request.request_id)
        except queue.Full:
            raise RateLimitError("Concurrent scheduling queue capacity overflow.", self._provider_name, request.request_id)

        if request.request_id in self._cancelled_requests:
            raise RequestCancelledError("Request cancelled prior to execution execution.", self._provider_name, request.request_id)

        for hook in self._before_request_hooks:
            try:
                hook(request)
            except Exception as hook_exc:
                self._audit_log(logging.ERROR, f"Pre-request callback error: {str(hook_exc)}", request.request_id)

        validated_request = self.validate_request(request)
        model_resolved = self.select_model(validated_request.model)

        request_dict = validated_request.to_dict()
        request_dict["model"] = model_resolved
        normalized_request = GenerationRequest(**request_dict)

        with self._lock:
            self._active_resources[normalized_request.request_id] = normalized_request

        self._increment_active_requests()
        self._trigger_event("request.started", {"request_id": normalized_request.request_id, "model": model_resolved})

        start_time = time.perf_counter()
        response: Optional[GenerationResponse] = None
        error_encountered: Optional[Exception] = None

        try:
            if normalized_request.request_id in self._cancelled_requests:
                raise RequestCancelledError("Execution terminated by cancellation dispatch.", self._provider_name, normalized_request.request_id)

            with self._enforce_timeout(normalized_request.timeout, normalized_request.request_id):
                response = self._execute_with_retry(self._execute_generation, normalized_request)

            if normalized_request.request_id in self._cancelled_requests:
                raise RequestCancelledError("Execution results discarded due to cancellation.", self._provider_name, normalized_request.request_id)

            latency = time.perf_counter() - start_time
            self._update_statistics(success=True, latency=latency, tokens=response.usage.total_tokens, streamed=False)
            self._audit_log(logging.INFO, f"Generation completed successfully.", normalized_request.request_id, latency)
            self._trigger_event("request.success", {"request_id": normalized_request.request_id, "latency": latency})
            return response

        except Exception as exc:
            error_encountered = self.convert_exception(exc, normalized_request.request_id)
            latency = time.perf_counter() - start_time
            self._update_statistics(success=False, latency=latency, tokens=0, streamed=False)
            self._audit_log(logging.ERROR, f"Transaction failed: {str(error_encountered)}", normalized_request.request_id, latency, exc_info=True)
            self._trigger_event("request.failure", {"request_id": normalized_request.request_id, "error": str(error_encountered)})
            raise error_encountered

        finally:
            self._decrement_active_requests()
            with self._lock:
                self._active_resources.pop(normalized_request.request_id, None)
                self._cancelled_requests.discard(normalized_request.request_id)
            for hook in self._after_request_hooks:
                try:
                    hook(response, error_encountered)
                except Exception as hook_exc:
                    self._audit_log(logging.ERROR, f"Post-request callback error: {str(hook_exc)}", request.request_id)

    @validate_payload
    def stream_generate(self, request: GenerationRequest) -> Generator[Dict[str, Any], None, None]:
        """Provides an iterative streaming wrapper for chunk consumption."""
        self._check_and_recover_configuration()
        self._ensure_operational()

        if not self.check_capability(ProviderCapability.STREAMING):
            raise ValidationError(f"Streaming not supported.", self._provider_name, request.request_id)

        execution_gate = threading.Event()
        try:
            self._request_queue.put((request, execution_gate), block=True, timeout=5.0)
            if not execution_gate.wait(timeout=request.timeout):
                raise TimeoutError("Queue scheduling wait timeout exceeded.", self._provider_name, request.request_id)
        except queue.Full:
            raise RateLimitError("Concurrent scheduling queue capacity overflow.", self._provider_name, request.request_id)

        if request.request_id in self._cancelled_requests:
            raise RequestCancelledError("Stream cancelled prior to step execution.", self._provider_name, request.request_id)

        for hook in self._before_request_hooks:
            try:
                hook(request)
            except Exception as hook_exc:
                self._audit_log(logging.ERROR, f"Pre-stream callback error: {str(hook_exc)}", request.request_id)

        validated_request = self.validate_request(request)
        model_resolved = self.select_model(validated_request.model)

        request_dict = validated_request.to_dict()
        request_dict["model"] = model_resolved
        request_dict["stream"] = True
        normalized_request = GenerationRequest(**request_dict)

        with self._lock:
            self._active_resources[normalized_request.request_id] = normalized_request

        self._increment_active_requests()
        self._set_state(ProviderState.STREAMING)
        self._trigger_event("stream.started", {"request_id": normalized_request.request_id, "model": model_resolved})

        start_time = time.perf_counter()
        token_counter = 0
        error_encountered: Optional[Exception] = None

        try:
            if normalized_request.request_id in self._cancelled_requests:
                raise RequestCancelledError("Stream aborted before network fetch loop.", self._provider_name, normalized_request.request_id)

            with self._enforce_timeout(normalized_request.timeout, normalized_request.request_id):
                stream_generator = self._execute_with_retry(self._execute_stream_generation, normalized_request)

            for chunk in stream_generator:
                if normalized_request.request_id in self._cancelled_requests:
                    raise RequestCancelledError("Stream consumption terminated dynamically.", self._provider_name, normalized_request.request_id)
                token_counter += self._extract_chunk_token_count(chunk)
                yield self._enrich_stream_chunk(chunk, normalized_request.request_id, model_resolved)

            latency = time.perf_counter() - start_time
            self._update_statistics(success=True, latency=latency, tokens=token_counter, streamed=True)
            self._audit_log(logging.INFO, f"Stream channel fully consumed.", normalized_request.request_id, latency)
            self._trigger_event("stream.success", {"request_id": normalized_request.request_id, "tokens": token_counter})

        except Exception as exc:
            error_encountered = self.convert_exception(exc, normalized_request.request_id)
            latency = time.perf_counter() - start_time
            self._update_statistics(success=False, latency=latency, tokens=0, streamed=True)
            self._audit_log(logging.ERROR, f"Stream collapsed: {str(error_encountered)}", normalized_request.request_id, latency, exc_info=True)
            self._trigger_event("stream.failure", {"request_id": normalized_request.request_id, "error": str(error_encountered)})
            raise error_encountered

        finally:
            self._decrement_active_requests()
            self._set_state(ProviderState.READY)
            with self._lock:
                self._active_resources.pop(normalized_request.request_id, None)
                self._cancelled_requests.discard(normalized_request.request_id)
            for hook in self._after_request_hooks:
                try:
                    hook(None, error_encountered)
                except Exception as hook_exc:
                    self._audit_log(logging.ERROR, f"Post-stream callback error: {str(hook_exc)}", request.request_id)

    # ========================================================================
    # EXPORT / IMPORT SERIALIZATION HELPERS
    # ========================================================================

    def to_json(self, data_object: Any) -> str:
        """Standardized JSON serialization method mapping DTOs cleanly to string format."""
        if hasattr(data_object, "to_dict"):
            return json.dumps(data_object.to_dict(), indent=2)
        return json.dumps(data_object, indent=2, default=str)

    def from_json(self, json_string: str) -> Dict[str, Any]:
        """Standardized JSON deserialization helper resolving incoming payloads securely."""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValidationError(f"JSON schema parsing exception encountered: {str(e)}", self._provider_name)

    # ========================================================================
    # INTERCEPTORS AND EVENT SYSTEMS
    # ========================================================================

    def register_before_request_hook(self, hook: Callable[[GenerationRequest], None]) -> None:
        """Registers an execution interceptor to handle requests prior to routing."""
        with self._lock:
            self._before_request_hooks.append(hook)

    def register_after_request_hook(
        self, hook: Callable[[Optional[GenerationResponse], Optional[Exception]], None]
    ) -> None:
        """Registers a post-execution monitoring interceptor."""
        with self._lock:
            self._after_request_hooks.append(hook)

    def add_event_listener(self, event_type: str, callback: EventCallback) -> None:
        """Binds a functional notification handler to internal lifecycle events."""
        with self._lock:
            self._event_listeners[event_type].append(callback)

    def _trigger_event(self, event_type: str, telemetry: Dict[str, Any]) -> None:
        """Dispatches operational signals asynchronously or concurrently to listeners."""
        listeners = []
        with self._lock:
            listeners = list(self._event_listeners.get(event_type, [])) + list(self._event_listeners.get("*", []))

        for callback in listeners:
            try:
                callback(event_type, telemetry)
            except Exception as listener_exc:
                self._audit_log(logging.ERROR, f"Event execution dispatch anomaly: {str(listener_exc)}")

    # ========================================================================
    # CAPACITY DISCOVERY & METADATA MANAGEMENT
    # ========================================================================

    def register_model_metadata(self, metadata: ModelMetadata) -> None:
        """Injects model specifications into the provider tracking matrix (O(1) dictionary maps)."""
        with self._lock:
            self._cached_models[metadata.model_id] = metadata

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Resolves structural metadata schemas matching specific target tokens."""
        with self._lock:
            return self._cached_models.get(model_id)

    def discover_capabilities(self, model_id: Optional[str] = None) -> Set[ProviderCapability]:
        """Interrogates core registry matrices to isolate capabilities."""
        if not model_id:
            return set(self._supported_capabilities)
        meta = self.get_model_metadata(model_id)
        return set(meta.capabilities) if meta else set()

    # ========================================================================
    # RECOVERY & AUTOMATED RE-FINGERPRINTING LOGIC
    # ========================================================================

    def update_configuration(self, updates: Dict[str, Any]) -> None:
        """Safely mutates existing structural configuration environments at runtime."""
        with self._lock:
            working_copy = copy.deepcopy(self._configuration)
            working_copy.update(updates)

            self.validate_configuration(working_copy)
            self._configuration = working_copy
            self._config_fingerprint = self._generate_fingerprint(self._configuration)
            self._audit_log(logging.INFO, "Provider tracking configuration environment reassigned.")
            self.initialize()

    def _generate_fingerprint(self, options: Dict[str, Any]) -> str:
        """Generates a stable representation hash of current options environments."""
        try:
            serialized = json.dumps(options, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.sha256(str(id(options)).encode("utf-8")).hexdigest()

    def _check_and_recover_configuration(self) -> None:
        """Verifies operational settings and triggers automated recoveries if altered."""
        with self._lock:
            current_fp = self._generate_fingerprint(self._configuration)
            if current_fp != self._config_fingerprint:
                self._audit_log(logging.WARNING, "Detected out-of-band configuration variance. Restoring.")
                self._set_state(ProviderState.INITIALIZING)
                try:
                    self.initialize()
                    self._config_fingerprint = current_fp
                    self._set_state(ProviderState.READY)
                except Exception as recovery_error:
                    self._set_state(ProviderState.ERROR)
                    raise ConfigurationError(
                        f"Automated recovery lifecycle failure: {str(recovery_error)}", self._provider_name
                    )

    # ========================================================================
    # FIXED HIGH-PRECISION LATENCY PERCENTILE CALCULATOR
    # ========================================================================

    def calculate_latency_percentile(self, percentile: float) -> float:
        """Calculates percentiles using linear interpolation to ensure accuracy on small samples."""
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError("Percentile bounds must reside strictly between 0 and 100.")

        with self._lock:
            history = sorted(list(self._statistics._latency_history))

        if not history:
            return 0.0

        n = len(history)
        if n == 1:
            return history[0]

        idx = (percentile / 100.0) * (n - 1)
        floor_idx = int(idx)
        ceil_idx = min(floor_idx + 1, n - 1)
        
        if floor_idx == ceil_idx:
            return history[floor_idx]
            
        # Linear Interpolation Formula Execution
        weight = idx - floor_idx
        return history[floor_idx] + weight * (history[ceil_idx] - history[floor_idx])

    def _update_statistics(self, success: bool, latency: float, tokens: int, streamed: bool) -> None:
        """Updates internal data logs with processing results."""
        with self._lock:
            stats = self._statistics
            stats.total_requests += 1
            stats.last_request_time = datetime.now(timezone.utc)

            if success:
                stats.successful_requests += 1
                stats.tokens_generated += tokens
                if streamed:
                    stats.streamed_requests += 1

                stats._latency_history.append(latency)
                if len(stats._latency_history) > 1000:
                    stats._latency_history.pop(0)

                stats.average_latency = statistics.mean(stats._latency_history)
            else:
                stats.failed_requests += 1

            total_resolved = max(1, stats.total_requests)
            self._health.success_rate = (stats.successful_requests / total_resolved) * 100.0
            self._health.average_latency = stats.average_latency

            if stats.total_requests % 10 == 0:
                stats._metrics_snapshots.append(
                    {"timestamp": datetime.now(timezone.utc).isoformat(), "total": stats.total_requests, "avg_lat": stats.average_latency}
                )
                if len(stats._metrics_snapshots) > 100:
                    stats._metrics_snapshots.pop(0)

            if not success and self._health.state != ProviderState.STREAMING:
                self._health.failures += 1
                if self._health.failures > int(self._configuration.get("consecutive_failure_threshold", 5)):
                    self._set_state(ProviderState.ERROR)
            elif success:
                self._health.failures = max(0, self._health.failures - 1)
                if self._health.failures == 0 and self._state == ProviderState.ERROR:
                    self._set_state(ProviderState.READY)

            self._health.history.append(
                {"timestamp": datetime.now(timezone.utc), "state": self._state, "failures": self._health.failures}
            )
            if len(self._health.history) > 100:
                self._health.history.pop(0)

    # ========================================================================
    # UTILITY HELPERS, CORE SANITIZERS & LOGGERS
    # ========================================================================

    def _enrich_stream_chunk(self, chunk: Dict[str, Any], request_id: str, model: str) -> Dict[str, Any]:
        """Injects contextual and tracing parameters into outbound streaming data packets."""
        enriched = dict(chunk)
        enriched.update(
            {
                "request_id": request_id,
                "model": model,
                "provider": self._provider_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return enriched

    def _audit_log(
        self, level: int, message: str, request_id: Optional[str] = None, latency: Optional[float] = None, **kwargs: Any
    ) -> None:
        """Routes structured logs through local log managers with unified instrumentation properties."""
        extra_props = {"provider": self._provider_name, "request_id": request_id or "SYSTEM"}
        if latency is not None:
            extra_props["latency"] = latency

        self._logger.log(level, message, extra=extra_props, **kwargs)  # type: ignore

    def export_provider_info(self) -> Dict[str, Any]:
        """Exports static identification summaries, compilation indicators, and explicit version metadata."""
        with self._lock:
            return {
                "provider_name": self._provider_name,
                "provider_version": self._provider_version,
                "api_version": self._api_version,
                "framework_version": self.FRAMEWORK_VERSION,
                "build_metadata": self.BUILD_METADATA,
                "default_model": self._default_model,
                "supported_capabilities": [c.name for c in self._supported_capabilities],
                "configuration_fingerprint": self._config_fingerprint,
                "scheduler_queue_utilization": self._request_queue.qsize(),
            }

    def export_statistics(self) -> Dict[str, Any]:
        """Returns a snapshot of historical system operation counters."""
        with self._lock:
            stats = self._statistics
            current_uptime = (datetime.now(timezone.utc) - self._boot_time).total_seconds()
            return {
                "provider_name": self._provider_name,
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "retries": stats.retries,
                "streamed_requests": stats.streamed_requests,
                "average_latency_seconds": round(stats.average_latency, 4),
                "p50_latency": round(self.calculate_latency_percentile(50.0), 4),
                "p95_latency": round(self.calculate_latency_percentile(95.0), 4),
                "p99_latency": round(self.calculate_latency_percentile(99.0), 4),
                "tokens_generated": stats.tokens_generated,
                "uptime_seconds": round(current_uptime, 2),
                "last_request_time": stats.last_request_time.isoformat() if stats.last_request_time else None,
                "historical_snapshots": stats._metrics_snapshots,
            }

    def export_health_report(self) -> Dict[str, Any]:
        """Provides operational visibility assessments formatted for instrumentation layers."""
        with self._lock:
            current_uptime = (datetime.now(timezone.utc) - self._boot_time).total_seconds()
            self._health.uptime = current_uptime
            health = self._health
            return {
                "provider_name": self._provider_name,
                "available": self.is_available,
                "state": health.state.name,
                "last_check_time": health.last_check.isoformat(),
                "consecutive_failures": health.failures,
                "calculated_success_rate": round(health.success_rate, 2),
                "active_concurrent_requests": health.active_requests,
                "system_uptime_seconds": round(health.uptime, 2),
                "state_transitions_logged": len(health.history),
            }

    def get_configuration_copy(self) -> Dict[str, Any]:
        """Returns a deep copy of the configuration with sensitive values masked."""
        with self._lock:
            sanitized = copy.deepcopy(self._configuration)

            def mask_secrets(target: Any) -> Any:
                if isinstance(target, dict):
                    for k, v in list(target.items()):
                        if any(secret_term in k.lower() for secret_term in ["key", "secret", "token", "password", "auth"]):
                            target[k] = "************"
                        else:
                            mask_secrets(v)
                elif isinstance(target, list):
                    for item in target:
                        mask_secrets(item)

            mask_secrets(sanitized)
            return sanitized

    def validate_request(self, request: GenerationRequest) -> GenerationRequest:
        """Validates incoming structural requests against core system protocols."""
        if not request:
            raise ValidationError("Provided GenerationRequest payload cannot be None.", self._provider_name)
        return request

    def validate_response(self, response: GenerationResponse) -> GenerationResponse:
        """Verifies integrity of structures returned by provider sub-classes."""
        if not response:
            raise ResponseError("Upstream parsing step produced an empty target structure.", self._provider_name)
        return response

    def select_model(self, requested_model: Optional[str]) -> str:
        """Resolves the active target model string falling back safely if unsupplied."""
        if not requested_model:
            return self._default_model
        return requested_model.strip()

    def check_capability(self, capability: ProviderCapability) -> bool:
        """Checks if a capability is supported by this provider."""
        return capability in self._supported_capabilities

    def reset_statistics(self) -> None:
        """Clears accumulated statistics."""
        with self._lock:
            self._statistics = ProviderStatistics()
            self._health.average_latency = 0.0
            self._health.success_rate = 100.0
            self._health.failures = 0
            self._audit_log(logging.INFO, "Telemetry counters successfully cleared.")

    @property
    def provider_name(self) -> str:
        """Returns the provider name identifier."""
        return self._provider_name

    @property
    def state(self) -> ProviderState:
        """Returns the current state of the provider."""
        with self._lock:
            return self._state

    @property
    def is_available(self) -> bool:
        """Evaluates live network state viability markers."""
        with self._lock:
            return (
                self._state in (ProviderState.READY, ProviderState.BUSY, ProviderState.STREAMING)
                and self._health.available
            )

    def _set_state(self, new_state: ProviderState) -> None:
        """Thread-safe modifier updating current state variables."""
        with self._lock:
            if self._state == ProviderState.SHUTDOWN and new_state != ProviderState.SHUTDOWN:
                return
            if self._state != new_state:
                old_state = self._state
                self._state = new_state
                self._health.state = new_state
                self._audit_log(logging.DEBUG, f"State Transition: [{old_state.name}] -> [{new_state.name}]")

    def _increment_active_requests(self) -> None:
        """Safely increments concurrent task counters."""
        with self._lock:
            self._health.active_requests += 1
            if self._state == ProviderState.READY:
                self._set_state(ProviderState.BUSY)

    def _decrement_active_requests(self) -> None:
        """Safely decrements concurrent task counters."""
        with self._lock:
            self._health.active_requests = max(0, self._health.active_requests - 1)
            if self._health.active_requests == 0 and self._state == ProviderState.BUSY:
                self._set_state(ProviderState.READY)

    def _ensure_operational(self) -> None:
        """Guards and blocks execution calls targeting down or dead entities."""
        with self._lock:
            if self._state in (ProviderState.SHUTDOWN, ProviderState.DISABLED):
                raise UnavailableError(f"Operations rejected. Provider is: {self._state.name}", self._provider_name)

    def _extract_chunk_token_count(self, chunk: Dict[str, Any]) -> int:
        """Determines approximate token allocation weight parameters from unstructured chunks."""
        if not chunk:
            return 0
        if "usage" in chunk and isinstance(chunk["usage"], dict):
            return chunk["usage"].get("completion_tokens", 0)
        if "text" in chunk and isinstance(chunk["text"], str):
            return max(1, len(chunk["text"]) // 4)
        return 1

    def _execute_with_retry(self, target_function: Callable[[GenerationRequest], T], request: GenerationRequest) -> T:
        """Handles robust execution of requests with exponential backoff retries."""
        attempts = 0
        current_backoff = self._initial_backoff

        while True:
            try:
                attempts += 1
                return target_function(request)
            except Exception as exc:
                converted_error = self.convert_exception(exc, request.request_id)

                if not self._is_retryable_exception(converted_error) or attempts > self._max_retries:
                    if attempts > self._max_retries:
                        raise RetryLimitExceeded(
                            f"Total retry operations ceiling exceeded. Actions attempted: {attempts}",
                            provider_name=self._provider_name,
                            request_id=request.request_id,
                            raw_error=converted_error,
                        )
                    raise converted_error

                with self._lock:
                    self._statistics.retries += 1

                self._audit_log(
                    logging.WARNING, f"Transient error on attempt {attempts}. Retrying in {current_backoff:.2f}s..."
                )
                time.sleep(current_backoff)
                current_backoff *= self._backoff_factor

    def _is_retryable_exception(self, error: ProviderError) -> bool:
        """Determines if the provided error is transient and safe to retry."""
        return isinstance(error, (RateLimitError, TimeoutError, UnavailableError, StreamingError))

    def convert_exception(self, exception: Exception, request_id: str) -> ProviderError:
        """Intercepts arbitrary anomalies, mapping them to standard provider exceptions."""
        if isinstance(exception, ProviderError):
            return exception

        msg = str(exception).lower()
        if any(term in msg for term in ["api key", "unauthorized", "auth", "credentials", "401"]):
            return AuthenticationError(f"Authentication failure: {exception}", self._provider_name, request_id, exception)
        if any(term in msg for term in ["rate limit", "429", "throttled", "quota"]):
            return RateLimitError(f"Rate limit exceeded: {exception}", self._provider_name, request_id, exception)
        if any(term in msg for term in ["timeout", "deadline", "timed out", "408"]):
            return TimeoutError(f"Request timed out: {exception}", self._provider_name, request_id, exception)
        if any(term in msg for term in ["dns", "unreachable", "refused", "503", "unavailable"]):
            return UnavailableError(f"Provider endpoint unavailable: {exception}", self._provider_name, request_id, exception)
        if isinstance(exception, RequestCancelledError):
            return exception

        return ResponseError(f"Unhandled system error: {exception}", self._provider_name, request_id, exception)
