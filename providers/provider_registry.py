"""
Cabangile AI Studio - Enterprise Provider Registry
File Location: studio/providers/provider_registry.py
Architecture: Clean Architecture, SOLID, DDD, Thread-safe, Async-ready
Python Version: 3.11+
Dependencies: Standard Library Only
"""

import asyncio
import dataclasses
import datetime
import enum
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
logger = logging.getLogger("CabangileAIStudio.ProviderRegistry")


# ==========================================
# SYSTEM ENUMERATIONS
# ==========================================
class ProviderStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    MAINTENANCE = "MAINTENANCE"
    CIRCUIT_BROKEN = "CIRCUIT_BROKEN"
    UNKNOWN = "UNKNOWN"


class ProviderType(str, enum.Enum):
    LLM = "LLM"
    EMBEDDING = "EMBEDDING"
    VISION = "VISION"
    AUDIO = "AUDIO"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    RERANKER = "RERANKER"
    HYBRID = "HYBRID"
    CUSTOM = "CUSTOM"


class ProviderCapability(str, enum.Enum):
    TEXT_GENERATION = "TEXT_GENERATION"
    FUNCTION_CALLING = "FUNCTION_CALLING"
    STREAMING = "STREAMING"
    VISION_INPUT = "VISION_INPUT"
    JSON_MODE = "JSON_MODE"
    EMBEDDING_GENERATION = "EMBEDDING_GENERATION"
    SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
    TEXT_TO_SPEECH = "TEXT_TO_SPEECH"
    IMAGE_EDITING = "IMAGE_EDITING"


class AuthenticationType(str, enum.Enum):
    API_KEY = "API_KEY"
    BEARER_TOKEN = "BEARER_TOKEN"
    OAUTH2 = "OAUTH2"
    MUTUAL_TLS = "MUTUAL_TLS"
    AWS_IAM = "AWS_IAM"
    NONE = "NONE"


class ProviderPriority(int, enum.Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class ProviderAvailability(str, enum.Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    DRAINING = "DRAINING"


class RegistryOperation(str, enum.Enum):
    REGISTER = "REGISTER"
    UNREGISTER = "UNREGISTER"
    UPDATE = "UPDATE"
    VALIDATE = "VALIDATE"
    BACKUP = "BACKUP"
    RESTORE = "RESTORE"
    SNAPSHOT = "SNAPSHOT"
    DIAGNOSTICS = "DIAGNOSTICS"


class RegistryEvent(str, enum.Enum):
    ON_REGISTER = "ON_REGISTER"
    ON_UNREGISTER = "ON_UNREGISTER"
    ON_UPDATE = "ON_UPDATE"
    ON_STATUS_CHANGE = "ON_STATUS_CHANGE"
    ON_CIRCUIT_BREAK = "ON_CIRCUIT_BREAK"


# ==========================================
# CUSTOM EXCEPTIONS
# ==========================================
class ProviderRegistryError(Exception):
    """Enterprise-grade domain exception for all Registry operations."""

    def __init__(
        self,
        error_code: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"[{error_code}] {message}")
        self.error_code = error_code
        self.message = message
        self.metadata = metadata or {}
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ==========================================
# IMMUTABLE DOMAIN DATA MODELS
# ==========================================
@dataclass(frozen=True)
class ProviderConfiguration:
    provider_id: str
    provider_name: str
    provider_type: ProviderType
    base_url: str
    api_version: str
    authentication_type: AuthenticationType
    api_key_required: bool
    timeout_seconds: float
    retry_limit: int
    rate_limit: int  # Requests per minute
    supported_models: List[str] = field(default_factory=list)
    supported_capabilities: List[ProviderCapability] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: ProviderPriority = ProviderPriority.MEDIUM
    health_status: ProviderStatus = ProviderStatus.UNKNOWN
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    load_balancing_weight: int = 100
    circuit_breaker_threshold: int = 5
    dependencies: List[str] = field(default_factory=list)

    def copy_with(self, **kwargs: Any) -> "ProviderConfiguration":
        data = dataclasses.asdict(self)
        data.update(kwargs)
        return ProviderConfiguration(**data)


@dataclass(frozen=True)
class ProviderStatistics:
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_latency: float = 0.0
    uptime_percentage: float = 100.0
    health_score: float = 100.0
    consecutive_failures: int = 0
    last_request_timestamp: Optional[str] = None
    registration_timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    latency_history: List[float] = field(default_factory=list)
    cost_accumulated: float = 0.0


@dataclass(frozen=True)
class ProviderRegistrationResult:
    success: bool
    provider_id: str
    message: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RegistrySnapshot:
    snapshot_id: str
    timestamp: str
    providers: Dict[str, Dict[str, Any]]
    statistics: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class RegistryDiagnosticReport:
    timestamp: str
    total_registered_providers: int
    active_providers: int
    disabled_providers: int
    unhealthy_providers: int
    circuit_broken_providers: int
    duplicate_providers: List[str]
    validation_failures: Dict[str, List[str]]
    warning_count: int
    error_count: int
    cache_statistics: Dict[str, Any]
    health_summary: Dict[str, str]
    registry_statistics: Dict[str, Any]
    integrity_verified: bool
    checksum: str


@dataclass(frozen=True)
class RegistryBackup:
    backup_id: str
    timestamp: str
    version: str
    payload: str  # Non-cryptographic obfuscated state string
    checksum: str


@dataclass(frozen=True)
class AuditLogEntry:
    timestamp: str
    operation: str
    provider_id: Optional[str]
    actor: str
    status: str
    details: Dict[str, Any]


# ==========================================
# MAIN CORE REGISTRY COMPONENT
# ==========================================
class ProviderRegistry:
    """Thread-safe, high-performance, async-ready central provider registry."""

    MAX_HISTORY: int = 100

    def __init__(self) -> None:
        # Atomic lock for all write/mutation paths
        self._lock = asyncio.Lock()

        # Storage Engine (In-Memory Maps)
        self._providers: Dict[str, ProviderConfiguration] = {}
        self._statistics: Dict[str, ProviderStatistics] = {}
        self._snapshots: Dict[str, RegistrySnapshot] = {}
        self._backups: Dict[str, RegistryBackup] = {}
        self._version_histories: Dict[str, List[Dict[str, Any]]] = {}
        self._failover_groups: Dict[str, List[str]] = {}

        # Optimized Indexes for fast multi-dimensional queries
        self._index_by_type: Dict[ProviderType, Set[str]] = {t: set() for t in ProviderType}
        self._index_by_capability: Dict[ProviderCapability, Set[str]] = {
            c: set() for c in ProviderCapability
        }
        self._index_by_status: Dict[ProviderStatus, Set[str]] = {s: set() for s in ProviderStatus}
        self._index_by_alias: Dict[str, str] = {}
        self._index_by_tag: Dict[str, Set[str]] = {}

        # Multi-tier Internal Caches
        self._provider_cache: Dict[str, ProviderConfiguration] = {}
        self._validation_cache: Dict[str, Tuple[bool, List[str]]] = {}
        self._metrics_cache: Dict[str, Any] = {}
        self._health_cache: Dict[str, ProviderStatus] = {}

        # Cache Metrics Tracking
        self._cache_hits = 0
        self._cache_misses = 0

        # System Logging, Audit Trail, and Hooks Engine
        self._warning_log: List[Tuple[str, str]] = []
        self._error_log: List[Tuple[str, str]] = []
        self._audit_trail: List[AuditLogEntry] = []
        self._event_listeners: Dict[RegistryEvent, List[Callable[[Dict[str, Any]], Any]]] = {
            ev: [] for ev in RegistryEvent
        }

        self._is_shutdown = False

    def _ensure_active(self) -> None:
        if self._is_shutdown:
            raise ProviderRegistryError(
                "REGISTRY_SHUTDOWN", "The provider registry is shut down and rejecting requests."
            )

    def _record_warning(self, msg: str) -> None:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._warning_log.append((ts, msg))
        logger.warning(msg)

    def _record_error(self, msg: str) -> None:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._error_log.append((ts, msg))
        logger.error(msg)

    def _log_audit(self, operation: str, provider_id: Optional[str], actor: str, status: str, details: Dict[str, Any]) -> None:
        entry = AuditLogEntry(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            operation=operation,
            provider_id=provider_id,
            actor=actor,
            status=status,
            details=details
        )
        self._audit_trail.append(entry)

    def _trigger_event(self, event: RegistryEvent, data: Dict[str, Any]) -> None:
        for cb in self._event_listeners.get(event, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(data))
                else:
                    cb(data)
            except Exception as e:
                self._record_error(f"Error executing hook for event {event.value}: {str(e)}")

    def _clear_cache(self) -> None:
        self._provider_cache.clear()
        self._validation_cache.clear()
        self._metrics_cache.clear()
        self._health_cache.clear()

    def _rebuild_indexes(self) -> None:
        for t in self._index_by_type:
            self._index_by_type[t].clear()
        for c in self._index_by_capability:
            self._index_by_capability[c].clear()
        for s in self._index_by_status:
            self._index_by_status[s].clear()
        self._index_by_alias.clear()
        self._index_by_tag.clear()

        for pid, p in self._providers.items():
            self._index_by_type[p.provider_type].add(pid)
            self._index_by_status[p.health_status].add(pid)
            for cap in p.supported_capabilities:
                self._index_by_capability[cap].add(pid)
            for alias in p.aliases:
                self._index_by_alias[alias.lower()] = pid
            for tag in p.tags:
                normalized_tag = tag.lower()
                if normalized_tag not in self._index_by_tag:
                    self._index_by_tag[normalized_tag] = set()
                self._index_by_tag[normalized_tag].add(pid)

    def _validate_provider_sync(self, provider: ProviderConfiguration) -> Tuple[bool, List[str]]:
        errors = []
        if not provider.provider_id or not provider.provider_id.strip():
            errors.append("provider_id cannot be blank.")
        elif not re.match(r"^[a-zA-Z0-9_\-]+$", provider.provider_id):
            errors.append(
                "provider_id contains invalid characters. Only alphanumeric, underscores, and hyphens permitted."
            )

        if not provider.provider_name or not provider.provider_name.strip():
            errors.append("provider_name cannot be empty.")

        url_pattern = re.compile(r"^(https?://)[^\s/$.?#].[^\s]*$", re.IGNORECASE)
        if not url_pattern.match(provider.base_url):
            errors.append(f"Invalid base_url format: {provider.base_url}")

        if provider.timeout_seconds <= 0:
            errors.append("timeout_seconds must be strictly positive.")
        if provider.retry_limit < 0:
            errors.append("retry_limit cannot be negative.")
        if provider.rate_limit < 0:
            errors.append("rate_limit cannot be negative.")
        if provider.load_balancing_weight < 0:
            errors.append("load_balancing_weight cannot be negative.")

        if not provider.supported_models:
            errors.append("At least one supported model must be declared.")

        if (
            provider.authentication_type != AuthenticationType.NONE
            and provider.api_key_required
            and not provider.configuration.get("api_key")
        ):
            errors.append("API key validation failed: configuration missing required api_key item.")

        if provider.provider_id in provider.dependencies:
            errors.append("Provider cannot depend on itself.")

        return len(errors) == 0, errors

    def _calculate_checksum(self) -> str:
        state_string = ""
        for pid in sorted(self._providers.keys()):
            p = self._providers[pid]
            state_string += f"{pid}:{p.health_status.value}:{p.enabled};"
        return hashlib.sha256(state_string.encode('utf-8')).hexdigest()

    def _serialize_provider(self, provider: ProviderConfiguration) -> Dict[str, Any]:
        data = dataclasses.asdict(provider)
        data["provider_type"] = provider.provider_type.value
        data["authentication_type"] = provider.authentication_type.value
        data["priority"] = provider.priority.value
        data["health_status"] = provider.health_status.value
        data["supported_capabilities"] = [c.value for c in provider.supported_capabilities]
        return data

    def _deserialize_provider(self, data: Dict[str, Any]) -> ProviderConfiguration:
        d = dict(data)
        d["provider_type"] = ProviderType(d["provider_type"])
        d["authentication_type"] = AuthenticationType(d["authentication_type"])
        d["priority"] = ProviderPriority(d["priority"])
        d["health_status"] = ProviderStatus(d["health_status"])
        d["supported_capabilities"] = [ProviderCapability(c) for c in d["supported_capabilities"]]
        return ProviderConfiguration(**d)

    def _obfuscate_secret(self, data_str: str) -> str:
        """Internal obfuscation only. Not cryptographic protection."""
        mask = 0x5A
        return "".join(chr(ord(c) ^ mask) for c in data_str)

    def _generate_backup_id(self) -> str:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"BACKUP-{ts}-{len(self._backups) + 1}"

    def _generate_snapshot_id(self) -> str:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"SNAPSHOT-{ts}-{len(self._snapshots) + 1}"

    # ==========================================
    # PUBLIC ASYNCHRONOUS API METHODS
    # ==========================================
    def register_listener(self, event: RegistryEvent, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Registers a reactive event consumer hook pattern inside system engine."""
        self._event_listeners[event].append(callback)

    def remove_listener(self, event: RegistryEvent, callback: Callable[[Dict[str, Any]], Any]) -> bool:
        """Removes a previously registered reactive event listener hook."""
        if callback in self._event_listeners.get(event, []):
            self._event_listeners[event].remove(callback)
            return True
        return False

    async def register_provider(
        self, provider: ProviderConfiguration
    ) -> ProviderRegistrationResult:
        """Asynchronously registers a new provider inside the system."""
        self._ensure_active()
        async with self._lock:
            if provider.provider_id in self._providers:
                msg = f"Duplicate provider collision: {provider.provider_id} already registered."
                self._record_error(msg)
                self._log_audit("REGISTER", provider.provider_id, "SYSTEM", "FAILED", {"reason": "Collision"})
                return ProviderRegistrationResult(
                    success=False, provider_id=provider.provider_id, message=msg, errors=[msg]
                )

            # Duplicate Alias Detection Checking
            for alias in provider.aliases:
                norm_alias = alias.lower()
                if norm_alias in self._index_by_alias and self._index_by_alias[norm_alias] != provider.provider_id:
                    msg = f"Alias collision: '{alias}' is already claimed by provider {self._index_by_alias[norm_alias]}."
                    self._record_error(msg)
                    self._log_audit("REGISTER", provider.provider_id, "SYSTEM", "FAILED", {"reason": "Alias Collision"})
                    return ProviderRegistrationResult(
                        success=False, provider_id=provider.provider_id, message=msg, errors=[msg]
                    )

            is_valid, errors = self._validate_provider_sync(provider)
            self._validation_cache[provider.provider_id] = (is_valid, errors)

            if not is_valid:
                msg = f"Validation failed during registration of provider: {provider.provider_id}"
                self._record_error(msg)
                self._log_audit("REGISTER", provider.provider_id, "SYSTEM", "FAILED", {"errors": errors})
                return ProviderRegistrationResult(
                    success=False, provider_id=provider.provider_id, message=msg, errors=errors
                )

            self._providers[provider.provider_id] = provider
            if provider.provider_id not in self._statistics:
                self._statistics[provider.provider_id] = ProviderStatistics()

            # Version History Record Tracking (Bounded Capacity)
            self._version_histories[provider.provider_id] = [{
                "version": 1,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "snapshot": self._serialize_provider(provider)
            }]

            self._rebuild_indexes()
            self._clear_cache()

            self._log_audit("REGISTER", provider.provider_id, "SYSTEM", "SUCCESS", {})
            self._trigger_event(RegistryEvent.ON_REGISTER, {"provider_id": provider.provider_id})

            logger.info(f"Successfully registered provider: {provider.provider_id}")
            return ProviderRegistrationResult(
                success=True,
                provider_id=provider.provider_id,
                message="Registration completed successfully.",
            )

    async def unregister_provider(self, provider_id: str) -> bool:
        """Removes a provider from the registry system cleanly."""
        self._ensure_active()
        async with self._lock:
            target_id = self._index_by_alias.get(provider_id.lower(), provider_id)
            if target_id not in self._providers:
                self._record_warning(
                    f"Attempted unregister operation on missing provider: {provider_id}"
                )
                return False

            del self._providers[target_id]
            if target_id in self._statistics:
                del self._statistics[target_id]
            if target_id in self._version_histories:
                del self._version_histories[target_id]

            self._rebuild_indexes()
            self._clear_cache()
            
            self._log_audit("UNREGISTER", target_id, "SYSTEM", "SUCCESS", {})
            self._trigger_event(RegistryEvent.ON_UNREGISTER, {"provider_id": target_id})
            logger.info(f"Provider {target_id} cleanly removed from registry.")
            return True

    async def update_provider(self, provider: ProviderConfiguration) -> ProviderRegistrationResult:
        """Updates the configuration details of an existing active provider."""
        self._ensure_active()
        async with self._lock:
            if provider.provider_id not in self._providers:
                msg = f"Update target provider matching id {provider.provider_id} not found."
                return ProviderRegistrationResult(
                    success=False, provider_id=provider.provider_id, message=msg, errors=[msg]
                )

            # Duplicate Alias Verification for Update Paths
            for alias in provider.aliases:
                norm_alias = alias.lower()
                if norm_alias in self._index_by_alias and self._index_by_alias[norm_alias] != provider.provider_id:
                    msg = f"Alias collision during update: '{alias}' is already claimed by provider {self._index_by_alias[norm_alias]}."
                    return ProviderRegistrationResult(
                        success=False, provider_id=provider.provider_id, message=msg, errors=[msg]
                    )

            is_valid, errors = self._validate_provider_sync(provider)
            self._validation_cache[provider.provider_id] = (is_valid, errors)

            if not is_valid:
                msg = f"Validation failed while updating provider: {provider.provider_id}"
                return ProviderRegistrationResult(
                    success=False, provider_id=provider.provider_id, message=msg, errors=errors
                )

            old_provider = self._providers[provider.provider_id]
            self._providers[provider.provider_id] = provider

            # Bounded Version History Updates Tracking
            history = self._version_histories.get(provider.provider_id, [])
            next_ver = history[-1]["version"] + 1 if history else 1
            history.append({
                "version": next_ver,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "snapshot": self._serialize_provider(provider)
            })
            if len(history) > self.MAX_HISTORY:
                history.pop(0)
            self._version_histories[provider.provider_id] = history

            self._rebuild_indexes()
            self._clear_cache()

            self._log_audit("UPDATE", provider.provider_id, "SYSTEM", "SUCCESS", {"version": next_ver})
            self._trigger_event(RegistryEvent.ON_UPDATE, {"provider_id": provider.provider_id})

            if old_provider.health_status != provider.health_status:
                self._trigger_event(RegistryEvent.ON_STATUS_CHANGE, {
                    "provider_id": provider.provider_id,
                    "old_status": old_provider.health_status.value,
                    "new_status": provider.health_status.value
                })

            return ProviderRegistrationResult(
                success=True, provider_id=provider.provider_id, message="Update applied completely."
            )

    async def get_provider(self, provider_id: str) -> Optional[ProviderConfiguration]:
        """Fetches the internal runtime object instance configuration mapping for target id/alias."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        
        if resolved_id in self._provider_cache:
            self._cache_hits += 1
            return self._provider_cache[resolved_id]

        self._cache_misses += 1
        async with self._lock:
            provider = self._providers.get(resolved_id)
            if provider:
                self._provider_cache[resolved_id] = provider
                return provider
            return None

    async def get_provider_configuration(self, provider_id: str) -> ProviderConfiguration:
        """Retrieves configuration or raises ProviderRegistryError if not found."""
        p = await self.get_provider(provider_id)
        if not p:
            raise ProviderRegistryError(
                "PROVIDER_NOT_FOUND", f"Target provider identity {provider_id} does not exist."
            )
        return p

    async def get_provider_by_alias(self, alias: str) -> Optional[ProviderConfiguration]:
        """Fetches a provider configuration via its registered alias directly."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(alias.lower())
        if resolved_id:
            return await self.get_provider(resolved_id)
        return None

    async def get_provider_by_tag(self, tag: str) -> List[ProviderConfiguration]:
        """Returns all provider configurations associated with the specified case-insensitive tag."""
        self._ensure_active()
        async with self._lock:
            pids = self._index_by_tag.get(tag.lower(), set())
            return [self._providers[pid] for pid in pids if pid in self._providers]

    async def get_provider_history(self, provider_id: str) -> List[Dict[str, Any]]:
        """Retrieves tracking version history entries for a target provider engine id."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        async with self._lock:
            if resolved_id not in self._providers:
                raise ProviderRegistryError("PROVIDER_NOT_FOUND", f"Target provider {provider_id} does not exist.")
            return list(self._version_histories.get(resolved_id, []))

    async def rollback_provider_version(self, provider_id: str, version: int) -> bool:
        """Rolls back an existing provider to a designated version sequence history state."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        async with self._lock:
            if resolved_id not in self._providers:
                return False
            
            history = self._version_histories.get(resolved_id, [])
            target_snapshot = None
            for entry in history:
                if entry["version"] == version:
                    target_snapshot = entry["snapshot"]
                    break
            
            if not target_snapshot:
                return False
                
            rolled_provider = self._deserialize_provider(target_snapshot)
            self._providers[resolved_id] = rolled_provider
            
            # Record the rollback itself as a new history update event element
            next_ver = history[-1]["version"] + 1 if history else 1
            history.append({
                "version": next_ver,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "snapshot": target_snapshot
            })
            if len(history) > self.MAX_HISTORY:
                history.pop(0)
            self._version_histories[resolved_id] = history
            
            self._rebuild_indexes()
            self._clear_cache()
            self._log_audit("ROLLBACK", resolved_id, "SYSTEM", "SUCCESS", {"target_version": version})
            return True

    async def get_provider_statistics(self, provider_id: str) -> ProviderStatistics:
        """Returns the collected statistics object for a given provider engine id."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        async with self._lock:
            if resolved_id not in self._providers:
                raise ProviderRegistryError(
                    "PROVIDER_NOT_FOUND", f"Target provider {provider_id} does not exist."
                )
            return self._statistics.get(resolved_id, ProviderStatistics())

    async def provider_exists(self, provider_id: str) -> bool:
        """Checks internal map presence for provider identifier or alias matching."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        return resolved_id in self._providers

    async def list_providers(self) -> List[ProviderConfiguration]:
        """Provides full view arrays of all registered providers in storage engine."""
        self._ensure_active()
        async with self._lock:
            return list(self._providers.values())

    async def list_active_providers(self) -> List[ProviderConfiguration]:
        """Lists providers that are enabled and not in an UNHEALTHY or CIRCUIT_BROKEN state."""
        self._ensure_active()
        async with self._lock:
            return [
                p for p in self._providers.values() 
                if p.enabled and p.health_status not in (ProviderStatus.UNHEALTHY, ProviderStatus.CIRCUIT_BROKEN)
            ]

    async def list_enabled_providers(self) -> List[ProviderConfiguration]:
        """Lists all providers that are enabled, regardless of operational health status."""
        self._ensure_active()
        async with self._lock:
            return [p for p in self._providers.values() if p.enabled]

    async def search_providers(self, query: str) -> List[ProviderConfiguration]:
        """Searches provider IDs, names, tags, aliases, and models using substring tracking matches."""
        self._ensure_active()
        async with self._lock:
            q = query.lower()
            results = []
            for p in self._providers.values():
                if (
                    q in p.provider_id.lower()
                    or q in p.provider_name.lower()
                    or any(q in m.lower() for m in p.supported_models)
                    or any(q in t.lower() for t in p.tags)
                    or any(q in al.lower() for al in p.aliases)
                ):
                    results.append(p)
            return results

    async def filter_by_capability(
        self, capability: ProviderCapability
    ) -> List[ProviderConfiguration]:
        """Leverages domain capability mapping matrix indexes for speed retrieval optimizations."""
        self._ensure_active()
        async with self._lock:
            pids = self._index_by_capability.get(capability, set())
            return [self._providers[pid] for pid in pids]

    async def filter_by_type(self, provider_type: ProviderType) -> List[ProviderConfiguration]:
        """Retrieves all providers of a specific type via index lookups."""
        self._ensure_active()
        async with self._lock:
            pids = self._index_by_type.get(provider_type, set())
            return [self._providers[pid] for pid in pids]

    async def filter_by_status(self, status: ProviderStatus) -> List[ProviderConfiguration]:
        """Retrieves all providers matching a given status via index lookups."""
        self._ensure_active()
        async with self._lock:
            pids = self._index_by_status.get(status, set())
            return [self._providers[pid] for pid in pids]

    async def enable_provider(self, provider_id: str) -> bool:
        """Switches runtime active flag status property true for target provider engine id."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        async with self._lock:
            if resolved_id not in self._providers:
                return False
            p = self._providers[resolved_id]
            if not p.enabled:
                self._providers[resolved_id] = p.copy_with(enabled=True)
                self._clear_cache()
                self._log_audit("ENABLE", resolved_id, "SYSTEM", "SUCCESS", {})
            return True

    async def disable_provider(self, provider_id: str) -> bool:
        """Switches runtime active flag status property false for target provider engine id."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        async with self._lock:
            if resolved_id not in self._providers:
                return False
            p = self._providers[resolved_id]
            if p.enabled:
                self._providers[resolved_id] = p.copy_with(enabled=False)
                self._clear_cache()
                self._log_audit("DISABLE", resolved_id, "SYSTEM", "SUCCESS", {})
            return True

    async def validate_provider(self, provider_id: str) -> Tuple[bool, List[str]]:
        """Performs deep code syntax logic verification validations on internal registry targets."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        async with self._lock:
            if resolved_id not in self._providers:
                return False, [f"Provider {provider_id} does not exist inside active engine map."]
            p = self._providers[resolved_id]
            res, errs = self._validate_provider_sync(p)
            self._validation_cache[resolved_id] = (res, errs)
            return res, errs

    async def refresh_provider(self, provider_id: str, external_status: ProviderStatus) -> None:
        """Bridges health checker metrics pipelines with individual runtime domain models."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        async with self._lock:
            if resolved_id not in self._providers:
                raise ProviderRegistryError("PROVIDER_NOT_FOUND", f"Provider {provider_id} unknown.")
            p = self._providers[resolved_id]
            old_status = p.health_status
            self._providers[resolved_id] = p.copy_with(health_status=external_status)
            self._rebuild_indexes()
            self._clear_cache()
            
            if old_status != external_status:
                self._trigger_event(RegistryEvent.ON_STATUS_CHANGE, {
                    "provider_id": resolved_id,
                    "old_status": old_status.value,
                    "new_status": external_status.value
                })

    async def refresh_all(self, status_mapping: Dict[str, ProviderStatus]) -> None:
        """Performs bulk transactional matrix health context status update parameters safely."""
        self._ensure_active()
        async with self._lock:
            for pid, status in status_mapping.items():
                resolved_id = self._index_by_alias.get(pid.lower(), pid)
                if resolved_id in self._providers:
                    p = self._providers[resolved_id]
                    if p.health_status != status:
                        self._providers[resolved_id] = p.copy_with(health_status=status)
            self._rebuild_indexes()
            self._clear_cache()

    async def report_metric(self, provider_id: str, success: bool, latency: float, cost: float = 0.0) -> None:
        """Updates transactional system operational counts, executing adaptive circuit checking metrics."""
        self._ensure_active()
        resolved_id = self._index_by_alias.get(provider_id.lower(), provider_id)
        async with self._lock:
            if resolved_id not in self._providers:
                return

            p = self._providers[resolved_id]
            current_stats = self._statistics.get(resolved_id, ProviderStatistics())
            
            rc = current_stats.request_count + 1
            sc = current_stats.success_count + (1 if success else 0)
            fc = current_stats.failure_count + (0 if success else 1)
            
            consec_fail = 0 if success else (current_stats.consecutive_failures + 1)
            
            history = list(current_stats.latency_history)
            history.append(latency)
            if len(history) > 50:
                history.pop(0)
            avg_l = sum(history) / len(history)

            uptime = (sc / rc) * 100.0
            score = max(0.0, 100.0 - (consec_fail * 15.0) - (fc / rc * 20.0))

            new_stats = ProviderStatistics(
                request_count=rc,
                success_count=sc,
                failure_count=fc,
                average_latency=avg_l,
                uptime_percentage=uptime,
                health_score=score,
                consecutive_failures=consec_fail,
                last_request_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                registration_timestamp=current_stats.registration_timestamp,
                latency_history=history,
                cost_accumulated=current_stats.cost_accumulated + cost
            )
            self._statistics[resolved_id] = new_stats

            if consec_fail >= p.circuit_breaker_threshold and p.health_status != ProviderStatus.CIRCUIT_BROKEN:
                self._providers[resolved_id] = p.copy_with(health_status=ProviderStatus.CIRCUIT_BROKEN)
                self._rebuild_indexes()
                self._clear_cache()
                self._trigger_event(RegistryEvent.ON_CIRCUIT_BREAK, {"provider_id": resolved_id, "consecutive_failures": consec_fail})
                self._log_audit("CIRCUIT_TRIP", resolved_id, "SYSTEM", "CRITICAL", {"consecutive_failures": consec_fail})

    async def set_failover_group(self, group_name: str, priority_list: List[str]) -> None:
        """Defines explicit routing arrays mapping automatic failover priority pathways."""
        self._ensure_active()
        async with self._lock:
            self._failover_groups[group_name] = list(priority_list)

    async def get_failover_group(self, group_name: str) -> Optional[List[str]]:
        """Retrieves priority array routing items for the designated failover group name."""
        self._ensure_active()
        async with self._lock:
            if group_name in self._failover_groups:
                return list(self._failover_groups[group_name])
            return None

    async def delete_failover_group(self, group_name: str) -> bool:
        """Deletes a designated automatic failover routing group element entirely."""
        self._ensure_active()
        async with self._lock:
            if group_name in self._failover_groups:
                del self._failover_groups[group_name]
                return True
            return False

    async def list_failover_groups(self) -> Dict[str, List[str]]:
        """Returns deep copies of all registered failover grouping structures."""
        self._ensure_active()
        async with self._lock:
            return {k: list(v) for k, v in self._failover_groups.items()}

    async def resolve_failover(self, group_name: str) -> Optional[ProviderConfiguration]:
        """Resolves target functional backup engine mapping inside active group schemas rules."""
        self._ensure_active()
        async with self._lock:
            group = self._failover_groups.get(group_name, [])
            for pid in group:
                resolved_id = self._index_by_alias.get(pid.lower(), pid)
                p = self._providers.get(resolved_id)
                if p and p.enabled and p.health_status in (ProviderStatus.ACTIVE, ProviderStatus.UNKNOWN, ProviderStatus.DEGRADED):
                    return p
            return None

    async def run_stale_cleanup(self, maximum_stale_days: int = 30) -> int:
        """Finds and purges tracking blocks containing stale records matching dead criteria rules."""
        self._ensure_active()
        async with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc)
            to_remove = []
            for pid, stats in self._statistics.items():
                ts_str = stats.last_request_timestamp or stats.registration_timestamp
                ts = datetime.datetime.fromisoformat(ts_str)
                if (now - ts).days >= maximum_stale_days:
                    to_remove.append(pid)

            for pid in to_remove:
                del self._providers[pid]
                del self._statistics[pid]
                if pid in self._version_histories:
                    del self._version_histories[pid]

            if to_remove:
                self._rebuild_indexes()
                self._clear_cache()
                self._log_audit("STALE_CLEANUP", None, "CRON", "SUCCESS", {"cleaned_count": len(to_remove)})

            return len(to_remove)

    async def get_audit_trail(self) -> List[AuditLogEntry]:
        """Returns read-only immutable system transaction audit trails data logs."""
        self._ensure_active()
        return list(self._audit_trail)

    async def clear_audit_log(self) -> None:
        """Clears the historical operational audit log logs trail structures completely."""
        self._ensure_active()
        async with self._lock:
            self._audit_trail.clear()

    async def clear_cache(self) -> None:
        """Purges internal tracking storage structures entirely."""
        async with self._lock:
            self._clear_cache()
            self._cache_hits = 0
            self._cache_misses = 0

    async def export_registry(self) -> str:
        """Exports entire tracking registry setup configurations layout cleanly into high density json."""
        self._ensure_active()
        async with self._lock:
            export_pack = {
                "registry_version": "1.2.0",
                "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "providers": [self._serialize_provider(p) for p in self._providers.values()],
                "failover_groups": self._failover_groups
            }
            return json.dumps(export_pack, indent=2)

    async def import_registry(self, json_data: str) -> int:
        """Imports configurations package structure completely from parsed configuration payloads."""
        self._ensure_active()
        try:
            pack = json.loads(json_data)
            providers_list = pack.get("providers", [])
            f_groups = pack.get("failover_groups", {})
        except Exception as e:
            raise ProviderRegistryError(
                "INVALID_IMPORT_FORMAT", "Failed parsing raw JSON package data stream completely.", {"raw_error": str(e)}
            )

        async with self._lock:
            imported_count = 0
            for raw in providers_list:
                try:
                    p = self._deserialize_provider(raw)
                    is_valid, _ = self._validate_provider_sync(p)
                    
                    # Prevent overwrites that cause alias collisions during broad imports
                    alias_collision = False
                    for alias in p.aliases:
                        norm_alias = alias.lower()
                        if norm_alias in self._index_by_alias and self._index_by_alias[norm_alias] != p.provider_id:
                            alias_collision = True
                            break

                    if is_valid and not alias_collision:
                        self._providers[p.provider_id] = p
                        if p.provider_id not in self._statistics:
                            self._statistics[p.provider_id] = ProviderStatistics()
                        imported_count += 1
                except Exception as ex:
                    self._record_error(f"Failed parsing single provider map node row context item: {str(ex)}")

            self._failover_groups.update(f_groups)
            self._rebuild_indexes()
            self._clear_cache()
            self._log_audit("IMPORT", None, "SYSTEM", "SUCCESS", {"count": imported_count})
            return imported_count

    async def backup(self) -> str:
        """Generates transactional obfuscated data schema dump instances representing exact state."""
        self._ensure_active()
        async with self._lock:
            bid = self._generate_backup_id()
            payload_raw = {
                "providers": {pid: self._serialize_provider(p) for pid, p in self._providers.items()},
                "statistics": {pid: dataclasses.asdict(s) for pid, s in self._statistics.items()},
                "failover_groups": self._failover_groups
            }
            serialized_str = json.dumps(payload_raw)
            checksum = hashlib.sha256(serialized_str.encode('utf-8')).hexdigest()
            obfuscated_payload = self._obfuscate_secret(serialized_str)

            backup_obj = RegistryBackup(
                backup_id=bid,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                version="1.2.0",
                payload=obfuscated_payload,
                checksum=checksum
            )
            self._backups[bid] = backup_obj
            return bid

    async def restore(self, backup_id: str) -> bool:
        """Restores a registry backup package completely from stored obfuscated backup payload data blocks."""
        self._ensure_active()
        async with self._lock:
            backup_obj = self._backups.get(backup_id)
            if not backup_obj:
                self._record_warning(f"Target system cluster restoration backup container {backup_id} missing.")
                return False

            try:
                decrypted_str = self._obfuscate_secret(backup_obj.payload)
                computed_cs = hashlib.sha256(decrypted_str.encode('utf-8')).hexdigest()
                if computed_cs != backup_obj.checksum:
                    raise ProviderRegistryError("BACKUP_CORRUPTED", "Checksum mismatch during secure restore pipeline execution.")

                payload = json.loads(decrypted_str)
            except Exception as e:
                self._record_error(f"Restoration integrity verification failure: {str(e)}")
                return False

            self._providers.clear()
            self._statistics.clear()
            self._failover_groups.clear()

            for pid, raw_p in payload.get("providers", {}).items():
                self._providers[pid] = self._deserialize_provider(raw_p)

            for pid, raw_s in payload.get("statistics", {}).items():
                self._statistics[pid] = ProviderStatistics(**raw_s)

            self._failover_groups.update(payload.get("failover_groups", {}))

            self._rebuild_indexes()
            self._clear_cache()
            self._log_audit("RESTORE", None, "SYSTEM", "SUCCESS", {"backup_id": backup_id})
            return True

    async def create_snapshot(self) -> str:
        """Creates an uncompressed internal snapshot instance map representation frame."""
        self._ensure_active()
        async with self._lock:
            sid = self._generate_snapshot_id()
            snap = RegistrySnapshot(
                snapshot_id=sid,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                providers={pid: self._serialize_provider(p) for pid, p in self._providers.items()},
                statistics={pid: dataclasses.asdict(s) for pid, s in self._statistics.items()},
            )
            self._snapshots[sid] = snap
            return sid

    async def restore_snapshot(self, snapshot_id: str) -> bool:
        """Overwrites local working memory blocks cleanly to target configuration snapshot instance points."""
        self._ensure_active()
        async with self._lock:
            snap = self._snapshots.get(snapshot_id)
            if not snap:
                return False

            self._providers.clear()
            self._statistics.clear()

            for pid, raw_p in snap.providers.items():
                self._providers[pid] = self._deserialize_provider(raw_p)

            for pid, raw_s in snap.statistics.items():
                self._statistics[pid] = ProviderStatistics(**raw_s)

            self._rebuild_indexes()
            self._clear_cache()
            return True

    async def run_diagnostics(self) -> RegistryDiagnosticReport:
        """Evaluates operational logic parameters and structural validation schemas across runtime layers."""
        self._ensure_active()
        async with self._lock:
            total = len(self._providers)
            active = sum(1 for p in self._providers.values() if p.enabled and p.health_status == ProviderStatus.ACTIVE)
            disabled = sum(1 for p in self._providers.values() if not p.enabled)
            unhealthy = sum(1 for p in self._providers.values() if p.health_status == ProviderStatus.UNHEALTHY)
            cb_tripped = sum(1 for p in self._providers.values() if p.health_status == ProviderStatus.CIRCUIT_BROKEN)

            # High density multi-alias and structural collision analysis verification
            alias_counts: Dict[str, List[str]] = {}
            for pid, p in self._providers.items():
                for alias in p.aliases:
                    norm = alias.lower()
                    if norm not in alias_counts:
                        alias_counts[norm] = []
                    alias_counts[norm].append(pid)
            
            duplicates = [norm for norm, pids in alias_counts.items() if len(pids) > 1]

            failures = {}
            for pid, p in self._providers.items():
                is_valid, errs = self._validate_provider_sync(p)
                if not is_valid:
                    failures[pid] = errs

            h_summary = {pid: p.health_status.value for pid, p in self._providers.items()}

            total_reqs = sum(s.request_count for s in self._statistics.values())
            total_succ = sum(s.success_count for s in self._statistics.values())
            total_fail = sum(s.failure_count for s in self._statistics.values())
            total_cost = sum(s.cost_accumulated for s in self._statistics.values())

            total_hits_misses = self._cache_hits + self._cache_misses
            c_util = (self._cache_hits / total_hits_misses * 100.0) if total_hits_misses > 0 else 0.0

            current_checksum = self._calculate_checksum()

            return RegistryDiagnosticReport(
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                total_registered_providers=total,
                active_providers=active,
                disabled_providers=disabled,
                unhealthy_providers=unhealthy,
                circuit_broken_providers=cb_tripped,
                duplicate_providers=duplicates,
                validation_failures=failures,
                warning_count=len(self._warning_log),
                error_count=len(self._error_log),
                cache_statistics={
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                    "utilization_percentage": c_util,
                },
                health_summary=h_summary,
                registry_statistics={
                    "aggregate_request_count": total_reqs,
                    "aggregate_success_count": total_succ,
                    "aggregate_failure_count": total_fail,
                    "aggregate_cost_accumulated": total_cost,
                },
                integrity_verified=(len(failures) == 0 and len(duplicates) == 0),
                checksum=current_checksum
            )

    async def export_report(self) -> str:
        """Runs the diagnostics engine and transforms the resultant report object into JSON."""
        rep = await self.run_diagnostics()
        return json.dumps(dataclasses.asdict(rep), indent=2)

    async def shutdown(self) -> None:
        """Sets internal active lifecycle flags off to gracefully terminate handling workflows."""
        async with self._lock:
            self._is_shutdown = True
            self._clear_cache()
            self._providers.clear()
            self._statistics.clear()
            self._failover_groups.clear()
            self._event_listeners.clear()
            self._version_histories.clear()
            logger.info("Provider Registry system successfully shut down.")
