#!/usr/bin/env python3
"""
Cabangile AI Studio - Enterprise Provider Factory Module

Location: studio/providers/provider_factory.py
Language: Python 3.11+
Dependencies: Standard Library Only

This module implements a production-ready, thread-safe, and asynchronous provider
factory adhering to Clean Architecture, SOLID, and Domain-Driven Design (DDD) principles.
It handles dynamic runtime mapping registration, automated alias resolution, robust
Least Recently Used (LRU) cache eviction, strict structural/inheritance validation,
and state lifecycle orchestration without depending directly on concrete providers.
"""

import asyncio
import dataclasses
import datetime
import enum
import importlib
import json
import logging
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger("studio.providers.provider_factory")


# ============================================================================
# 1. ENTERPRISE EXCEPTION HIERARCHY
# ============================================================================

class ProviderFactoryError(Exception):
    """Base exception for all errors originating from the Provider Factory."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.datetime.now(datetime.timezone.utc)


class ProviderCreationError(ProviderFactoryError):
    """Raised when an error occurs during the instantiation of a provider."""


class ProviderRegistrationError(ProviderFactoryError):
    """Raised when a validation or state violation occurs during provider registration."""


class UnknownProviderError(ProviderFactoryError):
    """Raised when an operation requests a provider identifier or alias that is not registered."""


# ============================================================================
# 2. ENUMERATIONS
# ============================================================================

class FactoryOperation(enum.Enum):
    """Supported tracking operations within the execution boundaries of the factory."""
    REGISTER = "REGISTER"
    UNREGISTER = "UNREGISTER"
    CREATE = "CREATE"
    VALIDATE = "VALIDATE"
    IMPORT = "IMPORT"
    EXPORT = "EXPORT"
    CLEAR = "CLEAR"
    SHUTDOWN = "SHUTDOWN"


class FactoryStatus(enum.Enum):
    """Operational status flags indicating the internal lifecycle state of the factory."""
    INITIALIZED = "INITIALIZED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    TERMINATED = "TERMINATED"


# ============================================================================
# 3. IMMUTABLE DATACLASSES
# ============================================================================

@dataclasses.dataclass(frozen=True)
class ProviderFactoryStatistics:
    """Read-only telemetry snapshot representing internal processing metrics."""
    total_registered_providers: int
    total_provider_creations: int
    successful_creations: int
    failed_creations: int
    cache_hits: int
    cache_misses: int
    average_creation_time_ms: float
    registry_estimated_size_bytes: int
    uptime_seconds: float
    last_creation_timestamp: Optional[datetime.datetime]


@dataclasses.dataclass(frozen=True)
class ProviderFactoryConfiguration:
    """Runtime structural and behavior configurations for the factory instance."""
    enable_cache: bool = True
    strict_mode: bool = True
    max_cache_size: int = 1000
    allowed_version_range: str = ">=1.0.0"
    base_provider_class_path: str = "studio.providers.base_provider.BaseProvider"


@dataclasses.dataclass(frozen=True)
class ProviderRegistrationRecord:
    """Encapsulated validation domain record for a successfully registered provider class."""
    provider_type: str
    target_class: Type[Any]
    aliases: Set[str]
    metadata: Dict[str, Any]
    registered_at: datetime.datetime
    version: str


@dataclasses.dataclass(frozen=True)
class ProviderCreationResult:
    """Output envelope returned after executing an independent creation request pipeline."""
    provider_instance: Any
    creation_time_ms: float
    cached: bool
    timestamp: datetime.datetime


# ============================================================================
# 4. PROVIDER FACTORY CORE IMPLEMENTATION
# ============================================================================

class ProviderFactory:
    """
    Thread-safe, asynchronous enterprise manufacturing pipeline for AI Providers.
    
    Operates via dynamic module/class resolution and strict inheritance validation,
    eliminating dependencies on hardcoded concrete implementations.
    """

    def __init__(self, configuration: Optional[ProviderFactoryConfiguration] = None) -> None:
        """Initializes internal storage mappings, locks, operational cache, and instrumentation."""
        self._config = configuration or ProviderFactoryConfiguration()
        self._lock = asyncio.Lock()
        self._status = FactoryStatus.INITIALIZED
        self._start_time = datetime.datetime.now(datetime.timezone.utc)

        # Core registries
        self._registry: Dict[str, ProviderRegistrationRecord] = {}
        self._alias_map: Dict[str, str] = {}
        
        # Performance cache state maps - using an ordered dict strategy for LRU eviction
        self._creation_cache: Dict[str, Any] = {}
        self._validation_cache: Set[str] = set()

        # Telemetry metrics primitives
        self._total_creations = 0
        self._successful_creations = 0
        self._failed_creations = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_creation_time_ms = 0.0
        self._last_creation_timestamp: Optional[datetime.datetime] = None

        # Structural observation hooks and audit log
        self._audit_log: List[Dict[str, Any]] = []
        self._callbacks: List[Callable[[FactoryOperation, Dict[str, Any]], Awaitable[None]]] = []
        
        self._status = FactoryStatus.ACTIVE

    async def register_provider_class(
        self,
        provider_type: str,
        target_class: Type[Any],
        aliases: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0"
    ) -> None:
        """
        Validates inheritance and structure, then registers a type constructor into the isolation boundaries.
        
        Raises ProviderRegistrationError upon failing signature validation or constraint conflicts.
        """
        async with self._lock:
            self._ensure_active()
            norm_type = provider_type.strip()
            norm_aliases = {a.strip() for a in (aliases or set()) if a.strip()}
            meta = metadata or {}

            if not norm_type:
                raise ProviderRegistrationError("Provider type string cannot be empty.")
            
            if norm_type in self._registry:
                raise ProviderRegistrationError(f"Duplicate registration error: '{norm_type}' is already registered.")

            for alias in norm_aliases:
                if alias in self._alias_map or alias in self._registry:
                    raise ProviderRegistrationError(f"Collision error: Alias or Type '{alias}' occupies namespace.")

            await self._validate_target_structure(norm_type, target_class, meta)

            record = ProviderRegistrationRecord(
                provider_type=norm_type,
                target_class=target_class,
                aliases=norm_aliases,
                metadata=meta,
                registered_at=datetime.datetime.now(datetime.timezone.utc),
                version=version
            )

            self._registry[norm_type] = record
            for alias in norm_aliases:
                self._alias_map[alias] = norm_type

            await self._record_audit(
                FactoryOperation.REGISTER,
                {
                    "provider_type": norm_type, 
                    "aliases": list(norm_aliases), 
                    "version": version,
                    "module": target_class.__module__,
                    "class": target_class.__qualname__
                }
            )
            logger.info("Successfully registered factory entity mapping for provider type '%s'.", norm_type)

    async def unregister_provider_class(self, provider_type: str) -> None:
        """Removes a registered entity mapping safely and invalidates dependent structural caches."""
        async with self._lock:
            self._ensure_active()
            resolved_type = self._resolve_type(provider_type)

            if resolved_type not in self._registry:
                raise UnknownProviderError(f"Cannot unregister non-existent provider type: '{provider_type}'.")

            record = self._registry.pop(resolved_type)
            for alias in record.aliases:
                self._alias_map.pop(alias, None)

            self._creation_cache.pop(resolved_type, None)
            self._validation_cache.discard(resolved_type)

            await self._record_audit(FactoryOperation.UNREGISTER, {"provider_type": resolved_type})
            logger.warning("Unregistered provider mapping and evacuated corresponding memory caches for: '%s'.", resolved_type)

    async def create_provider(self, provider_identifier: str, **kwargs: Any) -> ProviderCreationResult:
        """Instantiates, tracks performance, manages LRU caches, and returns target entity instances."""
        start_time = datetime.datetime.now(datetime.timezone.utc)
        async with self._lock:
            self._ensure_active()
            self._total_creations += 1
            
            try:
                resolved_type = self._resolve_type(provider_identifier)
            except UnknownProviderError as err:
                self._failed_creations += 1
                await self._record_audit(FactoryOperation.CREATE, {"identifier": provider_identifier, "success": False})
                raise ProviderCreationError(str(err)) from err

            # Cache HIT execution path (Only for argument-free instantiation to avoid state leaks)
            if self._config.enable_cache and not kwargs:
                if resolved_type in self._creation_cache:
                    self._cache_hits += 1
                    # Refresh LRU position by re-inserting key
                    instance = self._creation_cache.pop(resolved_type)
                    self._creation_cache[resolved_type] = instance
                    
                    duration = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds() * 1000.0
                    return ProviderCreationResult(
                        provider_instance=instance,
                        creation_time_ms=duration,
                        cached=True,
                        timestamp=datetime.datetime.now(datetime.timezone.utc)
                    )
                self._cache_misses += 1
            else:
                if not self._config.enable_cache:
                    self._cache_misses += 1

            record = self._registry[resolved_type]
            
            try:
                if inspect.iscoroutinefunction(record.target_class):
                    raise ProviderCreationError("Async factory constructor functions are structurally blocked via schema rules.")
                
                instance = record.target_class(**kwargs)
                
            except Exception as exc:
                self._failed_creations += 1
                await self._record_audit(FactoryOperation.CREATE, {"provider_type": resolved_type, "success": False, "error": str(exc)})
                raise ProviderCreationError(f"Failed initialization structure sequence for '{resolved_type}': {str(exc)}") from exc

            self._successful_creations += 1
            end_time = datetime.datetime.now(datetime.timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000.0
            self._total_creation_time_ms += duration_ms
            self._last_creation_timestamp = end_time

            # Handle LRU Cache Eviction and Storage
            if self._config.enable_cache and not kwargs:
                if len(self._creation_cache) >= self._config.max_cache_size:
                    # Evict the oldest item (first key in the dictionary)
                    oldest_key = next(iter(self._creation_cache))
                    self._creation_cache.pop(oldest_key)
                    logger.debug("LRU eviction triggered. Removed provider instance key: '%s' from cache.", oldest_key)
                
                self._creation_cache[resolved_type] = instance

            await self._record_audit(
                FactoryOperation.CREATE, 
                {"provider_type": resolved_type, "success": True, "duration_ms": duration_ms}
            )
            
            return ProviderCreationResult(
                provider_instance=instance,
                creation_time_ms=duration_ms,
                cached=False,
                timestamp=end_time
            )

    async def create_from_configuration(self, configuration_dict: Dict[str, Any]) -> ProviderCreationResult:
        """Parses decoupled structured configurations dictionary arguments and creates targets."""
        if "provider_type" not in configuration_dict:
            raise ProviderCreationError("Configuration mapping parameters dictionary must define 'provider_type'.")
        
        target_type = configuration_dict["provider_type"]
        parameters = configuration_dict.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ProviderCreationError("Configuration parameter properties payload segment must be encapsulated as a dictionary object.")
        
        return await self.create_provider(target_type, **parameters)

    async def create_multiple(self, identifiers: List[Union[str, Dict[str, Any]]]) -> List[ProviderCreationResult]:
        """Orchestrates structured concurrent batched initializations using asynchronous task mapping layers safely."""
        tasks = []
        for identifying_entity in identifiers:
            if isinstance(identifying_entity, str):
                tasks.append(self.create_provider(identifying_entity))
            elif isinstance(identifying_entity, dict):
                tasks.append(self.create_from_configuration(identifying_entity))
            else:
                raise ProviderCreationError("Batch entry item definition formatting shape context structurally invalid.")
        
        return list(await asyncio.gather(*tasks))

    async def provider_exists(self, provider_identifier: str) -> bool:
        """Determines if a target identity key mapping exists inside the registered dictionary context."""
        async with self._lock:
            try:
                self._resolve_type(provider_identifier)
                return True
            except UnknownProviderError:
                return False

    async def list_registered_provider_types(self) -> List[str]:
        """Extracts and outputs sorted explicit underlying unique operational mapping keys."""
        async with self._lock:
            return sorted(list(self._registry.keys()))

    async def list_registered_classes(self) -> List[Tuple[str, Type[Any]]]:
        """Provides direct access mapping elements output tuple slices safely."""
        async with self._lock:
            return [(k, v.target_class) for k, v in self._registry.items()]

    async def get_registration(self, provider_identifier: str) -> ProviderRegistrationRecord:
        """Finds and returns a snapshot representation mapping record describing registered properties."""
        async with self._lock:
            resolved = self._resolve_type(provider_identifier)
            return self._registry[resolved]

    async def validate_registration(self, provider_type: str) -> bool:
        """Inspects structural validation matrices against live system requirements caches."""
        async with self._lock:
            resolved = self._resolve_type(provider_type)
            return resolved in self._validation_cache

    async def clear_registry(self) -> None:
        """Completely purges registration matrices data parameters alongside performance tracking caches."""
        async with self._lock:
            self._ensure_active()
            self._registry.clear()
            self._alias_map.clear()
            self._creation_cache.clear()
            self._validation_cache.clear()
            await self._record_audit(FactoryOperation.CLEAR, {})
            logger.critical("Factory identity dynamic registry matrices data parameters completely purged successfully.")

    async def export_registry(self) -> str:
        """Generates static structural JSON metadata export definitions containing fully qualified class names."""
        async with self._lock:
            export_payload = {}
            for key, rec in self._registry.items():
                export_payload[key] = {
                    "provider_type": rec.provider_type,
                    "aliases": list(rec.aliases),
                    "metadata": rec.metadata,
                    "version": rec.version,
                    "module": rec.target_class.__module__,
                    "class": rec.target_class.__qualname__
                }
            await self._record_audit(FactoryOperation.EXPORT, {"count": len(export_payload)})
            return json.dumps(export_payload, indent=4)

    async def import_registry(self, serialized_json: str) -> None:
        """
        Deserializes an export payload and dynamically loads real provider classes into runtime boundaries.
        
        Uses full reflective module pathing to completely reconstruct functional registration state records.
        """
        async with self._lock:
            self._ensure_active()
            try:
                data = json.loads(serialized_json)
            except Exception as exc:
                raise ProviderRegistrationError("Target payload deserialization processing failed structurally.") from exc

            for key, fields in data.items():
                if not all(k in fields for k in ("provider_type", "version", "module", "class")):
                    raise ProviderRegistrationError("Required configuration fields missing inside the raw source mapping schema data blocks.")
                
                provider_type = fields["provider_type"]
                module_path = fields["module"]
                class_name = fields["class"]

                try:
                    # Dynamically reflect and load real underlying execution modules and target classes
                    module = importlib.import_module(module_path)
                    target_class = getattr(module, class_name)
                except Exception as exc:
                    raise ProviderRegistrationError(
                        f"Failed dynamic resolution path sequence for provider class '{module_path}.{class_name}': {str(exc)}"
                    ) from exc

                # Run structural assertions over the imported real class context
                meta = fields.get("metadata", {})
                await self._validate_target_structure(provider_type, target_class, meta)

                record = ProviderRegistrationRecord(
                    provider_type=provider_type,
                    target_class=target_class,
                    aliases=set(fields.get("aliases", [])),
                    metadata=meta,
                    registered_at=datetime.datetime.now(datetime.timezone.utc),
                    version=fields["version"]
                )
                
                self._registry[provider_type] = record
                for alias in record.aliases:
                    self._alias_map[alias] = provider_type
            
            await self._record_audit(FactoryOperation.IMPORT, {"count": len(data)})

    async def get_statistics(self) -> ProviderFactoryStatistics:
        """Calculates running telemetry metrics snapshots across active pipeline lifecycles."""
        async with self._lock:
            uptime = (datetime.datetime.now(datetime.timezone.utc) - self._start_time).total_seconds()
            avg_time = 0.0
            if self._successful_creations > 0:
                avg_time = self._total_creation_time_ms / self._successful_creations

            # Documented estimation layer based on current tracking boundaries
            est_size = len(self._registry) * 512 + len(self._alias_map) * 128 + len(self._creation_cache) * 1024

            return ProviderFactoryStatistics(
                total_registered_providers=len(self._registry),
                total_provider_creations=self._total_creations,
                successful_creations=self._successful_creations,
                failed_creations=self._failed_creations,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                average_creation_time_ms=avg_time,
                registry_estimated_size_bytes=est_size,
                uptime_seconds=uptime,
                last_creation_timestamp=self._last_creation_timestamp
            )

    async def reset_statistics(self) -> None:
        """Resets telemetry accumulation metrics safely without disrupting state registries."""
        async with self._lock:
            self._total_creations = 0
            self._successful_creations = 0
            self._failed_creations = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_creation_time_ms = 0.0
            self._last_creation_timestamp = None
            logger.info("Factory execution performance tracking telemetry parameters reset successfully.")

    async def shutdown(self) -> None:
        """Gracefully halts factory processing, sets operational flags, and purges structural operational memory."""
        async with self._lock:
            self._status = FactoryStatus.SHUTTING_DOWN
            await self._record_audit(FactoryOperation.SHUTDOWN, {})
            
            # Explicitly wipe all states, callbacks, registry elements and audit data logs to avoid memory leaks
            self._registry.clear()
            self._alias_map.clear()
            self._creation_cache.clear()
            self._validation_cache.clear()
            self._audit_log.clear()
            self._callbacks.clear()
            
            self._status = FactoryStatus.TERMINATED
            logger.info("Factory state context has successfully executed terminal shutdown and cleared internal structures.")

    def register_event_callback(self, callback: Callable[[FactoryOperation, Dict[str, Any]], Awaitable[None]]) -> None:
        """Injects custom structural observation event processing logic callbacks safely."""
        if not callable(callback):
            raise ValueError("Target callback must be fully callable.")
        self._callbacks.append(callback)

    # ============================================================================
    # INTERNAL HELPER & DEFENSIVE VALIDATION METHODS
    # ============================================================================

    def _ensure_active(self) -> None:
        """Guards and ensures operation executions cannot traverse invalid or non-active factory boundaries."""
        if self._status not in (FactoryStatus.INITIALIZED, FactoryStatus.ACTIVE):
            raise ProviderFactoryError(f"Operational interaction failed: Current factory status state is: {self._status.name}")

    def _resolve_type(self, identifier: str) -> str:
        """Decouples aliases and transforms standard system mapping indicators automatically."""
        norm = identifier.strip()
        if norm in self._registry:
            return norm
        if norm in self._alias_map:
            return self._alias_map[norm]
        raise UnknownProviderError(f"Target identifier entity reference target resolution lookup failure: '{norm}' does not map to any registered provider.")

    async def _validate_target_structure(self, provider_type: str, target_class: Type[Any], metadata: Dict[str, Any]) -> None:
        """Enforces inheritance verification, structural contracts, and required metadata matrices."""
        if not inspect.isclass(target_class):
            raise ProviderRegistrationError(f"Registration target for '{provider_type}' must point to a concrete class type.")

        # Dynamically evaluate real inheritance hierarchies against designated base abstractions
        if self._config.strict_mode:
            try:
                base_module_name, base_class_name = self._config.base_provider_class_path.rsplit(".", 1)
                base_module = importlib.import_module(base_module_name)
                base_provider_class = getattr(base_module, base_class_name)
                
                if not issubclass(target_class, base_provider_class):
                    raise ProviderRegistrationError(
                        f"Invalid inheritance: Class '{target_class.__name__}' must derive from base definition reference '{base_class_name}'."
                    )
            except (ImportError, AttributeError) as exc:
                logger.warning("Skipping concrete base class subtyping constraint check because reference target path was unreachable: %s", str(exc))

        # Core required runtime methods interface signatures checking boundaries
        required_methods = ["process_response", "route_model", "check_health"]
        for method_name in required_methods:
            if not hasattr(target_class, method_name) or not callable(getattr(target_class, method_name)):
                if self._config.strict_mode:
                    raise ProviderRegistrationError(
                        f"SOLID interface enforcement error: Class '{target_class.__name__}' is missing the required functional method: '{method_name}'"
                    )

        try:
            inspect.signature(target_class.__init__)
        except Exception as exc:
            raise ProviderRegistrationError(f"Unable to parse initialization validation signatures for: '{target_class.__name__}'.") from exc

        # Internal metadata requirements enforcement constraints checking layers
        required_metadata = ["description", "author"]
        if self._config.strict_mode:
            for metakey in required_metadata:
                if metakey not in metadata:
                    raise ProviderRegistrationError(f"Validation failure: Provider type metadata structural specification requires '{metakey}' definition parameters.")

        self._validation_cache.add(provider_type)

    async def _record_audit(self, operation: FactoryOperation, data: Dict[str, Any]) -> None:
        """Asynchronously dispatches structural logging tracking blocks to downstream dependencies safely."""
        payload = {
            "operation": operation.name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data": data
        }
        self._audit_log.append(payload)
        
        for cb in self._callbacks:
            try:
                await cb(operation, data)
            except Exception as cb_exc:
                logger.error("Internal event hook validation observer tracing callback faulted execution sequence: %s", str(cb_exc))
