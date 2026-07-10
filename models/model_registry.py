import asyncio
import copy
import fnmatch
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("CabangileAIStudio.ModelRegistry")

class ModelRegistryError(Exception):
    """Base exception for all Model Registry operations in Cabangile AI Studio."""
    pass

class EventEmitter:
    """Thread-safe and async-compatible internal event system modeled after Node.js EventEmitter."""
    
    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[..., Any]]] = {}
        self._lock = threading.RLock()
        
    def on(self, event: str, listener: Callable[..., Any]) -> None:
        """Register a synchronous or asynchronous listener for an event."""
        with self._lock:
            if event not in self._listeners:
                self._listeners[event] = []
            if listener not in self._listeners[event]:
                self._listeners[event].append(listener)
                
    def off(self, event: str, listener: Callable[..., Any]) -> None:
        """Remove a registered listener from an event."""
        with self._lock:
            if event in self._listeners:
                try:
                    self._listeners[event].remove(listener)
                except ValueError:
                    pass
                    
    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Emit an event, executing all synchronous listeners and scheduling async ones."""
        with self._lock:
            listeners = list(self._target_listeners(event))
            
        for listener in listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    try:
                        loop = asyncio.get_running_loop()
                        if loop.is_running():
                            loop.create_task(listener(*args, **kwargs))
                    except RuntimeError:
                        # No running event loop, execute in a temporary loop safely
                        asyncio.run(listener(*args, **kwargs))
                else:
                    listener(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in listener for event '{event}': {e}", exc_info=True)
                
    def _target_listeners(self, event: str) -> List[Callable[..., Any]]:
        return self._listeners.get(event, []) + self._listeners.get("*", [])

@dataclass
class ModelDefinition:
    """Data representation of an AI model within the registry."""
    id: str
    name: str
    family: str
    provider: str
    version: str
    description: str
    contextWindow: int
    maxOutputTokens: int
    pricing: Dict[str, float]  # e.g., {"input_1k": 0.0015, "output_1k": 0.002}
    capabilities: List[str]    # e.g., ["streaming", "vision", "function_calling", "embeddings", "reasoning"]
    qualityScore: float
    speedScore: float
    reliabilityScore: float
    popularityScore: float
    availability: str          # e.g., "online", "offline"
    tags: List[str]
    metadata: Dict[str, Any]
    deprecated: bool = False
    experimental: bool = False
    priority: int = 0
    createdAt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updatedAt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelDefinition":
        data_copy = copy.deepcopy(data)
        return cls(**data_copy)

class ModelRegistry:
    """
    Enterprise-grade, thread-safe, and async-safe Model Registry for Cabangile AI Studio.
    Provides indexing, metadata querying, aliases, sorting, filtering, and automated audit metrics.
    """
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events = EventEmitter()
        
        # Primary storage
        self._models: Dict[str, ModelDefinition] = {}
        self._aliases: Dict[str, str] = {}
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        
        # Multi-indexes for fast access
        self._index_provider: Dict[str, Set[str]] = {}
        self._index_family: Dict[str, Set[str]] = {}
        self._index_capability: Dict[str, Set[str]] = {}
        self._index_tag: Dict[str, Set[str]] = {}
        
        # System status and statistics
        self._start_time = time.time()
        self._stats = {
            "registered_models": 0,
            "removed_models": 0,
            "updates": 0,
            "searches": 0,
            "cache_hits": 0,
            "lookups": 0,
            "exports": 0,
            "imports": 0,
            "backups": 0,
            "restores": 0,
            "validation_failures": 0,
            "alias_usage": 0,
            "capability_usage": 0
        }
        self._audit_trail: List[Dict[str, Any]] = []
        
        # Background cleanup orchestration
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._cleanup_running = False
        
        # Internal Query/Lookup Cache
        self._cache: Dict[str, Any] = {}
        
    # --- Properties and Compatibility APIs ---
    
    @property
    def events(self) -> EventEmitter:
        """Expose the internal event management layer."""
        return self._events
        
    def getModels(self) -> List[Dict[str, Any]]:
        """Synchronous variant required for seamless compatibility with model_router.py."""
        with self._lock:
            self._stats["lookups"] += 1
            return [m.to_dict() for m in self._models.values()]
            
    async def getModelsAsync(self) -> List[Dict[str, Any]]:
        """Asynchronous variant required for seamless compatibility with model_router.py."""
        return self.getModels()
        
    def __getattr__(self, name: str) -> Any:
        """
        Dynamic descriptor fallback to support calling conventions where getModels 
        is expected to act as both a direct callable function and an awaitable coroutine.
        """
        if name == "getModels":
            class CompatibleCallable:
                def __init__(self, registry: "ModelRegistry"):
                    self.registry = registry
                def __call__(self) -> List[Dict[str, Any]]:
                    return self.registry.getModels()
                def __await__(self):
                    async def _async_wrapper():
                        return self.registry.getModels()
                    return _async_wrapper().__await__()
            return CompatibleCallable(self)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    # --- Core Lifecycle Management ---
    
    def register_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Registers an AI model configuration after schema validation."""
        with self._lock:
            self.validate_model(model_data)
            model_id = model_data["id"]
            
            if model_id in self._models:
                self._stats["validation_failures"] += 1
                raise ModelRegistryError(f"Model ID '{model_id}' is already registered. Use update_model instead.")
                
            now = datetime.now(timezone.utc).isoformat()
            model_data.setdefault("createdAt", now)
            model_data.setdefault("updatedAt", now)
            
            model = ModelDefinition.from_dict(model_data)
            self._models[model_id] = model
            
            self._rebuild_indices_for_model(model)
            self._clear_cache()
            
            self._stats["registered_models"] += 1
            self._log_audit("REGISTER", {"id": model_id, "provider": model.provider})
            self._events.emit("registered", model.to_dict())
            
            return model.to_dict()
            
    def unregister_model(self, model_id: str) -> Dict[str, Any]:
        """Unregisters an active model, clearing references and updating associated indexes."""
        with self._lock:
            resolved_id = self._resolve_id_internal(model_id)
            if resolved_id not in self._models:
                raise ModelRegistryError(f"Model tracking target '{model_id}' does not exist.")
                
            model = self._models.pop(resolved_id)
            
            # Remove associated active aliases
            aliases_to_remove = [k for k, v in self._aliases.items() if v == resolved_id]
            for alias in aliases_to_remove:
                del self._aliases[alias]
                
            self._remove_indices_for_model(model)
            self._clear_cache()
            
            self._stats["removed_models"] += 1
            self._log_audit("UNREGISTER", {"id": resolved_id})
            self._events.emit("unregistered", model.to_dict())
            
            return model.to_dict()
            
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Updates specific settings of a registered model definition."""
        with self._lock:
            resolved_id = self._resolve_id_internal(model_id)
            if resolved_id not in self._models:
                raise ModelRegistryError(f"Model target '{model_id}' not found for mutation updates.")
                
            current_dict = self._models[resolved_id].to_dict()
            
            # Protected immutability rules
            for invariant in ["id", "provider", "createdAt"]:
                if invariant in updates and updates[invariant] != current_dict[invariant]:
                    raise ModelRegistryError(f"Field updating constraint violation on immutable item: '{invariant}'.")
                    
            current_dict.update(updates)
            current_dict["updatedAt"] = datetime.now(timezone.utc).isoformat()
            
            # Revalidate schema
            self.validate_model(current_dict)
            
            old_model = self._models[resolved_id]
            self._remove_indices_for_model(old_model)
            
            new_model = ModelDefinition.from_dict(current_dict)
            self._models[resolved_id] = new_model
            self._rebuild_indices_for_model(new_model)
            
            self._clear_cache()
            self._stats["updates"] += 1
            self._log_audit("UPDATE", {"id": resolved_id, "changed": list(updates.keys())})
            self._events.emit("updated", new_model.to_dict())
            
            return new_model.to_dict()
            
    # --- Query and Lookup Interfaces ---
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Retrieves details of a registered model by its true ID or defined alias."""
        with self._lock:
            resolved_id = self._resolve_id_internal(model_id)
            if resolved_id not in self._models:
                raise ModelRegistryError(f"Model configuration tracking element '{model_id}' could not be located.")
            self._stats["lookups"] += 1
            return self._models[resolved_id].to_dict()
            
    def get_models(self) -> List[Dict[str, Any]]:
        """Retrieves all registered models."""
        return self.getModels()
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Retrieves all models marked available and not deprecated."""
        with self._lock:
            return [m.to_dict() for m in self._models.values() if m.availability == "online" and not m.deprecated]
            
    def get_models_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """Retrieves models indexed under the given provider name."""
        with self._lock:
            ids = self._index_provider.get(provider.lower(), set())
            return [self._models[m_id].to_dict() for m_id in ids if m_id in self._models]
            
    def get_models_by_family(self, family: str) -> List[Dict[str, Any]]:
        """Retrieves models indexed under the given family categorization."""
        with self._lock:
            ids = self._index_family.get(family.lower(), set())
            return [self._models[m_id].to_dict() for m_id in ids if m_id in self._models]
            
    def get_models_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Retrieves models matching a functional capability tag."""
        with self._lock:
            self._stats["capability_usage"] += 1
            ids = self._index_capability.get(capability.lower(), set())
            return [self._models[m_id].to_dict() for m_id in ids if m_id in self._models]
            
    def get_models_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Retrieves models tracking a descriptive user metadata tag."""
        with self._lock:
            ids = self._index_tag.get(tag.lower(), set())
            return [self._models[m_id].to_dict() for m_id in ids if m_id in self._models]
            
    def has_model(self, model_id: str) -> bool:
        """Validates existence presence without extracting complete configuration state payload."""
        with self._lock:
            try:
                self._resolve_id_internal(model_id)
                return True
            except ModelRegistryError:
                return False
                
    def count(self) -> int:
        """Total number of primary tracking entities recorded."""
        with self._lock:
            return len(self._models)
            
    def clear(self) -> None:
        """Flushes storage components, registries, indexes, and runtime query metrics cache."""
        with self._lock:
            self._models.clear()
            self._aliases.clear()
            self._index_provider.clear()
            self._index_family.clear()
            self._index_capability.clear()
            self._index_tag.clear()
            self._clear_cache()
            self._log_audit("CLEAR", {})
            self._events.emit("cleared", {})
            
    # --- Taxonomy Extraction APIs ---
    
    def list_families(self) -> List[str]:
        """Provides an atomic listing of distinct model architectures currently registered."""
        with self._lock:
            return list({m.family for m in self._models.values()})
            
    def list_providers(self) -> List[str]:
        """Provides an atomic listing of current discrete provider infrastructures."""
        with self._lock:
            return list({m.provider for m in self._models.values()})
            
    # --- Verification & Schema Enforcement ---
    
    def validate_model(self, model_data: Dict[str, Any]) -> None:
        """Validates model schema compliance for enterprise platform orchestration integration."""
        required_fields = {
            "id": str, "name": str, "family": str, "provider": str, "version": str,
            "description": str, "contextWindow": int, "maxOutputTokens": int,
            "pricing": dict, "capabilities": list, "qualityScore": (int, float),
            "speedScore": (int, float), "reliabilityScore": (int, float),
            "popularityScore": (int, float), "availability": str, "tags": list,
            "metadata": dict
        }
        for field_name, expected_type in required_fields.items():
            if field_name not in model_data:
                self._stats["validation_failures"] += 1
                raise ModelRegistryError(f"Schema violation: Missing field '{field_name}'.")
            val = model_data[field_name]
            if not isinstance(val, expected_type):
                self._stats["validation_failures"] += 1
                raise ModelRegistryError(
                    f"Schema type mismatch for item '{field_name}'. Expected {expected_type}, got {type(val)}."
                )
                
        # Confirm critical values within business-valid ranges
        if model_data["contextWindow"] <= 0 or model_data["maxOutputTokens"] <= 0:
            raise ModelRegistryError("Context Windows or Response tokens scale lengths must exceed 0 limits.")
            
    # --- Alias Virtual Resolution Layer ---
    
    def register_alias(self, alias: str, target_model_id: str) -> None:
        """Registers a logical alias mapping pointing to a model configuration."""
        with self._lock:
            if target_model_id not in self._models:
                raise ModelRegistryError(f"Target model tracking ID mapping missing: {target_model_id}")
            if alias in self._models:
                raise ModelRegistryError("Cannot assign an alias overriding an explicit unique registered model ID.")
            self._aliases[alias] = target_model_id
            self._log_audit("ALIAS_REGISTER", {"alias": alias, "target": target_model_id})
            
    def unregister_alias(self, alias: str) -> None:
        """Deletes a logical alias routing key reference."""
        with self._lock:
            if alias not in self._aliases:
                raise ModelRegistryError(f"Alias targeted mapping path '{alias}' does not exist.")
            del self._aliases[alias]
            self._log_audit("ALIAS_UNREGISTER", {"alias": alias})
            
    def resolve_alias(self, alias: str) -> str:
        """Resolves an alias to its underlying model ID."""
        with self._lock:
            return self._resolve_id_internal(alias)
            
    def register_capability(self, model_id: str, capability: str) -> None:
        """Dynamically appends a functional validation tag runtime index capability designation."""
        with self._lock:
            resolved_id = self._resolve_id_internal(model_id)
            model = self._models[resolved_id]
            if capability not in model.capabilities:
                model.capabilities.append(capability)
                model.updatedAt = datetime.now(timezone.utc).isoformat()
                self._rebuild_indices_for_model(model)
                self._clear_cache()
                
    def unregister_capability(self, model_id: str, capability: str) -> None:
        """Removes a capability designation tag from a model configuration profile."""
        with self._lock:
            resolved_id = self._resolve_id_internal(model_id)
            model = self._models[resolved_id]
            if capability in model.capabilities:
                model.capabilities.remove(capability)
                model.updatedAt = datetime.now(timezone.utc).isoformat()
                self._remove_indices_for_model(model)
                self._rebuild_indices_for_model(model)
                self._clear_cache()
                
    # --- Search, Filter and Sorter Framework engines ---
    
    def search_models(self, query_string: str = "*", filters: Optional[Dict[str, Any]] = None,
                      sort_by: Optional[str] = None, reverse: bool = False) -> List[Dict[str, Any]]:
        """
        Advanced, high-performance querying tool to isolate models using full wildcard searching,
        explicit multi-dimensional filters, metric evaluations, and ordered rank sorting execution.
        """
        with self._lock:
            self._stats["searches"] += 1
            
            # Generate deterministic cache identity string signature key
            cache_key = f"q:{query_string};f:{json.dumps(filters, sort_keys=True) if filters else ''};s:{sort_by};r:{reverse}"
            if cache_key in self._cache:
                self._stats["cache_hits"] += 1
                return copy.deepcopy(self._cache[cache_key])
                
            candidates = list(self._models.values())
            
            # Textual matching phase via pattern search execution
            if query_string and query_string != "*":
                pattern = query_string.lower()
                matched = []
                for m in candidates:
                    if (fnmatch.fnmatch(m.id.lower(), pattern) or
                        fnmatch.fnmatch(m.name.lower(), pattern) or
                        fnmatch.fnmatch(m.provider.lower(), pattern) or
                        fnmatch.fnmatch(m.family.lower(), pattern) or
                        any(fnmatch.fnmatch(t.lower(), pattern) for t in m.tags) or
                        any(fnmatch.fnmatch(c.lower(), pattern) for c in m.capabilities)):
                        matched.append(m)
                candidates = matched
                
            # Multi-dimensional operational filter criteria execution mapping phase
            if filters:
                candidates = [m for m in candidates if self._matches_filters(m, filters)]
                
            # Structural rank sequencing sorting execution phase
            if sort_by:
                candidates = self._sort_candidates(candidates, sort_by, reverse)
            elif reverse:
                candidates.reverse()
                
            result = [m.to_dict() for m in candidates]
            self._cache[cache_key] = result
            return copy.deepcopy(result)
            
    # --- Data Backup, Interchange, Snapshot states ---
    
    def export_registry(self) -> str:
        """Exports the active state of all models and configurations as a JSON string."""
        with self._lock:
            self._stats["exports"] += 1
            payload = {
                "models": [m.to_dict() for m in self._models.values()],
                "aliases": self._aliases.copy(),
                "exportedAt": datetime.now(timezone.utc).isoformat()
            }
            return json.dumps(payload, indent=2)
            
    def import_registry(self, json_data: str, clear_existing: bool = False) -> None:
        """Parses and restores active registry mapping layers from a JSON data payload."""
        with self._lock:
            try:
                payload = json.loads(json_data)
                if "models" not in payload or "aliases" not in payload:
                    raise ModelRegistryError("Invalid structural import profile schemas passed.")
                    
                if clear_existing:
                    self.clear()
                    
                for m_dict in payload["models"]:
                    # Handle conflicting imports using updating strategies safely
                    if m_dict["id"] in self._models:
                        self.update_model(m_dict["id"], m_dict)
                    else:
                        self.register_model(m_dict)
                        
                for alias, target in payload["aliases"].items():
                    if target in self._models:
                        self._aliases[alias] = target
                        
                self._stats["imports"] += 1
                self._log_audit("IMPORT", {"count": len(payload["models"])})
            except Exception as e:
                if not isinstance(e, ModelRegistryError):
                    raise ModelRegistryError(f"Import process validation processing error failure: {e}") from e
                raise
                
    def backup(self) -> bytes:
        """Generates a compressed system binary image file backup state conversion."""
        with self._lock:
            self._stats["backups"] += 1
            self._log_audit("BACKUP", {})
            return self.export_registry().encode("utf-8")
            
    def restore(self, backup_bytes: bytes) -> None:
        """Restores a registry backup state from raw bytes data serialization."""
        with self._lock:
            self._stats["restores"] += 1
            self._log_audit("RESTORE", {})
            self.import_registry(backup_bytes.decode("utf-8"), clear_existing=True)
            
    def create_snapshot(self, snapshot_id: str) -> None:
        """Captures internal tracking state directly onto isolated operational memories."""
        with self._lock:
            self._snapshots[snapshot_id] = {
                "models": copy.deepcopy(self._models),
                "aliases": self._aliases.copy(),
                "timestamp": time.time()
            }
            self._log_audit("SNAPSHOT_CREATE", {"snapshot_id": snapshot_id})
            
    def restore_snapshot(self, snapshot_id: str) -> None:
        """Restores internal data states directly from isolated snapshots."""
        with self._lock:
            if snapshot_id not in self._snapshots:
                raise ModelRegistryError(f"Target memory snapshot instance footprint ID '{snapshot_id}' tracking not found.")
            snap = self._snapshots[snapshot_id]
            self._models = copy.deepcopy(snap["models"])
            self._aliases = snap["aliases"].copy()
            self._rebuild_all_indices()
            self._clear_cache()
            self._log_audit("SNAPSHOT_RESTORE", {"snapshot_id": snapshot_id})
            
    # --- Runtime Diagnostics & Orchestrated Tasks ---
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Provides an operational health diagnostic footprint summary report."""
        with self._lock:
            uptime = time.time() - self._start_time
            return {
                "uptime_seconds": uptime,
                "registry_health": "GREEN" if len(self._models) > 0 else "AMBER",
                "registered_model_count": len(self._models),
                "providers": self.list_providers(),
                "families": self.list_families(),
                "capabilities_indexed_count": len(self._index_capability),
                "aliases_count": len(self._aliases),
                "statistics": self._stats.copy(),
                "cache_status": {"cached_queries_count": len(self._cache)},
                "memory_estimates": {
                    "audit_trail_length": len(self._audit_trail),
                    "snapshots_stored": list(self._snapshots.keys())
                }
            }
            
    def start_background_cleanup(self, interval_seconds: float = 3600.0) -> None:
        """Spawns an asynchronous background worker monitoring registry tracking loops."""
        with self._lock:
            if self._cleanup_running:
                return
            self._cleanup_running = True
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_loop_worker(interval_seconds))
            except RuntimeError:
                # Execution environment runtime safety context block
                pass
                
    def shutdown(self) -> None:
        """Halts worker threads, loops, and flushes runtime references cleanly."""
        with self._lock:
            self._cleanup_running = False
            if self._cleanup_task:
                self._cleanup_task.cancel()
                self._cleanup_task = None
            self.clear()
            
    # --- Engine Filtering and Utility Helpers ---
    
    def _resolve_id_internal(self, identifier: str) -> str:
        if identifier in self._models:
            return identifier
        if identifier in self._aliases:
            self._stats["alias_usage"] += 1
            return self._aliases[identifier]
        raise ModelRegistryError(f"Target tracker mapping could not resolve key references identifier: '{identifier}'.")
        
    def _matches_filters(self, model: ModelDefinition, filters: Dict[str, Any]) -> bool:
        # Evaluate standard filter criteria
        if "provider" in filters and model.provider.lower() != filters["provider"].lower():
            return False
        if "family" in filters and model.family.lower() != filters["family"].lower():
            return False
        if "deprecated" in filters and model.deprecated != filters["deprecated"]:
            return False
        if "experimental" in filters and model.experimental != filters["experimental"]:
            return False
        if "availability" in filters and model.availability.lower() != filters["availability"].lower():
            return False
            
        # Capability dynamic checking constraints
        for functional_cap in ["streaming", "vision", "function_calling", "embeddings", "reasoning"]:
            if filters.get(functional_cap) is True and functional_cap not in model.capabilities:
                return False
                
        # Custom explicit capability match criteria list checks
        if "capability" in filters and filters["capability"] not in model.capabilities:
            return False
            
        # Threshold capabilities metric scale range evaluations
        if "min_context_window" in filters and model.contextWindow < filters["min_context_window"]:
            return False
        if "min_quality_score" in filters and model.qualityScore < filters["min_quality_score"]:
            return False
            
        # Predicate structural parsing execution overrides
        if "predicate" in filters and callable(filters["predicate"]):
            try:
                if not filters["predicate"](model):
                    return False
            except Exception as ex:
                logger.error(f"Predicate filter parsing validation error execution constraint failure: {ex}")
                return False
                
        return True
        
    def _sort_candidates(self, candidates: List[ModelDefinition], sort_by: str, reverse: bool) -> List[ModelDefinition]:
        sort_selectors: Dict[str, Callable[[ModelDefinition], Any]] = {
            "quality": lambda m: m.qualityScore,
            "speed": lambda m: m.speedScore,
            "reliability": lambda m: m.reliabilityScore,
            "popularity": lambda m: m.popularityScore,
            "context_window": lambda m: m.contextWindow,
            "priority": lambda m: m.priority,
            "cost": lambda m: m.pricing.get("input_1k", 0.0) + m.pricing.get("output_1k", 0.0),
            "provider": lambda m: m.provider.lower(),
            "alphabetical": lambda m: m.name.lower()
        }
        selector = sort_selectors.get(sort_by.lower(), lambda m: m.id)
        # Numerical metrics sort largest to smallest by default, hence `not reverse` toggle
        descending_metrics = ["quality", "speed", "reliability", "popularity", "context_window", "priority"]
        actual_reverse = not reverse if sort_by in descending_metrics else reverse
        return sorted(candidates, key=selector, reverse=actual_reverse)
        
    def _rebuild_indices_for_model(self, model: ModelDefinition) -> None:
        m_id = model.id
        self._index_provider.setdefault(model.provider.lower(), set()).add(m_id)
        self._index_family.setdefault(model.family.lower(), set()).add(m_id)
        for cap in model.capabilities:
            self._index_capability.setdefault(cap.lower(), set()).add(m_id)
        for tag in model.tags:
            self._index_tag.setdefault(tag.lower(), set()).add(m_id)
            
    def _remove_indices_for_model(self, model: ModelDefinition) -> None:
        m_id = model.id
        self._index_provider.get(model.provider.lower(), set()).discard(m_id)
        self._index_family.get(model.family.lower(), set()).discard(m_id)
        for cap in model.capabilities:
            self._index_capability.get(cap.lower(), set()).discard(m_id)
        for tag in model.tags:
            self._index_tag.get(tag.lower(), set()).discard(m_id)
            
    def _rebuild_all_indices(self) -> None:
        self._index_provider.clear()
        self._index_family.clear()
        self._index_capability.clear()
        self._index_tag.clear()
        for model in self._models.values():
            self._rebuild_indices_for_model(model)
            
    def _clear_cache(self) -> None:
        self._cache.clear()
        
    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        self._audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details
        })
        if len(self._audit_trail) > 5000:
            self._audit_trail = self._audit_trail[-2500:]
            
    async def _cleanup_loop_worker(self, interval_seconds: float) -> None:
        while self._cleanup_running:
            try:
                await asyncio.sleep(interval_seconds)
                with self._lock:
                    # Automatically remove expired snapshots (older than 7 days)
                    now = time.time()
                    expired_snapshots = [k for k, v in self._snapshots.items() if now - v["timestamp"] > 86400 * 7]
                    for snap_id in expired_snapshots:
                        del self._snapshots[snap_id]
                    self._clear_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error encountered during registry optimization processing thread cycles: {e}", exc_info=True)
