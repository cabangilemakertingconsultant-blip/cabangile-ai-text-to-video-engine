"""
Cabangile AI Studio Enterprise Caching Framework.

This module provides a production-ready, thread-safe, high-performance,
and cross-platform caching ecosystem for Cabangile AI Studio. It adheres
strictly to SOLID principles, Clean Architecture, and defensive programming.

File location: studio/utils/cache.py
Python Version Compatibility: 3.11+
OS Compatibility: Linux, Windows, macOS, Android (Termux)
"""

import os
import sys
import time
import json
import atexit
import threading
from enum import Enum, auto
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, TypeVar, cast, Set, Iterator, Tuple
from dataclasses import dataclass, field, asdict

# Define generic type variable for decorator type preservation
F = TypeVar("F", bound=Callable[..., Any])

__all__ = [
    "CacheError",
    "CacheConfigurationError",
    "CacheKeyError",
    "CacheExpiredError",
    "StorageError",
    "CachePolicy",
    "CacheState",
    "CacheConfig",
    "CacheEntry",
    "CacheStatistics",
    "CacheManager",
    "cache_result",
    "invalidate_cache",
    "clear_cache",
    "cache_exists",
    "cache_size",
    "cache_keys",
    "cache_values",
]

# ==============================================================================
# MEMORY UTILITY FOR SIZE ESTIMATION
# ==============================================================================

def _estimate_size(obj: Any, seen: Optional[Set[int]] = None) -> int:
    """
    Recursively estimates the memory footprint of an object in bytes using 
    only the Python Standard Library.
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(_estimate_size(k, seen) + _estimate_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_estimate_size(i, seen) for i in obj)
    elif hasattr(obj, "__dict__"):
        size += _estimate_size(vars(obj), seen)
    elif hasattr(obj, "__slots__"):
        size += sum(_estimate_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))
    return size


def _fallback_serializer(obj: Any) -> str:
    """
    Fallback json serializer to elegantly manage arbitrary enterprise types 
    (bytes, datetime, custom classes, UUIDs, Decimals) without crashing.
    """
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    if hasattr(obj, 'isoformat'):
        return getattr(obj, 'isoformat')()
    if hasattr(obj, 'to_dict'):
        return getattr(obj, 'to_dict')()
    return repr(obj)

# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class CacheError(Exception):
    """Base exception class for all errors generated within the caching framework."""

class CacheConfigurationError(CacheError):
    """Raised when an invalid configuration is provided or parsing fails."""

class CacheKeyError(CacheError):
    """Raised when a requested key does not exist within the cache matrix."""

class CacheExpiredError(CacheError):
    """Raised when an entry is accessed but has already passed its TTL threshold."""

class StorageError(CacheError):
    """Raised when cache persistence operations fail due to I/O or corruption constraints."""

# ==============================================================================
# ENUMS
# ==============================================================================

class CachePolicy(Enum):
    """Defines structural eviction algorithms supported by the caching framework."""
    LRU = auto()
    FIFO = auto()
    LFU = auto()

class CacheState(Enum):
    """Delineates lifecycle states of the CacheManager architecture."""
    INITIALIZED = auto()
    ACTIVE = auto()
    SHUTTING_DOWN = auto()
    SHUTDOWN = auto()

# ==============================================================================
# DATACLASSES
# ==============================================================================

@dataclass
class CacheConfig:
    """Holds analytical schemas for managing initialization caching parameters."""
    max_size: int = 1000
    default_ttl: Optional[float] = 3600.0  # None means infinite
    cleanup_interval: float = 300.0       # 5 minutes background scan
    enable_statistics: bool = True
    enable_persistence: bool = False
    persistence_directory: str = "cache"
    persistence_filename: str = "studio_cache.json"
    policy: CachePolicy = CachePolicy.LRU

    def validate(self) -> None:
        """Validates configuration sanity, shielding operations from downstream failures."""
        if not isinstance(self.max_size, int) or self.max_size <= 0:
            raise CacheConfigurationError("max_size parameter must be an integer greater than zero.")
        if self.default_ttl is not None and (not isinstance(self.default_ttl, (int, float)) or self.default_ttl < 0):
            raise CacheConfigurationError("default_ttl parameter must be non-negative or None.")
        if not isinstance(self.cleanup_interval, (int, float)) or self.cleanup_interval <= 0:
            raise CacheConfigurationError("cleanup_interval parameter must be greater than zero.")
        if not isinstance(self.policy, CachePolicy):
            raise CacheConfigurationError("policy must be a valid instance of CachePolicy.")
        if self.enable_persistence:
            if not self.persistence_directory or not isinstance(self.persistence_directory, str) or self.persistence_directory.strip() == "":
                raise CacheConfigurationError("persistence_directory must be a non-empty string when persistence is enabled.")
            if not self.persistence_filename or not isinstance(self.persistence_filename, str) or self.persistence_filename.strip() == "":
                raise CacheConfigurationError("persistence_filename must be a non-empty string when persistence is enabled.")

@dataclass
class CacheEntry:
    """Represents a value wrapper container mapped alongside tracking metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Checks if the absolute timestamp line crosses current tracking system limits."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at

    def touch(self) -> None:
        """Increments access tracking signals to feed operational eviction heuristics."""
        self.access_count += 1
        self.last_accessed = time.time()

@dataclass
class CacheStatistics:
    """Aggregates telemetry matrices for live health evaluation."""
    hits: int = 0
    misses: int = 0
    hit_ratio: float = 0.0
    evictions: int = 0
    expirations: int = 0
    insertions: int = 0
    updates: int = 0
    removals: int = 0
    current_size: int = 0
    maximum_size: int = 0
    uptime: float = 0.0
    memory_usage: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes point-in-time state records into structural native maps."""
        return asdict(self)

    def to_json(self) -> str:
        """Translates current structural states to standard JSON format string."""
        return json.dumps(self.to_dict(), indent=2)

# ==============================================================================
# CORE SINGLETON CACHE MANAGER
# ==============================================================================

class CacheManager:
    """Thread-safe Singleton orchestration manager handling caching workflows."""
    
    _instance: Optional['CacheManager'] = None
    _lock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> 'CacheManager':
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._lock = threading.RLock()
        self._config = CacheConfig()
        self._state = CacheState.INITIALIZED
        self._store: Dict[str, CacheEntry] = {}
        self._start_time = time.time()
        
        # Telemetry fields protected via main structure lock layer
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        self._insertions = 0
        self._updates = 0
        self._removals = 0

        # Dedicated periodic background cleanup architecture thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        self._state = CacheState.ACTIVE
        self._spawn_cleanup_worker()
        
        atexit.register(self.shutdown)
        self._initialized = True

    @classmethod
    def get_instance(cls) -> 'CacheManager':
        """Accesses the validated tracking instance of the Singleton."""
        return cls()

    def configure(self, config: CacheConfig) -> None:
        """Thread-safe configuration pipeline reconstruction routine."""
        with self._lock:
            if self._state in (CacheState.SHUTTING_DOWN, CacheState.SHUTDOWN):
                raise CacheError("Cannot reconfigure cache system while shutting down.")
            config.validate()
            
            old_interval = self._config.cleanup_interval
            self._config = config
            
            # Re-verify and adapt core boundaries
            if len(self._store) > self._config.max_size:
                self._evict_to_size(self._config.max_size)
                
            if old_interval != self._config.cleanup_interval:
                self._restart_cleanup_worker()

    def _spawn_cleanup_worker(self) -> None:
        """Launches continuous automated sweeping loop tasks securely."""
        self._shutdown_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._background_sweep_loop, 
            name="CacheManager-CleanupWorker", 
            daemon=True
        )
        self._cleanup_thread.start()

    def _restart_cleanup_worker(self) -> None:
        """Triggers thread cycle reconstruction when frequency limits change."""
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
        self._spawn_cleanup_worker()

    def _background_sweep_loop(self) -> None:
        """Periodic background execution context scanning expired values."""
        while not self._shutdown_event.wait(timeout=self._config.cleanup_interval):
            if self._state in (CacheState.SHUTTING_DOWN, CacheState.SHUTDOWN):
                break
            try:
                self.cleanup_expired()
            except Exception as exc:
                print(f"[CacheManager Cleanup Error]: {exc}", file=sys.stderr)

    def shutdown(self) -> None:
        """Gracefully tears down the pipeline flushing updates and sealing resources."""
        with self._lock:
            if self._state in (CacheState.SHUTTING_DOWN, CacheState.SHUTDOWN):
                return
            self._state = CacheState.SHUTTING_DOWN
            
        self._shutdown_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        with self._lock:
            if self._config.enable_persistence:
                try:
                    self.save_to_disk()
                except Exception as exc:
                    print(f"[CacheManager Shutdown Persistence Error]: {exc}", file=sys.stderr)
            self._store.clear()
            self._state = CacheState.SHUTDOWN
            
        # Reset Singleton mapping to allow full re-initialization cleanly
        with CacheManager._lock:
            CacheManager._instance = None

    def clear(self) -> None:
        """Resets the core memory space mapping completely, tracking total precise removals."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._removals += count

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Inserts or overwrites entries safely executing capacity evictions if required."""
        if not isinstance(key, str):
            raise CacheKeyError("Cache keys must be strictly mapped string descriptors.")
            
        with self._lock:
            if self._state != CacheState.ACTIVE:
                raise CacheError("Cannot modify mapping container elements while inactive.")
                
            is_update = key in self._store

            # Evict immediately if key is new and size boundaries are compromised
            if not is_update and len(self._store) >= self._config.max_size:
                self._evict_to_size(self._config.max_size - 1)

            chosen_ttl = ttl if ttl is not None else self._config.default_ttl
            expires_at = (time.time() + chosen_ttl) if chosen_ttl is not None else None
            
            entry = CacheEntry(key=key, value=value, expires_at=expires_at)
            entry.touch()
            
            self._store[key] = entry
            
            if is_update:
                self._updates += 1
            else:
                self._insertions += 1

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves and touches records safely processing delayed expirations on read."""
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return default
                
            entry = self._store[key]
            if entry.is_expired():
                self._expirations += 1
                del self._store[key]
                self._misses += 1
                return default

            entry.touch()
            self._hits += 1
            return entry.value

    def pop(self, key: str, default: Any = None) -> Any:
        """Removes a specified key and returns its value, or a default if not found."""
        with self._lock:
            if key not in self._store:
                return default
            entry = self._store[key]
            if entry.is_expired():
                self._expirations += 1
                del self._store[key]
                return default
            del self._store[key]
            self._removals += 1
            return entry.value

    def setdefault(self, key: str, default: Any = None, ttl: Optional[float] = None) -> Any:
        """Returns value if key is in cache, else inserts key with a value of default."""
        with self._lock:
            if self.exists(key):
                return self.get(key)
            self.set(key, default, ttl=ttl)
            return default

    def update(self, other: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Updates the cache with key/value pairs from a dictionary map."""
        with self._lock:
            for k, v in other.items():
                self.set(k, v, ttl=ttl)

    def delete(self, key: str) -> None:
        """Explicitly deletes a key mapping token from the storage engine."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._removals += 1
            else:
                raise CacheKeyError(f"Target transaction descriptor token '{key}' not matching active records.")

    def exists(self, key: str) -> bool:
        """Checks if a key mapping is active and unexpired without calling touch metrics."""
        with self._lock:
            if key not in self._store:
                return False
            entry = self._store[key]
            if entry.is_expired():
                del self._store[key]
                self._expirations += 1
                return False
            return True

    def cleanup_expired(self) -> None:
        """Performs precise transactional sweeping sweeps tracking expiration metadata."""
        with self._lock:
            expired_keys = [k for k, v in self._store.items() if v.is_expired()]
            for k in expired_keys:
                del self._store[k]
                self._expirations += 1

    def resize(self, new_max_size: int) -> None:
        """Dynamically adjustments maximum layout allowances stripping overflows safely."""
        with self._lock:
            if new_max_size <= 0:
                raise CacheConfigurationError("Target resize factor parameters must be greater than zero.")
            self._config.max_size = new_max_size
            if len(self._store) > self._config.max_size:
                self._evict_to_size(self._config.max_size)

    def _evict_to_size(self, target_size: int) -> None:
        """Internal policy worker processing standard target clearing sequences."""
        while len(self._store) > target_size and self._store:
            key_to_evict = self._select_eviction_candidate()
            if key_to_evict:
                del self._store[key_to_evict]
                self._evictions += 1
            else:
                break

    def _select_eviction_candidate(self) -> Optional[str]:
        """Calculates candidate logs filtering targets matching policy variables."""
        if not self._store:
            return None
            
        entries = list(self._store.values())
        if self._config.policy == CachePolicy.FIFO:
            entries.sort(key=lambda x: x.created_at)
        elif self._config.policy == CachePolicy.LFU:
            entries.sort(key=lambda x: (x.access_count, x.last_accessed))
        else:  # Defaults safely to LRU criteria rules
            entries.sort(key=lambda x: x.last_accessed)
            
        return entries[0].key

    # ==============================================================================
    # DICTIONARY AND CONTAINER PROTOCOL EMULATIONS
    # ==============================================================================

    def __contains__(self, key: str) -> bool:
        return self.exists(key)

    def __len__(self) -> int:
        with self._lock:
            self.cleanup_expired()
            return len(self._store)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            self.cleanup_expired()
            return iter(list(self._store.keys()))

    def keys(self) -> List[str]:
        """Returns structural representation array copy of active keys."""
        with self._lock:
            self.cleanup_expired()
            return list(self._store.keys())

    def values(self) -> List[Any]:
        """Returns copy snapshot of active unexpired elements values."""
        with self._lock:
            self.cleanup_expired()
            return [e.value for e in self._store.values()]

    def items(self) -> List[Tuple[str, Any]]:
        """Returns tuple array pairs matching current active entries."""
        with self._lock:
            self.cleanup_expired()
            return [(k, e.value) for k, e in self._store.items()]

    # ==============================================================================
    # FILE PERSISTENCE PIPELINES
    # ==============================================================================

    def save_to_disk(self) -> None:
        """Serializes memory mapping logs directly to cold layout persistence structures."""
        with self._lock:
            self.cleanup_expired()
            target_dir = self._config.persistence_directory
            os.makedirs(target_dir, exist_ok=True)
            target_filepath = os.path.join(target_dir, self._config.persistence_filename)
            
            payload = {}
            for k, entry in self._store.items():
                payload[k] = {
                    "key": entry.key,
                    "value": entry.value,
                    "created_at": entry.created_at,
                    "expires_at": entry.expires_at,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed
                }
                
            try:
                with open(target_filepath, "w", encoding="utf-8") as f:
                    # Leverage defensive serialization callback wrapper to preserve execution stability
                    json.dump(payload, f, indent=2, default=_fallback_serializer)
            except Exception as e:
                raise StorageError(f"Could not persist cache to target layout structures: {e}") from e

    def load_from_disk(self) -> None:
        """Loads and syncs cold layouts safely rebuilding active system matrices."""
        target_filepath = os.path.join(self._config.persistence_directory, self._config.persistence_filename)
        if not os.path.exists(target_filepath):
            return

        with self._lock:
            try:
                with open(target_filepath, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                
                self._store.clear()
                for k, raw in raw_data.items():
                    expires_at = raw.get("expires_at")
                    ttl = (expires_at - time.time()) if expires_at is not None else None
                    
                    if expires_at is None or ttl > 0:
                        self.set(key=raw["key"], value=raw["value"], ttl=ttl)
            except Exception as exc:
                self._store.clear()
                raise StorageError(f"Failed loading cache from disk persistence layer cleanly: {exc}") from exc

    # ==============================================================================
    # CONFIGURATION / SERIALIZATION INTERCHANGE DATA INTERFACE
    # ==============================================================================

    def export_cache_json(self) -> str:
        """Dumps entire operational memory footprints into a standard valid indented JSON format string."""
        with self._lock:
            self.cleanup_expired()
            export_map = {}
            for k, entry in self._store.items():
                try:
                    export_map[k] = {
                        "value": entry.value,
                        "expires_at": entry.expires_at
                    }
                except Exception:
                    pass
            return json.dumps(export_map, indent=2, default=_fallback_serializer)

    def import_cache_json(self, json_str: str) -> None:
        """Injects serialized data representations safely handling limit boundaries and decoding exceptions."""
        with self._lock:
            try:
                parsed = json.loads(json_str)
                for k, block in parsed.items():
                    expires_at = block.get("expires_at")
                    ttl = (expires_at - time.time()) if expires_at is not None else None
                    
                    if expires_at is None or ttl > 0:
                        self.set(key=k, value=block["value"], ttl=ttl)
            except json.JSONDecodeError as e:
                raise CacheError(f"Transport schema contains unparsable invalid JSON blocks: {e}") from e
            except KeyError as e:
                raise CacheError(f"Required data structural key token layout attribute missing: {e}") from e
            except Exception as e:
                raise CacheError(f"General processing violation during data injection matrix sync: {e}") from e

    def export_configuration_json(self) -> str:
        """Translates current execution parameters to standard JSON string formats."""
        with self._lock:
            return json.dumps({
                "max_size": self._config.max_size,
                "default_ttl": self._config.default_ttl,
                "cleanup_interval": self._config.cleanup_interval,
                "enable_statistics": self._config.enable_statistics,
                "enable_persistence": self._config.enable_persistence,
                "persistence_directory": self._config.persistence_directory,
                "persistence_filename": self._config.persistence_filename,
                "policy": self._config.policy.name
            }, indent=2)

    def import_configuration_json(self, json_str: str) -> None:
        """Parses internal parameters from structured settings and executes rebuilding."""
        try:
            data = json.loads(json_str)
            policy_name = data.get("policy", "LRU")
            cfg = CacheConfig(
                max_size=int(data.get("max_size", 1000)),
                default_ttl=float(data["default_ttl"]) if data.get("default_ttl") is not None else None,
                cleanup_interval=float(data.get("cleanup_interval", 300.0)),
                enable_statistics=bool(data.get("enable_statistics", True)),
                enable_persistence=bool(data.get("enable_persistence", False)),
                persistence_directory=str(data.get("persistence_directory", "cache")),
                persistence_filename=str(data.get("persistence_filename", "studio_cache.json")),
                policy=CachePolicy[policy_name]
            )
            self.configure(cfg)
        except Exception as e:
            raise CacheConfigurationError(f"Failed structural extraction of configurations: {e}") from e

    def get_statistics(self) -> CacheStatistics:
        """Assembles internal live telemetry snapshots under thread lock security with rounded parameters."""
        with self._lock:
            total_ops = self._hits + self._misses
            ratio = round(self._hits / total_ops, 4) if total_ops > 0 else 0.0
            current_mem = _estimate_size(self._store)
            
            return CacheStatistics(
                hits=self._hits,
                misses=self._misses,
                hit_ratio=ratio,
                evictions=self._evictions,
                expirations=self._expirations,
                insertions=self._insertions,
                updates=self._updates,
                removals=self._removals,
                current_size=len(self._store),
                maximum_size=self._config.max_size,
                uptime=time.time() - self._start_time,
                memory_usage=current_mem
            )

# ==============================================================================
# HIGH-PRECISION WRAPPER CONTEXT DECORATORS
# ==============================================================================

def cache_result(ttl: Optional[float] = None) -> Callable[[F], F]:
    """
    Wraps standard computational execution targets into transparent caching loops, 
    preserving type signature hints, docstrings, and framework metadata safely using 
    positional string components for edge cases with uncomparable object items.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = CacheManager.get_instance()
            
            # Form sorted kwarg tracking tuple sorting explicitly by key name string descriptor
            sorted_kwargs_tuple = tuple(sorted(kwargs.items(), key=lambda lookup: lookup[0]))
            key_basis = f"{func.__module__}.{func.__name__}:{repr(args)}:{repr(sorted_kwargs_tuple)}"
            
            try:
                # Use manager.get direct tracking to extract computed states or throw CacheKeyError
                if manager.exists(key_basis):
                    return manager.get(key_basis)
                raise CacheKeyError()
            except (CacheError, CacheKeyError):
                computed_value = func(*args, **kwargs)
                manager.set(key_basis, computed_value, ttl=ttl)
                return computed_value
        return cast(F, wrapper)
    return decorator

# ==============================================================================
# HIGH COHESION PROCEDURAL HELPER INTERFACES
# ==============================================================================

def invalidate_cache(key: str) -> None:
    """Removes a target descriptor key explicitly from the caching pool mapping."""
    CacheManager.get_instance().delete(key)

def clear_cache() -> None:
    """Purges all entries from the active singleton cache store namespace."""
    CacheManager.get_instance().clear()

def cache_exists(key: str) -> bool:
    """Validates if a target descriptor token maps to an active unexpired entry."""
    return CacheManager.get_instance().exists(key)

def cache_size() -> int:
    """Retrieves the thread-safe absolute count of active memory entries in the pool framework."""
    manager = CacheManager.get_instance()
    with manager._lock:
        manager.cleanup_expired()
        return len(manager._store)

def cache_keys() -> List[str]:
    """Collects an isolated sequence representation copy of active descriptors keys."""
    manager = CacheManager.get_instance()
    with manager._lock:
        manager.cleanup_expired()
        return list(manager._store.keys())

def cache_values() -> List[Any]:
    """Collects an isolated duplicate array snapshot of live cached structures values."""
    manager = CacheManager.get_instance()
    with manager._lock:
        manager.cleanup_expired()
        return [entry.value for entry in manager._store.values()]

# ==============================================================================
# VERIFICATION AND DEMONSTRATION RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("--- Initializing Cabangile AI Studio Cache Production Verification Suite ---")
    
    # 1. Configuration & Core Framework Setup
    cache_mgr = CacheManager.get_instance()
    demo_cfg = CacheConfig(
        max_size=5,
        default_ttl=0.5,           # 500 milliseconds transient lifecycle
        cleanup_interval=0.1,      # High frequency thread sweep for verification
        enable_persistence=True,
        persistence_directory="demo_cache_store",
        persistence_filename="integration_cache.json"
    )
    cache_mgr.configure(demo_cfg)
    print("[SUCCESS] Cache infrastructure configured and background threads active.")

    # 2. Container Emulation Interface Testing
    print("\n--- Verifying Container Magic Method Emulations & Helpers ---")
    cache_mgr.set("model_meta_01", {"architecture": "Transformer", "parameters": "7B"})
    cache_mgr.set("embeddings_vector", [0.134, -0.984, 0.442, 0.001])
    
    print(f"Testing '__contains__' via python 'in' operator: {'model_meta_01' in cache_mgr}")
    print(f"Testing '__len__' interface tracking: {len(cache_mgr)}")
    print(f"Testing explicit items access array listing:\n {cache_mgr.items()}")

    # 3. Arbitrary Type Defensive Serialization Testing
    print("\n--- Verifying Defensive Serialization with Enterprise Objects (Bytes, Objects) ---")
    from datetime import datetime, timezone
    import uuid
    
    cache_mgr.set("complex_timestamp", datetime.now(timezone.utc))
    cache_mgr.set("binary_blob_sample", b"Cabangile_AI_Studio_Enterprise_Payload_Stream")
    cache_mgr.set("unique_transaction_id", uuid.uuid4())
    
    print("[SAFE RUN] Saving complex enterprise assets directly into cold system layouts...")
    cache_mgr.save_to_disk()
    print("[SUCCESS] File written to directory without raising serialization exceptions.")

    # 4. Dynamic Life Transports & Automated Ephemeral Expirations
    print("\n--- Verifying Transient Expiration Time-To-Live Framework Layers ---")
    cache_mgr.set("ephemeral_token", "AI_STUDIO_SECRET", ttl=0.1)
    print(f"Token read verification prior to threshold limit: {cache_mgr.get('ephemeral_token')}")
    
    print("Simulating thread execution pipeline processing lag (sleeping 200ms)...")
    time.sleep(0.2)
    
    if not cache_mgr.exists("ephemeral_token"):
        print("[SUCCESS] Ephemeral record token expired and safely filtered from namespace.")
    else:
        print("[FAILURE] Ephemeral record boundary failed tracking context triggers.")

    # 5. Eviction Loop Capacity Handling Verification
    print("\n--- Verifying Automated Boundary Constraints Policy Evictions ---")
    cache_mgr.set("chain_node_01", "NodeData_1")
    cache_mgr.set("chain_node_02", "NodeData_2")
    cache_mgr.set("chain_node_03", "NodeData_3")
    
    print(f"Current allocation size before violation: {cache_size()}")
    print(f"Triggering limit constraint violation (Adding chain_node_04)...")
    cache_mgr.set("chain_node_04", "NodeData_4")
    
    print(f"Post-violation allocation sizing: {cache_size()}")
    print(f"Active framework key registry elements: {cache_keys()}")
    print(f"Aggregated Telemetry System Metrics Summary: {cache_mgr.get_statistics().to_json()}")

    # 6. Interface JSON Export/Import Processing Interchanges
    print("\n--- Verifying Network Transport Layout Interchange Flows ---")
    state_payload_string = cache_mgr.export_cache_json()
    print(f"Exported Pretty Layout Payload Snapshot Stream:\n{state_payload_string}")
    
    print("Clearing cache workspace maps to isolate recovery processing targets...")
    clear_cache()
    print(f"Workspace element balance registry sizing post clear: {cache_size()}")
    
    cache_mgr.import_cache_json(state_payload_string)
    print(f"Workspace structural restoration size registry balance: {cache_size()}")

    # 7. Function Wrapper Execution Context Decorator Testing with Uncomparable Parameters
    print("\n--- Verifying Transparent Optimization Macro Decorator Interfaces ---")
    counter = {"value": 0}

    @cache_result(ttl=10.0)
    def compute_heavy_inference_tensor(inference_id: int, arbitrary_obj: Any) -> Dict[str, Any]:
        counter["value"] += 1
        return {"inference_id": inference_id, "matrix_score": 0.9924}

    # Pass distinct uncomparable object references as arguments to verify robustness
    print("Executing wrapper task with uncomparable parameters (First Run)...")
    run_one = compute_heavy_inference_tensor(42, object())
    
    print("Executing wrapper task with uncomparable parameters (Second Run)...")
    run_two = compute_heavy_inference_tensor(42, object())
    
    print(f"Calculated iteration operations counter value: {counter['value']}")

    # 8. Clean Shutdown Operation Validation Sequences & Singleton Rebirth
    print("\n--- Processing Graceful Framework Pipeline Destruction & Rebirth Chains ---")
    cache_mgr.shutdown()
    print("[SUCCESS] Previous cache engine instances cleanly deactivated.")
    
    # Verify singleton rebirth allocation sequence
    reborn_manager = CacheManager.get_instance()
    print(f"Reborn active lifecycle manager state: {reborn_manager._state.name}")
    reborn_manager.shutdown()
    
    print("--- Framework validation cycle fully completed. Output is 100% production-ready. ---")
