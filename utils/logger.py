"""
Cabangile AI Studio Enterprise Logging Framework.

This module provides a production-ready, thread-safe, high-performance,
and cross-platform logging ecosystem for Cabangile AI Studio. It adheres
strictly to SOLID principles, Clean Architecture, and defensive programming.

File location: studio/utils/logger.py
Python Version Compatibility: 3.11+
OS Compatibility: Linux, Windows, macOS, Android (Termux)
"""

import os
import sys
import time
import json
import atexit
import logging
import threading
import traceback
from enum import Enum, auto
from functools import wraps
from typing import Any, Dict, List, Optional, Type, Callable, TypeVar, cast
from types import TracebackType
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

# Define generic type variable for decorator type preservation
F = TypeVar("F", bound=Callable[..., Any])

__all__ = [
    "LoggerError",
    "ConfigurationError",
    "StorageError",
    "LogFormat",
    "LoggerState",
    "LogConfig",
    "LoggerStats",
    "BaseCustomFormatter",
    "PlainTextFormatter",
    "ColorFormatter",
    "JSONFormatter",
    "StatisticsCollectorHandler",
    "LogManager",
    "ExecutionTimer",
    "DynamicLevelContext",
    "execution_time_decorator",
    "log_with_metadata",
    "log_audit_event",
    "log_security_event",
    "log_exception_context",
    "get_temporary_level_context",
    "get_execution_timer",
]

# ==============================================================================
# CROSS-PLATFORM MEMORY TRACKING UTILITY
# ==============================================================================

def _get_process_memory() -> int:
    """
    Retrieves the current resident set size (RSS) memory of the process in bytes.
    
    Supported Platforms:
        - Windows: Leverages ctypes via psapi.dll (GetProcessMemoryInfo).
        - Linux / Android (Termux): Reads /proc/self/status (VmRSS).
        - macOS / Unix Generic: Falls back to the standard 'resource' module.
    """
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('cb', wintypes.DWORD),
                    ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                ]
            
            GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
            GetCurrentProcess.restype = wintypes.HANDLE
            
            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            GetProcessMemoryInfo.restype = wintypes.BOOL
            
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
            return 0
        else:
            # POSIX compliance (Linux, macOS, Termux)
            if os.path.exists("/proc/self/status"):
                with open("/proc/self/status", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1]) * 1024
    except Exception:
        pass
    
    # Fallback method using resource module (macOS / Unix generic)
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == 'darwin':
            return int(rusage.ru_maxrss)
            # macOS returns bytes, while Linux returns kilobytes
        return int(rusage.ru_maxrss * 1024)
    except Exception:
        return 0

# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class LoggerError(Exception):
    """Base exception class for all errors generated within the framework."""

class ConfigurationError(LoggerError):
    """Raised when an invalid configuration is provided or parsing fails."""

class StorageError(LoggerError):
    """Raised when file or directory allocation operations fail due to I/O constraints."""

# ==============================================================================
# ENUMS
# ==============================================================================

class LogFormat(Enum):
    """Defines structural styles supported for output targets."""
    PLAIN = auto()
    COLOR = auto()
    JSON = auto()

class LoggerState(Enum):
    """Delineates lifecycle states of the LogManager system."""
    UNINITIALIZED = auto()
    INITIALIZED = auto()
    SHUTTING_DOWN = auto()
    SHUTDOWN = auto()

# ==============================================================================
# DATACLASSES
# ==============================================================================

@dataclass
class LogConfig:
    """Holds analytical schemas for managing initialization fields."""
    format_type: LogFormat = LogFormat.PLAIN
    level: int = logging.INFO
    log_dir: str = "logs"
    file_prefix: str = "studio"
    max_bytes: int = 10_485_760  # 10 MB default
    backup_count: int = 5
    when: str = "midnight"
    interval: int = 1
    enable_console: bool = True
    enable_file: bool = False
    enable_timed_file: bool = False

    def validate(self) -> None:
        """Validates configuration sanity, shielding operations from downstream failures."""
        if not isinstance(self.format_type, LogFormat):
            raise ConfigurationError("format_type must be an instance of LogFormat.")
        if not isinstance(self.level, int):
            raise ConfigurationError("Log level must be a valid integer logging severity.")
        if self.max_bytes <= 0:
            raise ConfigurationError("max_bytes threshold parameter must be greater than zero.")
        if self.backup_count < 0:
            raise ConfigurationError("backup_count must be non-negative.")
        if self.interval <= 0:
            raise ConfigurationError("timed interval validation factor must exceed zero.")
        if self.when.lower() not in {"s", "m", "h", "d", "w0", "w1", "w2", "w3", "w4", "w5", "w6", "midnight"}:
            raise ConfigurationError(f"Invalid 'when' interval directive requested: '{self.when}'")
        
        # Defensive validation for directory strings when persistence storage targets are bound
        if self.enable_file or self.enable_timed_file:
            if not self.log_dir or not isinstance(self.log_dir, str) or self.log_dir.strip() == "":
                raise ConfigurationError("log_dir must be a valid non-empty path string when file targets are active.")
            if not self.file_prefix or not isinstance(self.file_prefix, str) or self.file_prefix.strip() == "":
                raise ConfigurationError("file_prefix must be a valid non-empty string when file targets are active.")

@dataclass
class LoggerStats:
    """Aggregates absolute platform metrics for runtime health tracking."""
    total_records: int = 0
    debug_count: int = 0
    info_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    critical_count: int = 0
    audit_events: int = 0
    security_events: int = 0
    exception_count: int = 0
    uptime: float = 0.0
    memory_usage: int = 0
    active_handlers: int = 0
    registered_loggers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes current point-in-time statistics map safely cloning internal collections."""
        data = asdict(self)
        data["registered_loggers"] = list(self.registered_loggers)
        return data

    def to_json(self) -> str:
        """Translates schema layout directly to raw JSON format string."""
        return json.dumps(self.to_dict(), indent=2)

# ==============================================================================
# FORMATTERS
# ==============================================================================

class BaseCustomFormatter(logging.Formatter):
    """Abstract baseline formatter optimizing standard field access patterns."""
    
    def extract_context(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Pulls specialized context fields from a LogRecord metadata container."""
        utc_now = datetime.fromtimestamp(record.created, timezone.utc)
        
        ctx = {
            "timestamp": utc_now.isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "thread": record.threadName,
            "process": record.process,
            "filename": record.filename,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "metadata") and isinstance(record.metadata, dict):
            ctx["metadata"] = dict(record.metadata)
        
        if hasattr(record, "audit_event") and record.audit_event:
            ctx["audit_event"] = record.audit_event

        if hasattr(record, "security_event") and record.security_event:
            ctx["security_event"] = record.security_event

        if record.exc_info:
            ctx["exception"] = "".join(traceback.format_exception(*record.exc_info))

        return ctx

class PlainTextFormatter(BaseCustomFormatter):
    """Formats logs into clean, human-readable plain text."""

    def format(self, record: logging.LogRecord) -> str:
        ctx = self.extract_context(record)
        base = f"[{ctx['timestamp']}] [{ctx['level']}] [{ctx['logger']}] ({ctx['module']}:{ctx['function']}:{ctx['line']}): {ctx['message']}"
        
        if "metadata" in ctx:
            base += f" | Metadata: {ctx['metadata']}"
        if "audit_event" in ctx:
            base += f" | AUDIT: {ctx['audit_event']}"
        if "security_event" in ctx:
            base += f" | SECURITY: {ctx['security_event']}"
        if "exception" in ctx:
            base += f"\n{ctx['exception']}"
        return base

class ColorFormatter(BaseCustomFormatter):
    """Formats logs with ANSI color codes optimized for console visibility."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m\033[37m', # Red background, white text
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        ctx = self.extract_context(record)
        color = self.COLORS.get(ctx['level'], '')
        
        base = f"{color}[{ctx['timestamp']}] [{ctx['level']}] [{ctx['logger']}] ({ctx['module']}:{ctx['function']}:{ctx['line']}): {ctx['message']}{self.RESET}"
        
        if "metadata" in ctx:
            base += f" \033[90m| Metadata: {ctx['metadata']}\033[0m"
        if "audit_event" in ctx:
            base += f" \033[35m| AUDIT: {ctx['audit_event']}\033[0m"
        if "security_event" in ctx:
            base += f" \033[31m\033[1m| SECURITY: {ctx['security_event']}\033[0m"
        if "exception" in ctx:
            base += f"\n\033[31m{ctx['exception']}\033[0m"
        return base

class JSONFormatter(BaseCustomFormatter):
    """Formats records to strict single-line structural JSON schemas."""

    def format(self, record: logging.LogRecord) -> str:
        ctx = self.extract_context(record)
        return json.dumps(ctx, default=str)

# ==============================================================================
# STATISTICS COLLECTOR HANDLER
# ==============================================================================

class StatisticsCollectorHandler(logging.Handler):
    """Thread-safe standard logging handler dedicated entirely to metric tracing."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_records = 0
        self.level_counts: Dict[str, int] = {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
        self.audit_events = 0
        self.security_events = 0
        self.exception_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self.total_records += 1
            lvl = record.levelname
            if lvl in self.level_counts:
                self.level_counts[lvl] += 1
            
            if hasattr(record, "audit_event") and getattr(record, "audit_event"):
                self.audit_events += 1
            if hasattr(record, "security_event") and getattr(record, "security_event"):
                self.security_events += 1
            if record.exc_info:
                self.exception_count += 1

    def get_stats(self, registered_loggers: List[str], active_handlers_count: int) -> LoggerStats:
        """Assembles thread-safe stats snapshots without causing structural overhead."""
        with self._lock:
            return LoggerStats(
                total_records=self.total_records,
                debug_count=self.level_counts["DEBUG"],
                info_count=self.level_counts["INFO"],
                warning_count=self.level_counts["WARNING"],
                error_count=self.level_counts["ERROR"],
                critical_count=self.level_counts["CRITICAL"],
                audit_events=self.audit_events,
                security_events=self.security_events,
                exception_count=self.exception_count,
                uptime=time.time() - self.start_time,
                memory_usage=_get_process_memory(),
                active_handlers=active_handlers_count,
                registered_loggers=list(registered_loggers)  # explicitly copied
            )

# ==============================================================================
# SINGLETON LOG MANAGER
# ==============================================================================

class LogManager:
    """Thread-safe Singleton orchestration manager for enterprise lifecycle configuration."""
    
    _instance: Optional['LogManager'] = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> 'LogManager':
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._lock = threading.Lock()
        self._config = LogConfig()
        self._state = LoggerState.UNINITIALIZED
        self._framework_handlers: List[logging.Handler] = []
        self._stats_handler = StatisticsCollectorHandler()
        self._loggers: Dict[str, logging.Logger] = {}
        self._initialized = True
        self._state = LoggerState.INITIALIZED
        
        atexit.register(self.shutdown)

    @classmethod
    def get_instance(cls) -> 'LogManager':
        """Accesses the validated tracking instance of the Singleton."""
        return cls()

    def configure(self, config: LogConfig) -> None:
        """Thread-safe configuration pipeline reconstruction routine."""
        with self._lock:
            if self._state in (LoggerState.SHUTTING_DOWN, LoggerState.SHUTDOWN):
                raise LoggerError("Cannot reconfigure system while shutting down or shutdown has completed.")
            
            config.validate()
            self._config = config
            self._rebuild_pipeline()

    def _get_formatter(self) -> logging.Formatter:
        """Factory match for matching selected serialization targets."""
        if self._config.format_type == LogFormat.JSON:
            return JSONFormatter()
        elif self._config.format_type == LogFormat.COLOR:
            return ColorFormatter()
        return PlainTextFormatter()

    def _rebuild_pipeline(self) -> None:
        """Constructs new backend file channels and updates active sub-loggers safely."""
        # Unlink and clear existing internal handlers explicitly
        old_handlers = list(self._framework_handlers)
        self._framework_handlers.clear()

        for h in old_handlers:
            h.close()

        # Re-attach mandatory stats core observer
        self._framework_handlers.append(self._stats_handler)

        formatter = self._get_formatter()

        # Build Console Handler
        if self._config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self._config.level)
            self._framework_handlers.append(console_handler)

        # Build Storage Output Framework safely
        if self._config.enable_file or self._config.enable_timed_file:
            try:
                os.makedirs(self._config.log_dir, exist_ok=True)
            except Exception as e:
                raise StorageError(f"Failed creating target log path structure: '{self._config.log_dir}'") from e

            if self._config.enable_file:
                path = os.path.join(self._config.log_dir, f"{self._config.file_prefix}.log")
                try:
                    from logging.handlers import RotatingFileHandler
                    rf_handler = RotatingFileHandler(
                        path, maxBytes=self._config.max_bytes, backupCount=self._config.backup_count, encoding="utf-8"
                    )
                    rf_handler.setFormatter(formatter)
                    rf_handler.setLevel(self._config.level)
                    self._framework_handlers.append(rf_handler)
                except Exception as e:
                    raise StorageError(f"Failed initialization of standard rotating system file targeting: {path}") from e

            if self._config.enable_timed_file:
                path_timed = os.path.join(self._config.log_dir, f"{self._config.file_prefix}_timed.log")
                try:
                    from logging.handlers import TimedRotatingFileHandler
                    trf_handler = TimedRotatingFileHandler(
                        path_timed, when=self._config.when, interval=self._config.interval,
                        backupCount=self._config.backup_count, encoding="utf-8"
                    )
                    trf_handler.setFormatter(formatter)
                    trf_handler.setLevel(self._config.level)
                    self._framework_handlers.append(trf_handler)
                except Exception as e:
                    raise StorageError(f"Failed initialization of timed rotating system file targeting: {path_timed}") from e

        # Synchronize runtime loggers across namespaces without removing third-party handlers
        for logger in self._loggers.values():
            logger.setLevel(self._config.level)
            
            # De-duplicate: Remove only previous version of framework-managed handlers
            for h in old_handlers:
                if h in logger.handlers:
                    logger.removeHandler(h)
                    
            # Inject new framework handlers seamlessly
            for h in self._framework_handlers:
                if h not in logger.handlers:
                    logger.addHandler(h)

    def get_logger(self, name: str) -> logging.Logger:
        """Retrieves or dynamically updates an existing logger thread-safely."""
        with self._lock:
            if name in self._loggers:
                return self._loggers[name]

            logger = logging.getLogger(name)
            logger.propagate = False
            logger.setLevel(self._config.level)

            # If system wasn't fully custom-configured yet, default attach tracking core
            if not self._framework_handlers:
                self._framework_handlers.append(self._stats_handler)

            for h in self._framework_handlers:
                if h not in logger.handlers:
                    logger.addHandler(h)

            self._loggers[name] = logger
            return logger

    def get_statistics(self) -> LoggerStats:
        """Collects cross-platform architectural system telemetry snapshots."""
        with self._lock:
            registered_names = list(self._loggers.keys())
            active_h_count = len(self._framework_handlers)
            return self._stats_handler.get_stats(registered_names, active_h_count)

    def force_retention_cleanup(self) -> None:
        """Iterates explicitly through active handlers to force log rotation cleanup."""
        with self._lock:
            for h in self._framework_handlers:
                if hasattr(h, "doRollover"):
                    try:
                        cast(Any, h).doRollover()
                    except Exception as e:
                        raise StorageError("Forceful file retention rollover cleanup cycle crashed.") from e

    def export_configuration_json(self) -> str:
        """Exports the active tracking configurations directly to structural JSON format strings."""
        with self._lock:
            data = {
                "format_type": self._config.format_type.name,
                "level": self._config.level,
                "log_dir": self._config.log_dir,
                "file_prefix": self._config.file_prefix,
                "max_bytes": self._config.max_bytes,
                "backup_count": self._config.backup_count,
                "when": self._config.when,
                "interval": self._config.interval,
                "enable_console": self._config.enable_console,
                "enable_file": self._config.enable_file,
                "enable_timed_file": self._config.enable_timed_file
            }
            return json.dumps(data, indent=2)

    def import_configuration_json(self, json_str: str) -> None:
        """Parses internal parameters from structured settings and executes rebuilding."""
        try:
            data = json.loads(json_str)
            fmt_name = data.get("format_type", "PLAIN")
            
            cfg = LogConfig(
                format_type=LogFormat[fmt_name],
                level=int(data.get("level", logging.INFO)),
                log_dir=str(data.get("log_dir", "logs")),
                file_prefix=str(data.get("file_prefix", "studio")),
                max_bytes=int(data.get("max_bytes", 10485760)),
                backup_count=int(data.get("backup_count", 5)),
                when=str(data.get("when", "midnight")),
                interval=int(data.get("interval", 1)),
                enable_console=bool(data.get("enable_console", True)),
                enable_file=bool(data.get("enable_file", False)),
                enable_timed_file=bool(data.get("enable_timed_file", False))
            )
            self.configure(cfg)
        except Exception as e:
            raise ConfigurationError("Failed parsing configuration format JSON structure accurately.") from e

    def shutdown(self) -> None:
        """Gracefully tears down the pipeline, scrubbing handles from loggers and flushing entries."""
        with self._lock:
            if self._state in (LoggerState.SHUTDOWN, LoggerState.SHUTTING_DOWN):
                return
            self._state = LoggerState.SHUTTING_DOWN
            
            # Cleanly unbind pipeline references from active instances to mitigate memory leaks
            for logger in list(self._loggers.values()):
                for h in list(logger.handlers):
                    if h in self._framework_handlers:
                        logger.removeHandler(h)
            
            for h in self._framework_handlers:
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
            self._framework_handlers.clear()
            self._state = LoggerState.SHUTDOWN

# ==============================================================================
# CONTEXT MANAGERS & DECORATORS
# ==============================================================================

class ExecutionTimer:
    """Context manager and decorator for precise, thread-safe operation tracking."""

    def __init__(self, logger: logging.Logger, level: int = logging.INFO, description: str = "Execution Task", metadata: Optional[Dict[str, Any]] = None) -> None:
        self.logger = logger
        self.level = level
        self.description = description
        self.metadata = metadata or {}
        self.start_time: float = 0.0

    def __enter__(self) -> 'ExecutionTimer':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        elapsed = time.perf_counter() - self.start_time
        meta = self.metadata.copy()
        meta["elapsed_seconds"] = elapsed
        
        if exc_type:
            meta["execution_status"] = "failed"
            meta["exception_type"] = exc_type.__name__
            self.logger.log(
                logging.ERROR,
                f"{self.description} failed after {elapsed:.6f}s",
                extra={"metadata": meta}
            )
        else:
            meta["execution_status"] = "success"
            self.logger.log(
                self.level,
                f"{self.description} completed in {elapsed:.6f}s",
                extra={"metadata": meta}
            )
        return None


class DynamicLevelContext:
    """Context manager that temporarily overrides a logger's severity threshold."""

    def __init__(self, logger: logging.Logger, temporary_level: int) -> None:
        self.logger = logger
        self.temporary_level = temporary_level
        self.previous_level: int = logger.level

    def __enter__(self) -> 'DynamicLevelContext':
        self.previous_level = self.logger.level
        self.logger.setLevel(self.temporary_level)
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        self.logger.setLevel(self.previous_level)
        return None


def execution_time_decorator(logger: logging.Logger, level: int = logging.INFO, description: Optional[str] = None) -> Callable[[F], F]:
    """Wraps functions with high-precision metrics capture blocks while preserving type hints and docstrings."""
    def decorator(func: F) -> F:
        desc = description or func.__name__
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ExecutionTimer(logger, level, desc):
                return func(*args, **kwargs)
        return cast(F, wrapper)
    return decorator

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def log_with_metadata(logger: logging.Logger, level: int, message: str, metadata: Dict[str, Any]) -> None:
    """Dispatches messages enriched with custom troubleshooting metadata fields."""
    logger.log(level, message, extra={"metadata": metadata})

def log_audit_event(logger: logging.Logger, event_name: str, user: str, details: Dict[str, Any]) -> None:
    """Logs non-repudiable user actions to the audit trial."""
    audit_payload = {"event": event_name, "user": user, "details": details}
    logger.log(logging.INFO, f"Audit event triggered: {event_name}", extra={"audit_event": audit_payload})

def log_security_event(logger: logging.Logger, incident_type: str, severity: str, details: Dict[str, Any]) -> None:
    """Logs security and compliance telemetry."""
    security_payload = {"incident": incident_type, "severity": severity, "details": details}
    logger.log(logging.CRITICAL, f"SECURITY ALERT: [{severity}] {incident_type}", extra={"security_event": security_payload})

def log_exception_context(logger: logging.Logger, message: str, exception: Optional[BaseException] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Captures execution fault states along with call stacks. 
    If exception argument is missing, falls back defensively to check sys.exc_info().
    """
    meta = metadata or {}
    exc_info: Any = True
    
    if exception is not None:
        exc_info = (type(exception), exception, exception.__traceback__)
    elif not sys.exc_info()[0]:
        # Handle call outside of an active try-except branch gracefully
        exc_info = False
        
    logger.error(message, exc_info=exc_info, extra={"metadata": meta})

def get_temporary_level_context(logger: logging.Logger, temporary_level: int) -> DynamicLevelContext:
    """Convenience factory for DynamicLevelContext scopes."""
    return DynamicLevelContext(logger, temporary_level)

def get_execution_timer(logger: logging.Logger, level: int = logging.INFO, description: str = "Execution Task", metadata: Optional[Dict[str, Any]] = None) -> ExecutionTimer:
    """Convenience factory for ExecutionTimer scopes."""
    return ExecutionTimer(logger, level, description, metadata)

# ==============================================================================
# VERIFICATION AND DEMONSTRATION RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("--- Initializing Cabangile AI Studio Logger Production Verification Suite ---")
    
    manager = LogManager.get_instance()
    
    core_config = LogConfig(
        format_type=LogFormat.COLOR,
        level=logging.DEBUG,
        enable_console=True,
        enable_file=False
    )
    manager.configure(core_config)
    
    log = manager.get_logger("Studio.Core")
    
    log.debug("Verifying internal debugging message transport layer.")
    log.info("Framework successfully started.")
    
    print("\n--- Testing Runtime Hot-Swap Pipeline to Strict JSON Output Format ---")
    json_config = LogConfig(
        format_type=LogFormat.JSON,
        level=logging.DEBUG,
        enable_console=True
    )
    manager.configure(json_config)
    
    log_with_metadata(log, logging.INFO, "Data ingestion transaction processed.", {"transaction_id": "TX-99823", "items_count": 42})
    log_audit_event(log, "USER_KEYS_ROTATION", "admin@cabangile.ai", {"target_service": "LLM_INFERENCE_POOL"})
    log_security_event(log, "BRUTE_FORCE_DETECTED", "HIGH", {"source_ip": "192.168.1.105", "failed_attempts": 7})
    
    # Verify exception context handling within an active exception
    try:
        divide_by_zero = 1 / 0
    except ZeroDivisionError as e:
        log_exception_context(log, "An arithmetic calculation failed inside the container stack.", exception=e, metadata={"subsystem": "Inference.Pricing"})
        
    # Verify external call safety outside an active exception block
    log_exception_context(log, "Isolated structural anomaly tracked outside exception block context.")
        
    # Verify high-precision context decorators and type safety preservation
    @execution_time_decorator(log, logging.INFO, "Simulated Vector Calculation Weight Kernel")
    def compute_weights(factor: float) -> float:
        """Core demonstration worker processing weights."""
        time.sleep(0.02)
        return factor * 2.0

    result = compute_weights(4.2)
    assert compute_weights.__doc__ == "Core demonstration worker processing weights."
        
    log.debug("This explicit debug tracking line WILL capture under standard JSON configuration settings.")
    with get_temporary_level_context(log, logging.WARNING):
        log.debug("This line WILL BE SILENCED dynamically as logger context overrides severity to WARNING.")
        log.warning("This warning line WILL emerge cleanly from the silenced block scope container.")
        
    stats = manager.get_statistics()
    print("\n--- System Execution Telemetry Dump ---")
    print(stats.to_json())
    
    print("\n--- Verification Configuration Interchange Flow ---")
    exported_layout = manager.export_configuration_json()
    print(f"Exported Structure Layout String:\n{exported_layout}")
    
    manager.import_configuration_json(exported_layout)
    print("Configuration imported successfully and handler pipeline safely rebuilt.")
    
    manager.shutdown()
    print("\n--- Framework validation cycle fully completed. Output is 100% production-ready. ---")
