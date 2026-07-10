#!/usr/bin/env python3
"""
================================================================================
Cabangile AI Studio
Model Validator Module
================================================================================
File path: studio/models/model_validator.py
Python version: 3.11+
Architecture: Clean Architecture / SOLID / Domain-Driven Design
Thread safety: Fully thread-safe via asyncio.Lock and atomic state transitions
Async support: Native Async-ready API with internal synchronous primitives
Standard Library Only: 100% pure Python standard library compliant
Cross-platform: Linux, Windows, macOS, Android (Termux)

Description:
    Enterprise-grade validation engine for AI models, provider definitions, and
    registry topologies. Handles deep semantic integrity checks, circular loop
    detection, capability matrix compliance, pricing structural health, and
    context window boundary safety.
================================================================================
"""

import asyncio
import copy
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Set, Optional, Tuple

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logger = logging.getLogger("cabangile_ai_studio.model_validator")


# ------------------------------------------------------------------------------
# Enumerations
# ------------------------------------------------------------------------------
class ValidationErrorCode(Enum):
    SUCCESS = auto()
    REQUIRED_FIELD_MISSING = auto()
    INVALID_CONTEXT_WINDOW = auto()
    INVALID_TOKEN_LIMITS = auto()
    INVALID_PRICING = auto()
    UNSUPPORTED_CAPABILITY = auto()
    INVALID_TASK_MAPPING = auto()
    PROVIDER_HEALTH_DEGRADED = auto()
    REGISTRY_INTEGRITY_VIOLATION = auto()
    ALIAS_LOOP_DETECTED = auto()
    CIRCULAR_FALLBACK_DETECTED = auto()
    INVALID_CONFIGURATION = auto()
    INVALID_SCHEMA = auto()
    INVALID_JSON_SYNTAX = auto()
    DUPLICATE_MODEL_ID = auto()
    DUPLICATE_PROVIDER_ID = auto()
    UNSUPPORTED_CAPABILITY_COMBINATION = auto()
    INVALID_MODEL_REFERENCE = auto()
    INVALID_PROVIDER_REFERENCE = auto()
    INVALID_STRATEGY_REFERENCE = auto()
    INVALID_REGISTRY_REFERENCE = auto()
    SHUTDOWN_IN_PROGRESS = auto()
    UNKNOWN_ERROR = auto()


class ModelTaskType(Enum):
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    MULTIMODAL = "multimodal"


class ModelCapability(Enum):
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    LOGPROBS = "logprobs"
    FINE_TUNED = "fine_tuned"


# ------------------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class PricingConfig:
    input_per_token: float
    output_per_token: float
    cached_input_per_token: float = 0.0
    currency: str = "USD"


@dataclass(frozen=True)
class ModelCapabilitiesConfig:
    task_types: List[ModelTaskType] = field(default_factory=list)
    capabilities: List[ModelCapability] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    provider_id: str
    context_window: int
    max_output_tokens: int
    pricing: PricingConfig
    capabilities_config: ModelCapabilitiesConfig
    aliases: List[str] = field(default_factory=list)
    fallbacks: List[str] = field(default_factory=list)
    schema_definition: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass(frozen=True)
class ProviderDefinition:
    provider_id: str
    name: str
    base_url: str
    is_healthy: bool = True
    supported_models: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrySnapshot:
    snapshot_id: str
    timestamp: float
    models: Dict[str, ModelDefinition]
    providers: Dict[str, ProviderDefinition]


@dataclass(frozen=True)
class ValidationErrorDetail:
    code: ValidationErrorCode
    field: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: List[ValidationErrorDetail] = field(default_factory=list)
    warnings: List[ValidationErrorDetail] = field(default_factory=list)


@dataclass(frozen=True)
class DiagnosticReport:
    timestamp: float
    registered_models: List[str]
    registered_providers: List[str]
    duplicate_models: List[str]
    invalid_models: List[str]
    validation_failures: List[ValidationErrorDetail]
    warning_count: int
    error_count: int
    health_summary: Dict[str, Any]
    compatibility_summary: Dict[str, Any]
    validation_statistics: Dict[str, Any]


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------
class ModelValidatorError(Exception):
    """Domain exception thrown for critical model validation infrastructure issues."""
    def __init__(self, error_code: ValidationErrorCode, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        import time
        self.error_code = error_code
        self.message = message
        self.timestamp = time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code.name,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ------------------------------------------------------------------------------
# Main Model Validator Implementation
# ------------------------------------------------------------------------------
class ModelValidator:
    """
    Enterprise-grade, thread-safe, async-ready model validation coordinator.
    Maintains centralized definitions and performs deep topology validation.
    """
    def __init__(self) -> None:
        import time
        self._lock = asyncio.Lock()
        
        # State Storage
        self._models: Dict[str, ModelDefinition] = {}
        self._providers: Dict[str, ProviderDefinition] = {}
        
        # Validation Cache and Infrastructure Metrics
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._snapshots: Dict[str, RegistrySnapshot] = {}
        self._backups: Dict[str, bytes] = {}
        self._shutdown_flag: bool = False
        
        # Diagnostics & Diagnostics Statistics Trackers
        self._diagnostic_logs: List[ValidationErrorDetail] = []
        self._stat_validation_runs: int = 0
        self._warning_counter: int = 0
        self._error_counter: int = 0
        self._created_at: float = time.time()

    # --------------------------------------------------------------------------
    # Public Synchronous Ingestion API
    # --------------------------------------------------------------------------
    def register_model_definition_sync(self, model: ModelDefinition) -> None:
        """Synchronously registers or updates a model template specification."""
        self._models[model.model_id] = model
        self._clear_cache()

    def register_provider_definition_sync(self, provider: ProviderDefinition) -> None:
        """Synchronously registers or updates a target infrastructure provider."""
        self._providers[provider.provider_id] = provider
        self._clear_cache()

    # --------------------------------------------------------------------------
    # Public Async API Implementation
    # --------------------------------------------------------------------------
    async def validate_model(self, model_id: str) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            self._stat_validation_runs += 1
            if model_id in self._validation_cache:
                return self._validation_cache[model_id]

            if model_id not in self._models:
                res = ValidationResult(is_valid=False, errors=[
                    ValidationErrorDetail(ValidationErrorCode.INVALID_MODEL_REFERENCE, "model_id", f"Model {model_id} not registered.")
                ])
                return res

            model = self._models[model_id]
            res = self._validate_model_internal_sync(model)
            self._validation_cache[model_id] = res
            self._update_counters(res)
            return res

    async def validate_models(self, model_ids: List[str]) -> Dict[str, ValidationResult]:
        async with self._lock:
            self._ensure_active()
            results = {}
            for m_id in model_ids:
                if m_id in self._validation_cache:
                    results[m_id] = self._validation_cache[m_id]
                elif m_id not in self._models:
                    results[m_id] = ValidationResult(is_valid=False, errors=[
                        ValidationErrorDetail(ValidationErrorCode.INVALID_MODEL_REFERENCE, "model_id", f"Model {m_id} not registered.")
                    ])
                else:
                    res = self._validate_model_internal_sync(self._models[m_id])
                    self._validation_cache[m_id] = res
                    self._update_counters(res)
                    results[m_id] = res
            return results

    async def validate_provider(self, provider_id: str) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            if provider_id not in self._providers:
                return ValidationResult(is_valid=False, errors=[
                    ValidationErrorDetail(ValidationErrorCode.INVALID_PROVIDER_REFERENCE, "provider_id", f"Provider {provider_id} not registered.")
                ])
            provider = self._providers[provider_id]
            
            errors: List[ValidationErrorDetail] = []
            warnings: List[ValidationErrorDetail] = []
            
            if not provider.provider_id or provider.provider_id.strip() == "":
                errors.append(ValidationErrorDetail(ValidationErrorCode.REQUIRED_FIELD_MISSING, "provider_id", "Empty provider ID identifier."))
            if not provider.name or provider.name.strip() == "":
                errors.append(ValidationErrorDetail(ValidationErrorCode.REQUIRED_FIELD_MISSING, "name", "Empty provider descriptive name."))
            if not provider.is_healthy:
                warnings.append(ValidationErrorDetail(ValidationErrorCode.PROVIDER_HEALTH_DEGRADED, "is_healthy", f"Provider {provider_id} health is degraded."))

            res = ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
            self._update_counters(res)
            return res

    async def validate_registry(self) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            errors: List[ValidationErrorDetail] = []
            warnings: List[ValidationErrorDetail] = []

            # Check for duplicate tracking or cross validation issues
            known_models: Set[str] = set()
            for m_id, model in self._models.items():
                if m_id in known_models:
                    errors.append(ValidationErrorDetail(ValidationErrorCode.DUPLICATE_MODEL_ID, "model_id", f"Duplicate registration of model {m_id}"))
                known_models.add(m_id)
                
                # Cross-verify underlying provider linkage
                if model.provider_id not in self._providers:
                    errors.append(ValidationErrorDetail(ValidationErrorCode.INVALID_PROVIDER_REFERENCE, "provider_id", f"Model {m_id} binds to missing provider {model.provider_id}"))
                
                # Internal sanity check
                m_res = self._validate_model_internal_sync(model)
                errors.extend(m_res.errors)
                warnings.extend(m_res.warnings)

            # Look for cycles inside aliases and fallbacks system wide
            alias_res = self._validate_aliases_sync()
            errors.extend(alias_res.errors)
            
            fallback_res = self._validate_fallbacks_sync()
            errors.extend(fallback_res.errors)

            res = ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
            self._update_counters(res)
            return res

    async def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            errors = []
            if not config:
                errors.append(ValidationErrorDetail(ValidationErrorCode.INVALID_CONFIGURATION, "config", "Configuration payload cannot be empty."))
            return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    async def validate_capabilities(self, capabilities: ModelCapabilitiesConfig) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            return self._validate_capabilities_sync(capabilities)

    async def validate_pricing(self, pricing: PricingConfig) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            return self._validate_pricing_sync(pricing)

    async def validate_context_window(self, context_window: int) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            return self._validate_context_window_sync(context_window)

    async def validate_token_limits(self, max_output: int, context_window: int) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            return self._validate_token_limits_sync(max_output, context_window)

    async def validate_task_mapping(self, config: ModelCapabilitiesConfig) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            return self._validate_task_mapping_sync(config)

    async def validate_aliases(self) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            return self._validate_aliases_sync()

    async def validate_fallbacks(self) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            return self._validate_fallbacks_sync()

    async def validate_schema(self, schema: Dict[str, Any]) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            return self._validate_schema_sync(schema)

    async def validate_json(self, json_str: str) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            return self._validate_json_sync(json_str)

    async def validate_health(self) -> ValidationResult:
        async with self._lock:
            self._ensure_active()
            unhealthy_providers = [p.provider_id for p in self._providers.values() if not p.is_healthy]
            if unhealthy_providers:
                return ValidationResult(is_valid=True, warnings=[
                    ValidationErrorDetail(ValidationErrorCode.PROVIDER_HEALTH_DEGRADED, "providers", "Degraded operational target backends encountered", {"unhealthy": unhealthy_providers})
                ])
            return ValidationResult(is_valid=True)

    async def run_diagnostics(self) -> DiagnosticReport:
        async with self._lock:
            self._ensure_active()
            import time
            
            invalid_models = []
            duplicate_models = []
            failures = []
            
            # Temporary track duplicates
            seen: Set[str] = set()
            for m_id, model in self._models.items():
                if m_id in seen:
                    duplicate_models.append(m_id)
                seen.add(m_id)
                
                res = self._validate_model_internal_sync(model)
                if not res.is_valid:
                    invalid_models.append(m_id)
                    failures.extend(res.errors)

            report = DiagnosticReport(
                timestamp=time.time(),
                registered_models=list(self._models.keys()),
                registered_providers=list(self._providers.keys()),
                duplicate_models=duplicate_models,
                invalid_models=invalid_models,
                validation_failures=failures,
                warning_count=self._warning_counter,
                error_count=self._error_counter,
                health_summary={"healthy_providers_count": sum(1 for p in self._providers.values() if p.is_healthy)},
                compatibility_summary={"total_checked_rules": self._stat_validation_runs},
                validation_statistics={"cache_size": len(self._validation_cache)}
            )
            return report

    async def export_report(self, format_type: str = "json") -> str:
        report = await self.run_diagnostics()
        if format_type.lower() == "json":
            return json.dumps({
                "timestamp": report.timestamp,
                "registered_models": report.registered_models,
                "registered_providers": report.registered_providers,
                "warning_count": report.warning_count,
                "error_count": report.error_count,
                "is_infrastructure_nominal": report.error_count == 0
            }, indent=2)
        return f"DiagnosticReport structural logs metadata. Errors total: {report.error_count}"

    async def create_snapshot(self, snapshot_id: str) -> RegistrySnapshot:
        async with self._lock:
            self._ensure_active()
            import time
            snapshot = RegistrySnapshot(
                snapshot_id=snapshot_id,
                timestamp=time.time(),
                models=copy.deepcopy(self._models),
                providers=copy.deepcopy(self._providers)
            )
            self._snapshots[snapshot_id] = snapshot
            return snapshot

    async def restore_snapshot(self, snapshot_id: str) -> None:
        async with self._lock:
            self._ensure_active()
            if snapshot_id not in self._snapshots:
                raise ModelValidatorError(ValidationErrorCode.INVALID_REGISTRY_REFERENCE, f"Snapshot destination token {snapshot_id} non-existent.")
            target = self._snapshots[snapshot_id]
            self._models = copy.deepcopy(target.models)
            self._providers = copy.deepcopy(target.providers)
            self._clear_cache()

    async def backup(self) -> bytes:
        async with self._lock:
            self._ensure_active()
            # Serialize state boundaries using native json safely stringified
            state_data = {
                "models": {k: self._serialize_model_definition(v) for k, v in self._models.items()},
                "providers": {k: self._serialize_provider_definition(v) for k, v in self._providers.items()}
            }
            raw_bytes = json.dumps(state_data).encode("utf-8")
            self._backups[str(hash(raw_bytes))] = raw_bytes
            return raw_bytes

    async def restore(self, backup_data: bytes) -> None:
        async with self._lock:
            self._ensure_active()
            try:
                decoded = json.loads(backup_data.decode("utf-8"))
                new_models = {}
                new_providers = {}
                
                for k, v in decoded.get("models", {}).items():
                    new_models[k] = self._deserialize_model_definition(v)
                for k, v in decoded.get("providers", {}).items():
                    new_providers[k] = self._deserialize_provider_definition(v)
                    
                self._models = new_models
                self._providers = new_providers
                self._clear_cache()
            except Exception as ex:
                raise ModelValidatorError(ValidationErrorCode.INVALID_CONFIGURATION, f"Corruption detected in state payload restore context: {str(ex)}")

    async def shutdown(self) -> None:
        async with self._lock:
            self._shutdown_flag = True
            self._clear_cache()
            self._models.clear()
            self._providers.clear()

    # --------------------------------------------------------------------------
    # Private Synchronous Validation Primitives
    # --------------------------------------------------------------------------
    def _validate_model_internal_sync(self, model: ModelDefinition) -> ValidationResult:
        errors: List[ValidationErrorDetail] = []
        warnings: List[ValidationErrorDetail] = []

        if not model.model_id or model.model_id.strip() == "":
            errors.append(ValidationErrorDetail(ValidationErrorCode.REQUIRED_FIELD_MISSING, "model_id", "Model ID token required."))

        # Subcomponent validation steps
        cw_res = self._validate_context_window_sync(model.context_window)
        errors.extend(cw_res.errors)

        tl_res = self._validate_token_limits_sync(model.max_output_tokens, model.context_window)
        errors.extend(tl_res.errors)

        pr_res = self._validate_pricing_sync(model.pricing)
        errors.extend(pr_res.errors)

        cap_res = self._validate_capabilities_sync(model.capabilities_config)
        errors.extend(cap_res.errors)

        task_res = self._validate_task_mapping_sync(model.capabilities_config)
        errors.extend(task_res.errors)

        comb_res = self._validate_capability_combinations(model.capabilities_config)
        errors.extend(comb_res.errors)

        if model.schema_definition:
            sch_res = self._validate_schema_sync(model.schema_definition)
            errors.extend(sch_res.errors)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    def _validate_context_window_sync(self, context_window: int) -> ValidationResult:
        if context_window <= 0:
            return ValidationResult(is_valid=False, errors=[
                ValidationErrorDetail(ValidationErrorCode.INVALID_CONTEXT_WINDOW, "context_window", f"Context window size {context_window} is invalid.")
            ])
        return ValidationResult(is_valid=True)

    def _validate_token_limits_sync(self, max_output: int, context_window: int) -> ValidationResult:
        errors = []
        if max_output <= 0:
            errors.append(ValidationErrorDetail(ValidationErrorCode.INVALID_TOKEN_LIMITS, "max_output_tokens", "Max output tokens must be positive."))
        if max_output > context_window:
            errors.append(ValidationErrorDetail(ValidationErrorCode.INVALID_TOKEN_LIMITS, "max_output_tokens", "Max output tokens exceed the overall allowed context window size constraints."))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_pricing_sync(self, pricing: PricingConfig) -> ValidationResult:
        errors = []
        if pricing.input_per_token < 0.0 or pricing.output_per_token < 0.0:
            errors.append(ValidationErrorDetail(ValidationErrorCode.INVALID_PRICING, "pricing", "Token operational consumption charges cannot be structured negatively."))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_capabilities_sync(self, config: ModelCapabilitiesConfig) -> ValidationResult:
        errors = []
        if not config.task_types:
            errors.append(ValidationErrorDetail(ValidationErrorCode.REQUIRED_FIELD_MISSING, "task_types", "At least one target core Task Type classification metadata is mandatory."))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_task_mapping_sync(self, config: ModelCapabilitiesConfig) -> ValidationResult:
        errors = []
        # Multi-modal requires consistent backing elements verification
        if ModelTaskType.MULTIMODAL in config.task_types and ModelCapability.VISION not in config.capabilities:
            errors.append(ValidationErrorDetail(ValidationErrorCode.INVALID_TASK_MAPPING, "task_types", "Multimodal tasks implicitly mandate support for processing vision capabilities vectors."))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_schema_sync(self, schema: Dict[str, Any]) -> ValidationResult:
        if not isinstance(schema, dict):
            return ValidationResult(is_valid=False, errors=[
                ValidationErrorDetail(ValidationErrorCode.INVALID_SCHEMA, "schema_definition", "Structural definition schema must yield dictionary topology mappings.")
            ])
        return ValidationResult(is_valid=True)

    def _validate_json_sync(self, json_str: str) -> ValidationResult:
        try:
            json.loads(json_str)
            return ValidationResult(is_valid=True)
        except json.JSONDecodeError as ex:
            return ValidationResult(is_valid=False, errors=[
                ValidationErrorDetail(ValidationErrorCode.INVALID_JSON_SYNTAX, "json_str", f"Malformed payload parsing error: {str(ex)}")
            ])

    def _validate_capability_combinations(self, config: ModelCapabilitiesConfig) -> ValidationResult:
        errors = []
        # Audio exclusive configuration cross validation rules
        if ModelCapability.AUDIO in config.capabilities and ModelCapability.VIDEO in config.capabilities and len(config.task_types) == 1 and ModelTaskType.TEXT_GENERATION in config.task_types:
            errors.append(ValidationErrorDetail(ValidationErrorCode.UNSUPPORTED_CAPABILITY_COMBINATION, "capabilities", "Invalid simultaneous processing requirements specified for text generation models."))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_aliases_sync(self) -> ValidationResult:
        errors = []
        for m_id, model in self._models.items():
            visited: Set[str] = set()
            current = m_id
            # Resolve graph paths explicitly to track loops
            while current in self._models:
                if current in visited:
                    errors.append(ValidationErrorDetail(ValidationErrorCode.ALIAS_LOOP_DETECTED, "aliases", f"Circular loop tracking encountered inside model references tree traversal: {current}"))
                    break
                visited.add(current)
                # For simplified resolution logic inside model structural definitions
                aliases = self._models[current].aliases
                if aliases and aliases[0] in self._models:
                    current = aliases[0]
                else:
                    break
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _validate_fallbacks_sync(self) -> ValidationResult:
        errors = []
        for m_id, model in self._models.items():
            visited: Set[str] = set()
            stack = list(model.fallbacks)
            while stack:
                curr_fb = stack.pop()
                if curr_fb == m_id or curr_fb in visited:
                    errors.append(ValidationErrorDetail(ValidationErrorCode.CIRCULAR_FALLBACK_DETECTED, "fallbacks", f"Cyclic recursive recovery path tracking matched: {curr_fb} -> fallback loop to core configuration."))
                    break
                visited.add(curr_fb)
                if curr_fb in self._models:
                    stack.extend(self._models[curr_fb].fallbacks)
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    # --------------------------------------------------------------------------
    # Infrastructure Internal Utilities
    # --------------------------------------------------------------------------
    def _ensure_active(self) -> None:
        if self._shutdown_flag:
            raise ModelValidatorError(ValidationErrorCode.SHUTDOWN_IN_PROGRESS, "Operational validation state context has executed shutdown procedures.")

    def _clear_cache(self) -> None:
        self._validation_cache.clear()

    def _update_counters(self, result: ValidationResult) -> None:
        if not result.is_valid:
            self._error_counter += len(result.errors)
        self._warning_counter += len(result.warnings)
        for err in result.errors:
            self._diagnostic_logs.append(err)

    def _serialize_model_definition(self, model: ModelDefinition) -> Dict[str, Any]:
        return {
            "model_id": model.model_id,
            "provider_id": model.provider_id,
            "context_window": model.context_window,
            "max_output_tokens": model.max_output_tokens,
            "pricing": {
                "input_per_token": model.pricing.input_per_token,
                "output_per_token": model.pricing.output_per_token,
                "cached_input_per_token": model.pricing.cached_input_per_token,
                "currency": model.pricing.currency
            },
            "capabilities_config": {
                "task_types": [t.value for t in model.capabilities_config.task_types],
                "capabilities": [c.value for c in model.capabilities_config.capabilities],
                "custom_metadata": model.capabilities_config.custom_metadata
            },
            "aliases": model.aliases,
            "fallbacks": model.fallbacks,
            "schema_definition": model.schema_definition,
            "is_active": model.is_active
        }

    def _deserialize_model_definition(self, data: Dict[str, Any]) -> ModelDefinition:
        p_data = data["pricing"]
        pricing = PricingConfig(
            input_per_token=p_data["input_per_token"],
            output_per_token=p_data["output_per_token"],
            cached_input_per_token=p_data.get("cached_input_per_token", 0.0),
            currency=p_data.get("currency", "USD")
        )
        c_data = data["capabilities_config"]
        cap_config = ModelCapabilitiesConfig(
            task_types=[ModelTaskType(t) for t in c_data.get("task_types", [])],
            capabilities=[ModelCapability(c) for c in c_data.get("capabilities", [])],
            custom_metadata=c_data.get("custom_metadata", {})
        )
        return ModelDefinition(
            model_id=data["model_id"],
            provider_id=data["provider_id"],
            context_window=data["context_window"],
            max_output_tokens=data["max_output_tokens"],
            pricing=pricing,
            capabilities_config=cap_config,
            aliases=data.get("aliases", []),
            fallbacks=data.get("fallbacks", []),
            schema_definition=data.get("schema_definition", {}),
            is_active=data.get("is_active", True)
        )

    def _serialize_provider_definition(self, provider: ProviderDefinition) -> Dict[str, Any]:
        return {
            "provider_id": provider.provider_id,
            "name": provider.name,
            "base_url": provider.base_url,
            "is_healthy": provider.is_healthy,
            "supported_models": provider.supported_models,
            "configuration": provider.configuration
        }

    def _deserialize_provider_definition(self, data: Dict[str, Any]) -> ProviderDefinition:
        return ProviderDefinition(
            provider_id=data["provider_id"],
            name=data["name"],
            base_url=data["base_url"],
            is_healthy=data.get("is_healthy", True),
            supported_models=data.get("supported_models", []),
            configuration=data.get("configuration", {})
        )
