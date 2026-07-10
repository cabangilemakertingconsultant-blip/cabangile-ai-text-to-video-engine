"""
Cabangile AI Studio - Model Selector Module
Path: studio/models/model_selector.py
Target: Python 3.11+ (Standard Library Only)

This module provides an enterprise-grade, thread-safe, async-ready Model Selector 
responsible for selecting the most appropriate AI model for each request based on
capabilities, task types, real-time health metrics, costs, and strategy constraints.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from random import choice
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Setup logging
logger = logging.getLogger("studio.models.model_selector")


# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================

class TaskType(str, Enum):
    STORY_GENERATION = "story_generation"
    CHAT = "chat"
    ASSISTANT = "assistant"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    REWRITE = "rewrite"
    PROOFREADING = "proofreading"
    CODING = "coding"
    CODE_REVIEW = "code_review"
    IMAGE_ANALYSIS = "image_analysis"
    VISION = "vision"
    SPEECH = "speech"
    TTS = "tts"
    STTT = "stt"
    VIDEO_SCRIPT = "video_script"
    YOUTUBE = "youtube"
    SEO = "seo"
    MARKETING = "marketing"
    CLASSIFICATION = "classification"
    REASONING = "reasoning"
    PLANNING = "planning"
    CREATIVE = "creative"
    GENERAL = "general"


class ModelCapability(str, Enum):
    TEXT = "text"
    VISION = "vision"
    STREAMING = "streaming"
    REASONING = "reasoning"
    TOOL_CALLING = "tool_calling"
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"
    LONG_CONTEXT = "long_context"
    EMBEDDING = "embedding"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"


class SelectionStrategy(str, Enum):
    BALANCED = "balanced"
    LOWEST_COST = "lowest_cost"
    HIGHEST_QUALITY = "highest_quality"
    LOWEST_LATENCY = "lowest_latency"
    HIGHEST_AVAILABILITY = "highest_availability"
    HIGHEST_RELIABILITY = "highest_reliability"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"


@dataclass
class ModelMetadata:
    model_id: str
    provider_id: str
    capabilities: Set[ModelCapability]
    quality_score: float  # Scale 0.0 to 1.0
    input_cost_per_1k: float
    output_cost_per_1k: float
    context_window: int
    base_latency_ms: float
    supported_tasks: Set[TaskType] = field(default_factory=set)
    family: str = "general"


@dataclass
class SelectionCriteria:
    task_type: TaskType
    required_capabilities: Set[ModelCapability] = field(default_factory=set)
    strategy: SelectionStrategy = SelectionStrategy.BALANCED
    max_budget_per_1k: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_context_window: Optional[int] = None
    require_streaming: bool = False
    require_vision: bool = False
    require_tool_calling: bool = False
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSelectionResult:
    model_id: str
    provider_id: str
    strategy_used: str
    score: float
    estimated_cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ERROR HANDLING
# ============================================================================

class ModelSelectorError(Exception):
    """Base exception for Model Selector errors with rich metadata tracking."""
    def __init__(self, message: str, error_code: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "error_code": self.error_code,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict())


# ============================================================================
# MODEL ROUTER & PROVIDER STUBS FOR COMPATIBILITY (INTEGRATION GUARANTEE)
# ============================================================================

class MockProviderManager:
    """Interface wrapper matching studio/providers/provider_manager.py footprint."""
    def __init__(self) -> None:
        self._providers: Dict[str, Any] = {}

    def get_provider_status(self, provider_id: str) -> Dict[str, Any]:
        return {"status": "healthy", "current_load": 0, "priority": 1}

    def is_provider_available(self, provider_id: str) -> bool:
        return True


class MockModelRegistry:
    """Interface wrapper matching registry footprints inside ModelRouter or engines."""
    def __init__(self) -> None:
        self._models: Dict[str, ModelMetadata] = {}

    def register_model(self, model: ModelMetadata) -> None:
        self._models[f"{model.provider_id}/{model.model_id}"] = model

    def get_all_models(self) -> List[ModelMetadata]:
        return list(self._models.values())


# ============================================================================
# CORE MODEL SELECTOR ENGINE
# ============================================================================

class ModelSelector:
    """
    Intelligent Model Selector core responsible for dynamic tier routing,
    multi-parametric optimization metrics, caching, and failover analysis.
    """
    def __init__(
        self,
        provider_manager: Any,
        model_registry: Any,
        health_monitor: Optional[Any] = None,
        telemetry_manager: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.provider_manager = provider_manager or MockProviderManager()
        self.model_registry = model_registry or MockModelRegistry()
        self.health_monitor = health_monitor
        self.telemetry_manager = telemetry_manager
        self.config = config or {}

        # Thread safety primitive
        self._lock = asyncio.Lock()

        # Dynamic strategy and structure tables
        self._custom_strategies: Dict[str, Callable[[List[ModelMetadata], SelectionCriteria], List[ModelMetadata]]] = {}
        self._task_mappings: Dict[TaskType, List[str]] = {}
        self._model_aliases: Dict[str, str] = {}
        self._provider_aliases: Dict[str, str] = {}
        
        # Round-robin trackers
        self._round_robin_index: Dict[str, int] = {}

        # Cache infrastructure
        self._cache: Dict[str, Tuple[ModelSelectionResult, float]] = {}
        self._cache_ttl = float(self.config.get("cache_ttl_seconds", 30))

        # Core operational statistics
        self._stats = {
            "total_selections": 0,
            "provider_usage": {},
            "model_usage": {},
            "selection_failures": 0,
            "average_latency": 0.0,
            "estimated_costs": 0.0,
            "success_rate": 1.0,
            "successful_selections": 0
        }

        # Setup standard baseline mappings
        self._initialize_default_mappings()

    def _initialize_default_mappings(self) -> None:
        self._task_mappings = {
            TaskType.CODING: ["code", "reasoning"],
            TaskType.CODE_REVIEW: ["code", "reasoning"],
            TaskType.STORY_GENERATION: ["creative", "general"],
            TaskType.CREATIVE: ["creative"],
            TaskType.REASONING: ["reasoning", "advanced"],
            TaskType.PLANNING: ["reasoning"],
            TaskType.VISION: ["vision"],
            TaskType.IMAGE_ANALYSIS: ["vision"],
            TaskType.TRANSLATION: ["multilingual", "general"],
        }

    # ============================================================================
    # PUBLIC API METHODS
    # ============================================================================

    async def select_model(self, criteria: SelectionCriteria) -> ModelSelectionResult:
        """Selects optimal model given runtime evaluation parameters."""
        async with self._lock:
            try:
                self.validate_request(criteria)
                
                # Check target cache
                cache_key = f"{criteria.task_type.value}:{criteria.strategy.value}:{hash(frozenset(criteria.required_capabilities))}"
                if cache_key in self._cache:
                    result, timestamp = self._cache[cache_key]
                    if (datetime.now().timestamp() - timestamp) < self._cache_ttl:
                        await self._update_statistics(result.model_id, result.provider_id, True, 0.0, result.estimated_cost)
                        return result

                candidates = await self._filter_candidates(criteria)
                if not candidates:
                    raise ModelSelectorError("No viable models match the target criteria requirements.", "NO_CANDIDATES_FOUND")

                ranked_candidates = await self._apply_strategy(candidates, criteria)
                best_candidate, score = ranked_candidates[0]

                est_cost = self.estimate_cost(best_candidate, 1000, 1000)
                
                result = ModelSelectionResult(
                    model_id=best_candidate.model_id,
                    provider_id=best_candidate.provider_id,
                    strategy_used=criteria.strategy.value,
                    score=score,
                    estimated_cost=est_cost,
                    metadata={
                        "context_window": best_candidate.context_window,
                        "selection_time": datetime.now(timezone.utc).isoformat()
                    }
                )

                self._cache[cache_key] = (result, datetime.now().timestamp())
                await self._update_statistics(result.model_id, result.provider_id, True, best_candidate.base_latency_ms, est_cost)
                return result

            except Exception as e:
                await self._update_statistics("", "", False, 0.0, 0.0)
                self._log_error("Selection cycle failed", e)
                if isinstance(e, ModelSelectorError):
                    raise e
                raise ModelSelectorError(f"Internal selection failure: {str(e)}", "SELECTION_FAILURE")

    async def select_best_model(self, task_type: TaskType, capabilities: Set[ModelCapability]) -> str:
        """Helper shorthand wrapper to pull directly target optimal structural string id."""
        criteria = SelectionCriteria(task_type=task_type, required_capabilities=capabilities)
        res = await self.select_model(criteria)
        return f"{res.provider_id}/{res.model_id}"

    async def select_provider(self, criteria: SelectionCriteria) -> str:
        """Identifies target operational optimal structural provider base string."""
        res = await self.select_model(criteria)
        return res.provider_id

    async def select_candidates(self, criteria: SelectionCriteria) -> List[ModelMetadata]:
        """Provides raw capability-matched, unfiltered variants list."""
        async with self._lock:
            return await self._filter_candidates(criteria)

    def estimate_cost(self, model: ModelMetadata, input_tokens: int, output_tokens: int) -> float:
        """Calculates deterministic pricing tier structures."""
        in_cost = (input_tokens / 1000.0) * model.input_cost_per_1k
        out_cost = (output_tokens / 1000.0) * model.output_cost_per_1k
        return in_cost + out_cost

    def estimate_tokens(self, text: str, context_type: str = "text") -> int:
        """Determines context boundaries metrics sizing."""
        if not text:
            return 0
        if context_type == "vision":
            return 85  # Deterministic placeholder factor per asset block base standard
        return len(text.split()) * 4 // 3

    async def rank_models(self, criteria: SelectionCriteria) -> List[Tuple[ModelMetadata, float]]:
        """Provides fully quantified prioritization mapping list elements."""
        async with self._lock:
            candidates = await self._filter_candidates(criteria)
            return await self._apply_strategy(candidates, criteria)

    def validate_request(self, criteria: SelectionCriteria) -> None:
        """Evaluates integrity matrix states before pipeline entry."""
        if not isinstance(criteria.task_type, TaskType):
            raise ModelSelectorError("Invalid targeted task variant type.", "INVALID_TASK_TYPE")
        if criteria.max_budget_per_1k is not None and criteria.max_budget_per_1k < 0:
            raise ModelSelectorError("Budget limits cannot be set negative values.", "INVALID_BUDGET")

    def get_available_models(self) -> List[str]:
        """Gathers available unique unified fully qualified identities."""
        models = self.model_registry.get_all_models()
        return [f"{m.provider_id}/{m.model_id}" for m in models if self.provider_manager.is_provider_available(m.provider_id)]

    def get_available_providers(self) -> List[str]:
        """Extracts uniquely isolated context providers identifiers."""
        models = self.model_registry.get_all_models()
        return list({m.provider_id for m in models if self.provider_manager.is_provider_available(m.provider_id)})

    def register_selection_strategy(self, name: str, strategy_fn: Callable[[List[ModelMetadata], SelectionCriteria], List[ModelMetadata]]) -> None:
        """Registers custom strategy callback functions to selection tables."""
        self._custom_strategies[name] = strategy_fn

    def unregister_selection_strategy(self, name: str) -> None:
        """Removes an active custom selection strategy function."""
        if name in self._custom_strategies:
            del self._custom_strategies[name]

    def register_task_mapping(self, task_type: TaskType, families: List[str]) -> None:
        """Binds customized mapping paths across capability variants families."""
        self._task_mappings[task_type] = families

    def register_model_alias(self, alias: str, target_model_id: str) -> None:
        """Links unified transparent logical alias targets strings."""
        self._model_aliases[alias] = target_model_id

    def register_provider_alias(self, alias: str, target_provider_id: str) -> None:
        """Links provider layers configuration logical aliases maps."""
        self._provider_aliases[alias] = target_provider_id

    def create_snapshot(self) -> Dict[str, Any]:
        """Captures fully deterministic architectural system states maps."""
        return {
            "task_mappings": copy.deepcopy({k.value: v for k, v in self._task_mappings.items()}),
            "model_aliases": copy.deepcopy(self._model_aliases),
            "provider_aliases": copy.deepcopy(self._provider_aliases),
            "config": copy.deepcopy(self.config),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Re-injects prior snapshot matrices elements safely."""
        self._task_mappings = {TaskType(k): v for k, v in snapshot.get("task_mappings", {}).items()}
        self._model_aliases = copy.deepcopy(snapshot.get("model_aliases", {}))
        self._provider_aliases = copy.deepcopy(snapshot.get("provider_aliases", {}))
        self.config = copy.deepcopy(snapshot.get("config", {}))

    def export_configuration(self) -> str:
        """Serializes current active structural setup properties string configurations."""
        return json.dumps(self.create_snapshot(), indent=2)

    def import_configuration(self, config_str: str) -> None:
        """Parses active structures directly over functional configurations instances."""
        snapshot = json.loads(config_str)
        self.restore_snapshot(snapshot)

    def run_diagnostics(self) -> Dict[str, Any]:
        """Performs structural health sanity tracking check tasks."""
        all_models = self.model_registry.get_all_models()
        return {
            "status": "operational",
            "registered_models_count": len(all_models),
            "configured_aliases_count": len(self._model_aliases),
            "cached_elements_count": len(self._cache),
            "statistics": copy.deepcopy(self._stats)
        }

    def backup(self) -> Dict[str, Any]:
        """Alias target wrapping match for lifecycle persistence layers patterns."""
        return self.create_snapshot()

    def restore(self, backup_data: Dict[str, Any]) -> None:
        """Alias state target wrapping recovery lifecycle structures patterns."""
        self.restore_snapshot(backup_data)

    def shutdown(self) -> None:
        """Gracefully flushes open execution caches layers pools structures."""
        self._cache.clear()
        self._custom_strategies.clear()

    # ============================================================================
    # PRIVATE INTERNAL IMPLEMENTATION HOOKS
    # ============================================================================

    async def _filter_candidates(self, criteria: SelectionCriteria) -> List[ModelMetadata]:
        """Excludes sub-optimal models based on mandatory target parameters constraints."""
        all_models = self.model_registry.get_all_models()
        filtered: List[ModelMetadata] = []

        for model in all_models:
            # Resolve aliases
            resolved_model_id = self._resolve_aliases(model.model_id, self._model_aliases)
            resolved_provider_id = self._resolve_aliases(model.provider_id, self._provider_aliases)
            
            if not self.provider_manager.is_provider_available(resolved_provider_id):
                continue
            
            # Validate core structural capabilities
            if not self._validate_capabilities(model, criteria):
                continue
                
            # Filter criteria boundaries mappings validation checks
            if criteria.max_budget_per_1k is not None:
                if model.input_cost_per_1k > criteria.max_budget_per_1k:
                    continue
            if criteria.min_context_window is not None:
                if model.context_window < criteria.min_context_window:
                    continue
                    
            filtered.append(model)
            
        return filtered

    def _validate_capabilities(self, model: ModelMetadata, criteria: SelectionCriteria) -> bool:
        """Checks if a model possesses the minimum required capabilities."""
        if criteria.require_streaming and ModelCapability.STREAMING not in model.capabilities:
            return False
        if criteria.require_vision and ModelCapability.VISION not in model.capabilities:
            return False
        if criteria.require_tool_calling and ModelCapability.TOOL_CALLING not in model.capabilities and ModelCapability.FUNCTION_CALLING not in model.capabilities:
            return False
        for req in criteria.required_capabilities:
            if req not in model.capabilities:
                return False
        return True

    def _resolve_aliases(self, entity_id: str, alias_map: Dict[str, str]) -> str:
        """Resolves recursive definition reference identifiers loops protection tags."""
        visited = set()
        current = entity_id
        while current in alias_map:
            if current in visited:
                break # Cyclic protection match boundary exit hook sequence
            visited.add(current)
            current = alias_map[current]
        return current

    async def _apply_strategy(self, candidates: List[ModelMetadata], criteria: SelectionCriteria) -> List[Tuple[ModelMetadata, float]]:
        """Applies tactical algorithmic priority filters maps structures rankings."""
        strategy = criteria.strategy

        if strategy.value in self._custom_strategies:
            custom_fn = self._custom_strategies[strategy.value]
            custom_sorted = custom_fn(candidates, criteria)
            return [(m, 1.0 - (idx / len(custom_sorted))) for idx, m in enumerate(custom_sorted)]

        # Procedural matrix evaluators assignments switches implementation setup
        scored_candidates: List[Tuple[ModelMetadata, float]] = []

        if strategy == SelectionStrategy.RANDOM:
            return [(m, 1.0) for m in sorted(candidates, key=lambda x: choice([0, 1]))]

        if strategy == SelectionStrategy.ROUND_ROBIN:
            key = f"{criteria.task_type.value}"
            idx = self._round_robin_index.get(key, 0)
            next_idx = (idx + 1) % len(candidates) if candidates else 0
            self._round_robin_index[key] = next_idx
            # Pivot targeted indexing rotation alignment prioritization array items structures shift mapping layers
            rotated = candidates[idx:] + candidates[:idx]
            return [(m, 1.0 - (i / len(rotated))) for i, m in enumerate(rotated)]

        for model in candidates:
            score = await self._score_candidate(model, criteria)
            scored_candidates.append((model, score))

        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    async def _score_candidate(self, model: ModelMetadata, criteria: SelectionCriteria) -> float:
        """Assembles compound dynamic balance scores mappings calculations."""
        strategy = criteria.strategy
        
        # Calculate individual metric dimensions mappings base components configurations
        q_score = self._calculate_quality(model, criteria)
        l_score = self._calculate_latency(model)
        h_score = self._calculate_health(model)
        c_score = self._calculate_cost(model)
        r_score = self._calculate_reliability(model)

        # Baseline balanced priority distribution values assignments tracking configuration parameters mapping values
        w_quality, w_latency, w_health, w_cost, w_reliability = 0.25, 0.15, 0.20, 0.20, 0.20

        if strategy == SelectionStrategy.HIGHEST_QUALITY:
            w_quality, w_latency, w_health, w_cost, w_reliability = 0.60, 0.10, 0.10, 0.05, 0.15
        elif strategy == SelectionStrategy.LOWEST_COST:
            w_quality, w_latency, w_health, w_cost, w_reliability = 0.05, 0.10, 0.10, 0.60, 0.15
        elif strategy == SelectionStrategy.LOWEST_LATENCY:
            w_quality, w_latency, w_health, w_cost, w_reliability = 0.10, 0.60, 0.10, 0.05, 0.15
        elif strategy == SelectionStrategy.HIGHEST_AVAILABILITY or strategy == SelectionStrategy.LEAST_LOADED:
            w_quality, w_latency, w_health, w_cost, w_reliability = 0.10, 0.10, 0.60, 0.05, 0.15
        elif strategy == SelectionStrategy.HIGHEST_RELIABILITY:
            w_quality, w_latency, w_health, w_cost, w_reliability = 0.15, 0.10, 0.10, 0.05, 0.60

        # Structural functional balancing execution logic summation computation algorithms operations patterns elements
        total_score = (
            (q_score * w_quality) +
            (l_score * w_latency) +
            (h_score * w_health) +
            (c_score * w_cost) +
            (r_score * w_reliability)
        )
        return total_score

    def _calculate_quality(self, model: ModelMetadata, criteria: SelectionCriteria) -> float:
        """Determines context capability performance quality multipliers metrics scores configurations."""
        base_quality = model.quality_score
        
        # Boost if model family explicitly matches task preferred mappings rules configurations vectors properties
        preferred_families = self._task_mappings.get(criteria.task_type, [])
        if model.family in preferred_families:
            base_quality = min(1.0, base_quality + 0.15)
            
        if criteria.task_type in model.supported_tasks:
            base_quality = min(1.0, base_quality + 0.10)
            
        return base_quality

    def _calculate_latency(self, model: ModelMetadata) -> float:
        """Transforms expected structural processing durations scales values normalized weights metrics formats bounds."""
        # 0ms maps to optimal score 1.0; 5000ms maps downwards scales bounds limits 0.0 values baseline elements parameters
        return max(0.0, 1.0 - (model.base_latency_ms / 5000.0))

    def _calculate_health(self, model: ModelMetadata) -> float:
        """Extracts running status dynamics weights from real-time monitoring subcomponents contexts layers pools."""
        if self.health_monitor:
            try:
                status = self.health_monitor.get_model_health(model.model_id, model.provider_id)
                return float(status.get("score", 1.0))
            except Exception:
                return 0.8  # Degrade gracefully to safe conservative fallback defaults settings limits
        return 1.0

    def _calculate_cost(self, model: ModelMetadata) -> float:
        """Evaluates inverse pricing curves calculations scores properties parameters setups metrics boundaries frames."""
        combined_base_cost = model.input_cost_per_1k + model.output_cost_per_1k
        # Standardized normalization threshold baseline factors cost models evaluations configurations vectors parameters
        return max(0.0, 1.0 - (combined_base_cost / 0.10))

    def _calculate_reliability(self, model: ModelMetadata) -> float:
        """Calculates past telemetry accuracy records success patterns values dimensions variables formulas boundaries."""
        if self.telemetry_manager:
            try:
                metrics = self.telemetry_manager.get_metrics(model.model_id, model.provider_id)
                return float(metrics.get("success_rate", 1.0))
            except Exception:
                return 0.9
        
        # Default internal statistics calculations feedback loop integration fallback mechanisms tracking hooks setups elements
        model_key = f"{model.provider_id}/{model.model_id}"
        usages = self._stats["model_usage"].get(model_key, 0)
        if usages == 0:
            return 1.0
        return 1.0

    def _estimate_context_usage(self, prompt_tokens: int, max_tokens: int, context_window: int) -> float:
        """Evaluates model saturation threshold limits capacity safety metrics boundaries structures checks."""
        total_needed = prompt_tokens + max_tokens
        if total_needed > context_window:
            return 0.0
        return 1.0 - (total_needed / context_window)

    async def _update_statistics(self, model_id: str, provider_id: str, success: bool, latency: float, cost: float) -> None:
        """Updates internal telemetry profiling logs state indicators tables values elements maps properties."""
        self._stats["total_selections"] += 1
        if success:
            self._stats["successful_selections"] += 1
            model_key = f"{provider_id}/{model_id}"
            self._stats["model_usage"][model_key] = self._stats["model_usage"].get(model_key, 0) + 1
            self._stats["provider_usage"][provider_id] = self._stats["provider_usage"].get(provider_id, 0) + 1
            
            # Running exponential tracking updates calculations structures elements variables averages components setup formulas
            old_avg = self._stats["average_latency"]
            count = self._stats["successful_selections"]
            self._stats["average_latency"] = old_avg + ((latency - old_avg) / count)
            self._stats["estimated_costs"] += cost
        else:
            self._stats["selection_failures"] += 1

        total = self._stats["total_selections"]
        if total > 0:
            self._stats["success_rate"] = self._stats["successful_selections"] / total

    def _log_error(self, ctx: str, err: Exception) -> None:
        """Passes diagnostic system alerts directly downwards standard infrastructure logging adapters frameworks."""
        logger.error(f"[ModelSelector] Context error hook triggered: {ctx} | Internal Exception Trace: {str(err)}")

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        """Structured baseline tracking contextual outputs mapping adapters channels routing components structures execution paths."""
        logger.log(level, f"[ModelSelector] {msg}")
