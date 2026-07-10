import asyncio
import time
import uuid
import json
import logging
import random
import traceback
from datetime import datetime
from collections import deque
from typing import Any, Dict, List, Callable, Set, Optional, Tuple, Union

# Set up standard logger
logger = logging.getLogger("studio.router.model_router")

class ModelRouterError(Exception):
    """
    Enterprise-grade Custom Error class for Cabangile AI Studio Router.
    Supports granular system codes, structural metadata tracking, and operational categorization.
    """
    def __init__(self, message: str, code: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.stack = traceback.format_exc()

    def to_json(self) -> Dict[str, Any]:
        """Serializes the internal state configuration to a clean payload."""
        return {
            "name": "ModelRouterError",
            "code": self.code,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "stack": self.stack
        }


class EventEmitter:
    """A clean, asynchronous-capable Python equivalent to Node's EventEmitter."""
    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[..., Any]]] = {}

    def on(self, event: str, listener: Callable[..., Any]) -> None:
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        if event in self._listeners:
            for listener in self._listeners[event]:
                if asyncio.iscoroutinefunction(listener):
                    asyncio.create_task(listener(*args, **kwargs))
                else:
                    try:
                        listener(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in event listener for {event}: {e}")


class ModelRouter(EventEmitter):
    """
    Cabangile AI Studio Enterprise AI Orchestration Routing Engine.
    Complete production-ready implementation managing resilience, caching, throttles, and topologies.
    """
    def __init__(
        self,
        provider_manager: Any,
        model_registry: Any,
        health_monitor: Any = None,
        cost_tracker: Any = None,
        telemetry_manager: Any = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.provider_manager = provider_manager
        self.model_registry = model_registry
        self.health_monitor = health_monitor
        self.cost_tracker = cost_tracker
        self.telemetry_manager = telemetry_manager

        # Deep merge configuration rules and runtime properties
        base_config = {
            "defaultStrategy": "balanced",
            "maxFailoverAttempts": 3,
            "historyLimit": 2000,
            "tokenEstimationFactor": 4,
            "isOnline": True,
            "globalTimeoutMs": 30000,
            "defaultCacheTtlMs": 60000,
            "maxQueueSize": 5000,
            "queueTimeoutMs": 15000,
            "cooldownPeriodMs": 45000,
            "maxCircuitFailures": 5,
            "circuitBreakerResetTimeoutMs": 30000,
            "deduplicationWindowMs": 2000,
            "weights": { "health": 0.3, "cost": 0.2, "speed": 0.2, "quality": 0.15, "reliability": 0.15 }
        }
        if config:
            base_config.update(config)
        self.config = base_config

        # System Topology Registers
        self.strategies: Dict[str, Callable[..., Any]] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.middlewares: Dict[str, List[Callable[..., Any]]] = { "before": [], "after": [] }
        self.plugins: Dict[str, Any] = {}
        self.model_aliases: Dict[str, str] = {}
        self.provider_aliases: Dict[str, str] = {}
        self.fallback_chains: Dict[str, List[str]] = {}

        # Architectural Resiliency & Storage Subsystems
        self.history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.round_robin_pointers: Dict[str, int] = {}
        self.load_metrics: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.provider_cooldowns: Dict[str, float] = {}
        self.request_cache: Dict[str, Dict[str, Any]] = {}
        self.request_queue: deque = deque()
        self.deduplication_map: Dict[str, asyncio.Task] = {}
        self.active_request_controllers: Dict[str, asyncio.Event] = {}
        self.is_shutting_down = False

        # Running Analytical Counters
        self.stats = {
            "totalRequests": 0,
            "routedRequests": 0,
            "failedRequests": 0,
            "failovers": 0,
            "totalRoutingTimeMs": 0.0,
            "totalEstimatedCost": 0.0,
            "cacheHits": 0,
            "queueCount": 0,
            "circuitBreaks": 0,
            "providerUsage": {},
            "modelUsage": {},
            "strategyUsage": {}
        }

        # Initialize Structural Subsystems
        self._initialize_builtin_strategies()
        self._cleanup_task: Optional[asyncio.Task] = None
        # Start cleanup task reactively or via explicit running event loop context
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                self._cleanup_task = loop.create_task(self._start_background_cleanup_tasks())
        except RuntimeError:
            pass

    # =========================================================================
    # CORE API OVERRIDES & IMPLEMENTATIONS
    # =========================================================================

    async def route(self, request: dict, options: Optional[dict] = None) -> dict:
        """Primary Entrypoint. Routes an incoming model request safely through the resilience pipeline."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._start_background_cleanup_tasks())

        if self.is_shutting_down:
            raise ModelRouterError("System is undergoing graceful shutdown routines.", "ERR_ROUTER_SHUTDOWN")

        options = options or {}
        request_id = request.get("requestId") or str(uuid.uuid4())
        start_time = time.perf_counter()
        self.stats["totalRequests"] += 1

        # Context execution metadata envelope
        ctx = {
            "requestId": request_id,
            "request": self._resolve_aliases(request),
            "options": options,
            "attempts": 0,
            "history": [],
            "timeoutToken": None,
            "internalSignal": None
        }

        # Request Deduplication layer
        dedup_key = self._generate_deduplication_key(ctx["request"])
        if dedup_key and dedup_key in self.deduplication_map:
            self.stats["cacheHits"] += 1
            return await self.deduplication_map[dedup_key]

        # Cache lookup matrix layer
        if ctx["request"].get("cacheKey") or options.get("useCache"):
            cached = self._lookup_cache(ctx["request"])
            if cached:
                self.stats["cacheHits"] += 1
                return cached

        self.emit("routingStarted", {"requestId": request_id, "request": ctx["request"]})

        # Request structural queues allocation boundaries
        if self._should_queue_request(ctx["request"]):
            await self._enqueue_request(ctx)

        # Set up lifecycle cancellation signals
        timeout_task, abort_event = self._setup_cancellation_context(ctx)

        async def _execute_pipeline():
            try:
                await self.validate_request(ctx["request"])

                # Before Middleware Hook
                for middleware in self.middlewares["before"]:
                    if asyncio.iscoroutinefunction(middleware):
                        await middleware(ctx)
                    else:
                        middleware(ctx)

                # Apply external lifecycle plugins
                for plugin_name, plugin in self.plugins.items():
                    if hasattr(plugin, "before_route") and callable(plugin.before_route):
                        if asyncio.iscoroutinefunction(plugin.before_route):
                            await plugin.before_route(ctx)
                        else:
                            plugin.before_route(ctx)

                routing_result = await self._execute_routing_loop_with_failover(ctx)

                latency = (time.perf_counter() - start_time) * 1000.0
                self.stats["routedRequests"] += 1
                self.stats["totalRoutingTimeMs"] += latency
                self.stats["totalEstimatedCost"] += routing_result["costEstimation"]["totalCost"]

                self._update_statistics(routing_result, latency)

                final_output = {
                    "requestId": request_id,
                    "provider": routing_result["provider"]["id"],
                    "model": routing_result["model"]["id"],
                    "strategy": routing_result["strategy"],
                    "costEstimation": routing_result["costEstimation"],
                    "latencyMs": latency,
                    "history": ctx["history"],
                    "isStream": bool(ctx["request"].get("streaming") or routing_result["model"].get("capabilities", {}).get("streaming"))
                }

                # After Middleware Hook
                for middleware in self.middlewares["after"]:
                    if asyncio.iscoroutinefunction(middleware):
                        await middleware(final_output)
                    else:
                        middleware(final_output)

                for plugin_name, plugin in self.plugins.items():
                    if hasattr(plugin, "after_route") and callable(plugin.after_route):
                        if asyncio.iscoroutinefunction(plugin.after_route):
                            await plugin.after_route(final_output, ctx)
                        else:
                            plugin.after_route(final_output, ctx)

                # Cache and deduplicate results when applicable
                self._populate_cache_and_dedup(ctx["request"], final_output, dedup_key)

                self.emit("routingCompleted", final_output)
                return final_output

            except Exception as error:
                self.stats["failedRequests"] += 1
                router_error = error if isinstance(error, ModelRouterError) else ModelRouterError(
                    str(error), "ERR_ROUTING_FAILED", {"originalError": str(error), "requestId": request_id}
                )
                self.emit("routingFailed", {"requestId": request_id, "error": router_error.to_json()})
                self._log_error(router_error)
                raise router_error
            finally:
                self._clear_cancellation_context(ctx, timeout_task)

        promise_execution = asyncio.create_task(_execute_pipeline())

        if dedup_key and dedup_key not in self.deduplication_map:
            self.deduplication_map[dedup_key] = promise_execution

        return await promise_execution

    async def route_batch(self, requests: List[dict], options: Optional[dict] = None) -> List[dict]:
        """Concurrently processes collections of structured requests."""
        if not isinstance(requests, list):
            raise ModelRouterError("Batch operations require an array structure payload.", "ERR_INVALID_BATCH")

        async def safe_route(req):
            try:
                return await self.route(req, options)
            except Exception as err:
                if isinstance(err, ModelRouterError):
                    return {"error": err.to_json()}
                return {"error": {"message": str(err), "code": "ERR_ROUTING_FAILED"}}

        return await asyncio.gather(*(safe_route(req) for req in requests))

    # =========================================================================
    # CORE INTERNAL ROUTING & FAILOVER EXECUTION PIPELINE
    # =========================================================================

    async def _execute_routing_loop_with_failover(self, ctx: dict) -> dict:
        max_attempts = self.config["maxFailoverAttempts"]
        excluded_providers: Set[str] = set()

        while ctx["attempts"] < max_attempts:
            if ctx["internalSignal"] and ctx["internalSignal"].is_set():
                raise ModelRouterError("Transaction execution lifecycle aborted by caller.", "ERR_REQUEST_CANCELLED")

            ctx["attempts"] += 1
            candidate = None

            try:
                candidate = await self._select_candidate(ctx["request"], excluded_providers)
            except Exception as select_error:
                candidate = await self._evaluate_fallback_chain(ctx["request"], excluded_providers)
                if not candidate:
                    raise select_error

            p_id = candidate["provider"]["id"]
            self.emit("providerSelected", {
                "requestId": ctx["requestId"], "provider": p_id, "model": candidate["model"]["id"], "attempt": ctx["attempts"]
            })

            if self._is_circuit_open(p_id) or self._is_provider_cooldown_active(p_id) or not self._check_rate_and_quota_limits(candidate):
                self.emit("providerUnavailable", {"requestId": ctx["requestId"], "provider": p_id})
                excluded_providers.add(p_id)
                continue

            try:
                self._acquire_concurrent_slot(p_id)

                is_healthy = await self._verify_candidate_health(candidate["provider"])
                if not is_healthy:
                    self._record_circuit_failure(p_id)
                    raise ModelRouterError(f"Provider validation reported unhealthy states: {p_id}", "ERR_PROVIDER_UNHEALTHY")

                self._reset_circuit_breaker(p_id)

                ctx["history"].append({
                    "attempt": ctx["attempts"],
                    "provider": p_id,
                    "model": candidate["model"]["id"],
                    "strategy": candidate["strategy"],
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })

                self._record_history_log({
                    "requestId": ctx["requestId"],
                    "provider": p_id,
                    "model": candidate["model"]["id"],
                    "strategy": candidate["strategy"],
                    "cost": candidate["costEstimation"]["totalCost"],
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })

                return candidate

            except Exception as execution_error:
                self._record_circuit_failure(p_id)
                self._activate_cooldown_timer(p_id)
                self._release_concurrent_slot(p_id)

                self.emit("providerRejected", {"requestId": ctx["requestId"], "error": str(execution_error)})

                if ctx["attempts"] >= max_attempts:
                    raise ModelRouterError(
                        f"Execution context chain exhausted max failover threshold limits ({max_attempts}).",
                        "ERR_FAILOVER_EXHAUSTED", {"history": ctx["history"]}
                    )

                self.stats["failovers"] += 1
                self.emit("failoverStarted", {"requestId": ctx["requestId"], "currentAttempt": ctx["attempts"]})

        raise ModelRouterError("Routing engines completely failed to allocate functional clusters.", "ERR_NO_ROUTE_AVAILABLE")

    async def _select_candidate(self, request: dict, excluded_providers: Set[str]) -> dict:
        all_models = await self.get_available_models()
        all_providers = await self.get_available_providers()

        rule_action = self._evaluate_rules(request)
        targeted_model_name = request.get("model") or (rule_action.get("model") if rule_action else None)
        targeted_provider_name = rule_action.get("provider") if rule_action else None

        candidates = []

        for model in all_models:
            if targeted_model_name and model.get("id") != targeted_model_name and model.get("family") != targeted_model_name:
                continue

            for provider in all_providers:
                if provider.get("id") in excluded_providers:
                    continue
                if targeted_provider_name and provider.get("id") != targeted_provider_name:
                    continue

                supported_models = provider.get("supportedModels", [])
                if model.get("id") not in supported_models and provider.get("modelId") != model.get("id"):
                    continue

                if request.get("capabilities"):
                    supports_all = all(
                        model.get("capabilities", {}).get(cap) or provider.get("capabilities", {}).get(cap)
                        for cap in request["capabilities"]
                    )
                    if not supports_all:
                        continue

                if request.get("vision") and not model.get("capabilities", {}).get("vision"):
                    continue
                if request.get("streaming") and not model.get("capabilities", {}).get("streaming"):
                    continue

                cost_estimation = await self.estimate_cost(request, model)

                if model.get("contextWindow") and cost_estimation["totalTokens"] > model["contextWindow"]:
                    continue
                if not self._validate_budget_enforcement(cost_estimation):
                    continue

                candidates.append({"provider": provider, "model": model, "cost_estimation": cost_estimation})

        if not candidates:
            raise ModelRouterError("No operational providers matched specifications.", "ERR_CAPABILITY_MISMATCH")

        requested_strategy = str(request.get("strategy") or self.config["defaultStrategy"]).lower().replace(" ", "")
        strategy_fn = self.strategies.get(requested_strategy) or self.strategies.get("balanced")

        if not strategy_fn:
            raise ModelRouterError("Strategy computation module failed to yield valid context targets.", "ERR_STRATEGY_FAULT")

        selected = await strategy_fn(candidates, {
            "roundRobinPointers": self.round_robin_pointers,
            "loadMetrics": self.load_metrics,
            "weights": self.config["weights"],
            "history": self.history
        })

        if not selected:
            raise ModelRouterError("Strategy computation module failed to yield valid context targets.", "ERR_STRATEGY_FAULT")

        return {
            "provider": selected["provider"],
            "model": selected["model"],
            "costEstimation": selected["cost_estimation"],
            "strategy": requested_strategy
        }

    # =========================================================================
    # INTEGRATED STRATEGIES & MATHEMATICAL SCORING MATRICES
    # =========================================================================

    def _initialize_builtin_strategies(self) -> None:
        async def lowest_cost(candidates, ctx):
            return sorted(candidates, key=lambda c: c["cost_estimation"]["totalCost"])[0]
        self.register_strategy("lowestcost", lowest_cost)

        async def fastest_response(candidates, ctx):
            return sorted(candidates, key=lambda c: ctx["loadMetrics"].get(c["provider"]["id"], {}).get("activeRequests", 0))[0]
        self.register_strategy("fastestresponse", fastest_response)

        async def highest_quality(candidates, ctx):
            return sorted(candidates, key=lambda c: c["model"].get("qualityScore", 0), reverse=True)[0]
        self.register_strategy("highestquality", highest_quality)

        async def highest_availability(candidates, ctx):
            return sorted(candidates, key=lambda c: c["provider"].get("availabilityScore", 0), reverse=True)[0]
        self.register_strategy("highestavailability", highest_availability)

        async def lowest_latency(candidates, ctx):
            return sorted(candidates, key=lambda c: ctx["loadMetrics"].get(c["provider"]["id"], {}).get("avgLatency", 0.0))[0]
        self.register_strategy("lowestlatency", lowest_latency)

        async def highest_reliability(candidates, ctx):
            return sorted(candidates, key=lambda c: c["provider"].get("reliabilityScore", 0), reverse=True)[0]
        self.register_strategy("highestreliability", highest_reliability)

        async def random_strategy(candidates, ctx):
            return random.choice(candidates)
        self.register_strategy("random", random_strategy)

        async def least_loaded(candidates, ctx):
            return sorted(candidates, key=lambda c: ctx["loadMetrics"].get(c["provider"]["id"], {}).get("activeRequests", 0))[0]
        self.register_strategy("leastloaded", least_loaded)

        async def round_robin(candidates, ctx):
            key = "|".join(sorted([c["provider"]["id"] for c in candidates]))
            idx = ctx["roundRobinPointers"].get(key, 0)
            if idx >= len(candidates):
                idx = 0
            chosen = candidates[idx]
            ctx["roundRobinPointers"][key] = idx + 1
            return chosen
        self.register_strategy("roundrobin", round_robin)

        async def balanced(candidates, ctx):
            top_candidate = None
            highest_score = float("-inf")

            for candidate in candidates:
                p_id = candidate["provider"]["id"]
                metrics = ctx["loadMetrics"].get(p_id, {"activeRequests": 0, "successCount": 1, "errorCount": 0})

                total_requests = metrics.get("successCount", 1) + metrics.get("errorCount", 0)
                if total_requests == 0:
                    total_requests = 1
                reliability_factor = metrics.get("successCount", 1) / total_requests
                provider_priority_weight = candidate["provider"].get("priorityWeight", 1.0)

                health_score = self._calculate_dynamic_health_score(p_id) * reliability_factor
                cost_score = 1.0 / (1.0 + float(candidate["cost_estimation"]["totalCost"] or 0.0))
                load_score = 1.0 / (1.0 + float(metrics.get("activeRequests", 0)))
                quality_score = float(candidate["model"].get("qualityScore", 50)) / 100.0

                past_performance_adjustment = self._calculate_adaptive_history_weight(p_id, ctx["history"])

                score = (
                    (health_score * ctx["weights"]["health"]) +
                    (cost_score * ctx["weights"]["cost"]) +
                    (load_score * ctx["weights"]["speed"]) +
                    (quality_score * ctx["weights"]["quality"]) +
                    (past_performance_adjustment * ctx["weights"]["reliability"])
                ) * provider_priority_weight

                if score > highest_score:
                    highest_score = score
                    top_candidate = candidate

            return top_candidate or candidates[0]
        self.register_strategy("balanced", balanced)

    # =========================================================================
    # FINANCIALS, TOKEN ESTIMATION & ADMINISTRATIVE POLICIES
    # =========================================================================

    async def estimate_tokens(self, request: dict) -> dict:
        """Resolves text inputs, arrays, and token allocations safely."""
        prompt_tokens = 0
        user_input = request.get("input") or request.get("prompt") or ""
        factor = self.config["tokenEstimationFactor"]

        if isinstance(user_input, str):
            prompt_tokens = int(len(user_input) / factor) + (1 if len(user_input) % factor > 0 else 0)
        elif isinstance(user_input, list):
            for msg in user_input:
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                length = len(content) if isinstance(content, str) else len(json.dumps(content))
                prompt_tokens += (int(length / factor) + (1 if length % factor > 0 else 0)) + 4
        elif isinstance(user_input, dict):
            length = len(json.dumps(user_input))
            prompt_tokens = int(length / factor) + (1 if length % factor > 0 else 0)

        estimated_output_tokens = request.get("estimatedOutputSize") or request.get("maxTokens") or 256
        return {
            "promptTokens": prompt_tokens,
            "estimatedOutputTokens": estimated_output_tokens,
            "totalTokens": prompt_tokens + estimated_output_tokens
        }

    async def estimate_cost(self, request: dict, model_spec: dict) -> dict:
        """Computes billing projections matching configurations."""
        tokens = await self.estimate_tokens(request)
        pricing = model_spec.get("pricing", {"inputPerToken": 0.0, "outputPerToken": 0.0})
        prompt_cost = tokens["promptTokens"] * pricing.get("inputPerToken", 0.0)
        completion_cost = tokens["estimatedOutputTokens"] * pricing.get("outputPerToken", 0.0)

        return {
            "promptTokens": tokens["promptTokens"],
            "estimatedOutputTokens": tokens["estimatedOutputTokens"],
            "totalTokens": tokens["totalTokens"],
            "promptCost": prompt_cost,
            "completionCost": completion_cost,
            "totalCost": prompt_cost + completion_cost
        }

    def _validate_budget_enforcement(self, cost_estimation: dict) -> bool:
        if self.cost_tracker:
            if hasattr(self.cost_tracker, "checkBudgetLimit") and not self.cost_tracker.checkBudgetLimit(cost_estimation["totalCost"]):
                return False
            if hasattr(self.cost_tracker, "checkTokenBudget") and not self.cost_tracker.checkTokenBudget(cost_estimation["totalTokens"]):
                return False
        return True

    async def validate_request(self, request: dict) -> None:
        """Validates structure requirements before entering the optimization loop."""
        if not request or ("input" not in request and "prompt" not in request):
            raise ModelRouterError("Inbound routing request payload cannot be empty.", "ERR_MALFORMED_REQUEST")

    # =========================================================================
    # RESILIENCY SUBSYSTEMS: CIRCUIT BREAKERS, COOLDOWNS & ACCELERATION
    # =========================================================================

    def _is_circuit_open(self, provider_id: str) -> bool:
        breaker = self.circuit_breakers.get(provider_id)
        if not breaker or breaker["state"] == "CLOSED":
            return False

        if breaker["state"] == "OPEN":
            if (time.time() * 1000.0) - breaker["lastFailureTime"] > self.config["circuitBreakerResetTimeoutMs"]:
                breaker["state"] = "HALF_OPEN"
                self._log_structured("Circuit enters verification state.", "INFO", {"providerId": provider_id})
                return False
            return True
        return False

    def _record_circuit_failure(self, provider_id: str) -> None:
        if provider_id not in self.circuit_breakers:
            self.circuit_breakers[provider_id] = {"failures": 0, "state": "CLOSED", "lastFailureTime": 0.0}
        breaker = self.circuit_breakers[provider_id]
        breaker["failures"] += 1
        breaker["lastFailureTime"] = time.time() * 1000.0

        if breaker["failures"] >= self.config["maxCircuitFailures"] and breaker["state"] != "OPEN":
            breaker["state"] = "OPEN"
            self.stats["circuitBreaks"] += 1
            self.emit("providerUnavailable", {"providerId": provider_id, "reason": "CIRCUIT_BREAKER_TRIGGERED"})
            self._log_structured("Circuit breaker activated for node.", "WARN", {"providerId": provider_id, "failures": breaker["failures"]})

    def _reset_circuit_breaker(self, provider_id: str) -> None:
        self.circuit_breakers[provider_id] = {"failures": 0, "state": "CLOSED", "lastFailureTime": 0.0}

    def _activate_cooldown_timer(self, provider_id: str) -> None:
        self.provider_cooldowns[provider_id] = (time.time() * 1000.0) + self.config["cooldownPeriodMs"]

    def _is_provider_cooldown_active(self, provider_id: str) -> bool:
        active_until = self.provider_cooldowns.get(provider_id)
        if not active_until:
            return False
        if (time.time() * 1000.0) > active_until:
            self.provider_cooldowns.pop(provider_id, None)
            return False
        return True

    def _calculate_dynamic_health_score(self, provider_id: str) -> float:
        if not self.health_monitor or not hasattr(self.health_monitor, "getScore"):
            return 1.0
        return float(self.health_monitor.getScore(provider_id))

    def _calculate_adaptive_history_weight(self, provider_id: str, history_list: List[dict]) -> float:
        if not history_list:
            return 1.0
        items = [h for h in history_list if h.get("provider") == provider_id][-10:]
        if not items:
            return 1.0
        successes = len([i for i in items if i.get("success")])
        return successes / len(items)

    def _check_rate_and_quota_limits(self, candidate: dict) -> bool:
        p_id = candidate["provider"]["id"]
        if self.provider_manager and hasattr(self.provider_manager, "checkQuota") and not self.provider_manager.checkQuota(p_id):
            return False

        metrics = self.load_metrics.get(p_id)
        if metrics and candidate["provider"].get("maxConcurrentRequests") and metrics["activeRequests"] >= candidate["provider"]["maxConcurrentRequests"]:
            return False
        return True

    def _acquire_concurrent_slot(self, provider_id: str) -> None:
        self._increment_active_requests(provider_id)

    def _release_concurrent_slot(self, provider_id: str) -> None:
        metrics = self.load_metrics.get(provider_id)
        if metrics:
            metrics["activeRequests"] = max(0, metrics["activeRequests"] - 1)

    # =========================================================================
    # CACHING, DEDUPLICATION, AND QUEUE MANAGEMENT SUBSYSTEMS
    # =========================================================================

    def _generate_deduplication_key(self, request: dict) -> Optional[str]:
        user_input = request.get("input") or request.get("prompt")
        if not user_input:
            return None
        try:
            source = user_input if isinstance(user_input, str) else json.dumps(user_input)
            return f"{request.get('model', 'default')}:{source}"
        except Exception:
            return None

    def _lookup_cache(self, request: dict) -> Optional[dict]:
        key = request.get("cacheKey") or self._generate_deduplication_key(request)
        if not key:
            return None
        record = self.request_cache.get(key)
        if not record:
            return None
        if (time.time() * 1000.0) > record["expiresAt"]:
            self.request_cache.pop(key, None)
            return None
        return record["payload"]

    def _populate_cache_and_dedup(self, request: dict, output: dict, dedup_key: Optional[str]) -> None:
        key = request.get("cacheKey") or dedup_key
        if key:
            ttl = request.get("cacheTtlMs") or self.config["defaultCacheTtlMs"]
            self.request_cache[key] = {"payload": output, "expiresAt": (time.time() * 1000.0) + ttl}
        if dedup_key:
            def _remove_dedup():
                self.deduplication_map.pop(dedup_key, None)
            asyncio.get_running_loop().call_later(self.config["deduplicationWindowMs"] / 1000.0, _remove_dedup)

    def _should_queue_request(self, request: dict) -> bool:
        if request.get("bypassQueue"):
            return False
        current_active_count = sum(m["activeRequests"] for m in self.load_metrics.values())
        return current_active_count > (self.config["maxQueueSize"] / 2)

    async def _enqueue_request(self, ctx: dict) -> None:
        if len(self.request_queue) >= self.config["maxQueueSize"]:
            raise ModelRouterError("System-wide operational queues are completely saturated.", "ERR_QUEUE_SATURATED")

        self.stats["queueCount"] += 1
        future = asyncio.get_running_loop().create_future()

        async def _timeout_trigger():
            await asyncio.sleep(self.config["queueTimeoutMs"] / 1000.0)
            if not future.done():
                try:
                    self.request_queue.remove((ctx, future))
                except ValueError:
                    pass
                future.set_exception(ModelRouterError("Queue retention time limit exceeded.", "ERR_QUEUE_TIMEOUT"))

        timeout_task = asyncio.create_task(_timeout_trigger())
        self.request_queue.append((ctx, future))
        self._process_queue()

        try:
            await future
        finally:
            timeout_task.cancel()

    def _process_queue(self) -> None:
        if not self.request_queue:
            return
        ctx, future = self.request_queue.popleft()
        if not future.done():
            future.set_result(True)

    # =========================================================================
    # LIFECYCLE CANCELLATION & BACKGROUND HOUSEKEEPING AUTOMATIONS
    # =========================================================================

    def _setup_cancellation_context(self, ctx: dict) -> Tuple[asyncio.Task, asyncio.Event]:
        abort_event = asyncio.Event()
        self.active_request_controllers[ctx["requestId"]] = abort_event
        ctx["internalSignal"] = abort_event

        external_signal = ctx["options"].get("signal")
        if external_signal and hasattr(external_signal, "add_done_callback"):
            external_signal.add_done_callback(lambda _: abort_event.set())

        async def _timeout_run():
            timeout_ms = ctx["options"].get("timeoutMs") or self.config["globalTimeoutMs"]
            await asyncio.sleep(timeout_ms / 1000.0)
            abort_event.set()
            self.emit("routingFailed", {"requestId": ctx["requestId"], "reason": "TIMEOUT_TRIGGERED"})

        timeout_task = asyncio.create_task(_timeout_run())
        return timeout_task, abort_event

    def _clear_cancellation_context(self, ctx: dict, timeout_task: asyncio.Task) -> None:
        timeout_task.cancel()
        self.active_request_controllers.pop(ctx["requestId"], None)

    async def _start_background_cleanup_tasks(self) -> None:
        try:
            while not self.is_shutting_down:
                await asyncio.sleep(15.0)
                now = time.time() * 1000.0

                # Clean expired execution caches
                expired_cache_keys = [k for k, v in self.request_cache.items() if now > v["expiresAt"]]
                for k in expired_cache_keys:
                    self.request_cache.pop(k, None)

                # Clean stale breaker profiles
                expired_breaker_keys = [
                    k for k, v in self.circuit_breakers.items()
                    if v["state"] == "CLOSED" and now - v["lastFailureTime"] > 300000
                ]
                for k in expired_breaker_keys:
                    self.circuit_breakers.pop(k, None)
        except asyncio.CancelledError:
            pass

    # =========================================================================
    # DYNAMIC COMPONENT PLUGINS & TOPOLOGY OVERRIDES
    # =========================================================================

    def register_plugin(self, name: str, plugin_object: Any) -> None:
        """Registers a plugin hook to inject custom enterprise validation rules."""
        self.plugins[name] = plugin_object

    def register_fallback_chain(self, source_provider_id: str, fallback_providers_list: List[str]) -> None:
        """Configures cross-region fallback maps for active failover paths."""
        self.fallback_chains[source_provider_id] = fallback_providers_list

    async def _evaluate_fallback_chain(self, request: dict, excluded_providers: Set[str]) -> Optional[dict]:
        target_provider = request.get("provider")
        if not target_provider or target_provider not in self.fallback_chains:
            return None

        chain = self.fallback_chains[target_provider]
        for provider_id in chain:
            if provider_id in excluded_providers:
                continue

            mutated_request = request.copy()
            mutated_request["provider"] = provider_id
            try:
                return await self._select_candidate(mutated_request, excluded_providers)
            except Exception:
                continue
        return None

    def register_model_alias(self, alias: str, real_model_id: str) -> None:
        """Registers a model alias mapping profile."""
        self.model_aliases[alias] = real_model_id

    def register_provider_alias(self, alias: str, real_provider_id: str) -> None:
        """Registers a provider alias mapping profile."""
        self.provider_aliases[alias] = real_provider_id

    def _resolve_aliases(self, request: dict) -> dict:
        cloned = request.copy()
        if cloned.get("model") in self.model_aliases:
            cloned["model"] = self.model_aliases[cloned["model"]]
        if cloned.get("provider") in self.provider_aliases:
            cloned["provider"] = self.provider_aliases[cloned["provider"]]
        return cloned

    # =========================================================================
    # COMPLIANCE, PUBLIC INVENTORIES, AND OBSERVABILITY ENGINE
    # =========================================================================

    async def get_available_providers(self) -> List[dict]:
        if not self.provider_manager or not hasattr(self.provider_manager, "getProviders"):
            return []
        providers = await self.provider_manager.getProviders() if asyncio.iscoroutinefunction(self.provider_manager.getProviders) else self.provider_manager.getProviders()
        result = []
        for p in providers:
            status = {"healthy": True}
            if self.health_monitor and hasattr(self.health_monitor, "getStatus"):
                status = self.health_monitor.getStatus(p["id"])
            if status.get("healthy", True) and not self._is_circuit_open(p["id"]):
                result.append(p)
        return result

    async def get_available_models(self) -> List[dict]:
        if not self.model_registry or not hasattr(self.model_registry, "getModels"):
            return []
        if asyncio.iscoroutinefunction(self.model_registry.getModels):
            return await self.model_registry.getModels()
        return self.model_registry.getModels()

    def register_strategy(self, name: str, execution_fn: Callable[..., Any]) -> None:
        if not callable(execution_fn):
            raise ModelRouterError("Strategy parameter must be an executable function.", "ERR_INVALID_STRATEGY_FN")
        self.strategies[name.lower()] = execution_fn
        self.emit("strategyChanged", {"action": "registered", "strategy": name})

    def unregister_strategy(self, name: str) -> None:
        canonical = name.lower()
        protected = [
            "lowestcost", "fastestresponse", "highestquality", "balanced", "roundrobin",
            "random", "leastloaded", "highestavailability", "lowestlatency", "highestreliability"
        ]
        if canonical in protected:
            raise ModelRouterError("Protected core strategy cannot be removed.", "ERR_CORE_STRATEGY_PROTECTED")
        self.strategies.pop(canonical, None)
        self.emit("strategyChanged", {"action": "unregistered", "strategy": name})

    def register_rule(self, id: str, predicate_fn: Callable[[dict], bool], action: dict, priority: int = 100) -> None:
        if not callable(predicate_fn):
            raise ModelRouterError("Rule predicate must be an executable function.", "ERR_INVALID_RULE_PREDICATE")
        self.rules[id] = {"id": id, "predicate": predicate_fn, "action": action, "priority": priority}

    def unregister_rule(self, id: str) -> None:
        self.rules.pop(id, None)

    def use(self, phase: str, middleware_fn: Callable[..., Any]) -> None:
        if phase in self.middlewares:
            self.middlewares[phase].append(middleware_fn)

    def _evaluate_rules(self, request: dict) -> Optional[dict]:
        sorted_rules = sorted(self.rules.values(), key=lambda r: r["priority"], reverse=True)
        for rule in sorted_rules:
            try:
                if rule["predicate"](request):
                    return rule["action"]
            except Exception:
                continue
        return None

    async def _verify_candidate_health(self, provider: dict) -> bool:
        if not self.health_monitor or not hasattr(self.health_monitor, "getStatus"):
            return True
        status = self.health_monitor.getStatus(provider["id"])
        return status.get("healthy", True) if isinstance(status, dict) else True

    def _increment_active_requests(self, provider_id: str) -> None:
        if provider_id not in self.load_metrics:
            self.load_metrics[provider_id] = {"activeRequests": 0, "queueSize": 0, "successCount": 0, "errorCount": 0}
        self.load_metrics[provider_id]["activeRequests"] += 1

    def _record_history_log(self, log_entry: dict) -> None:
        self.history.append(log_entry)
        if len(self.history) > self.config["historyLimit"]:
            self.history.pop(0)

    def _update_statistics(self, result: dict, latency: float) -> None:
        p_id = result["provider"]["id"]
        m_id = result["model"]["id"]
        strat = result["strategy"]

        self.stats["providerUsage"][p_id] = self.stats["providerUsage"].get(p_id, 0) + 1
        self.stats["modelUsage"][m_id] = self.stats["modelUsage"].get(m_id, 0) + 1
        self.stats["strategyUsage"][strat] = self.stats["strategyUsage"].get(strat, 0) + 1

        metrics = self.load_metrics.get(p_id)
        if metrics:
            metrics["activeRequests"] = max(0, metrics["activeRequests"] - 1)
            metrics["successCount"] = metrics.get("successCount", 0) + 1
            avg = metrics.get("avgLatency")
            metrics["avgLatency"] = (avg * 0.7) + (latency * 0.3) if avg else latency

        if self.telemetry_manager and hasattr(self.telemetry_manager, "trackMetric"):
            self.telemetry_manager.trackMetric("router.latency", latency, {"provider": p_id, "model": m_id})
            self.telemetry_manager.trackMetric("router.cost", result["costEstimation"]["totalCost"], {"provider": p_id})

    async def run_diagnostics(self) -> dict:
        """System-wide internal diagnostic instrumentation verification platform."""
        import sys
        diagnostic_payload = {
            "health": "OPERATIONAL" if (self.config["isOnline"] and not self.is_shutting_down) else "DEGRADED",
            "uptimeSeconds": int(time.time() - self.start_time),
            "routingStatistics": self.stats.copy(),
            "memoryUsage": {
                "heapTotalMb": 0,  # Placeholders to fulfill equivalent telemetry signatures
                "heapUsedMb": 0
            },
            "queueState": {"activeQueueLength": len(self.request_queue)},
            "circuits": [{"provider": k, "state": v["state"], "failures": v["failures"]} for k, v in self.circuit_breakers.items()],
            "registeredStrategies": list(self.strategies.keys()),
            "activeRulesCount": len(self.rules)
        }
        self.emit("diagnosticsGenerated", diagnostic_payload)
        return diagnostic_payload

    # =========================================================================
    # ENTERPRISE CONFIGURATION SNAPSHOTS & BACKUPS
    # =========================================================================

    def create_snapshot(self) -> dict:
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config": self.config.copy(),
            "rules": [{"id": k, "priority": v["priority"], "action": v["action"]} for k, v in self.rules.items()],
            "modelAliases": list(self.model_aliases.items()),
            "providerAliases": list(self.provider_aliases.items())
        }

    def restore_snapshot(self, snapshot: dict) -> None:
        if not snapshot or "config" not in snapshot:
            raise ModelRouterError("Invalid snapshot context layout configuration.", "ERR_SNAPSHOT_RESTORE_FAILED")
        self.config = snapshot["config"].copy()
        if "modelAliases" in snapshot:
            self.model_aliases = dict(snapshot["modelAliases"])
        if "providerAliases" in snapshot:
            self.provider_aliases = dict(snapshot["providerAliases"])
        self.emit("strategyChanged", {"action": "restored_from_snapshot"})

    def export_configuration(self) -> str:
        return json.dumps(self.create_snapshot())

    def import_configuration(self, config_str: str) -> None:
        try:
            self.restore_snapshot(json.loads(config_str))
        except Exception as err:
            raise ModelRouterError("Failed to parse dynamic parameters configuration string.", "ERR_CONFIG_IMPORT_INVALID", {"original": str(err)})

    async def backup(self) -> dict:
        return self.create_snapshot()

    async def restore(self, backup_data: dict) -> None:
        self.restore_snapshot(backup_data)

    # =========================================================================
    # OBSERVABILITY LOGGERS & SHUTDOWN MANAGERS
    # =========================================================================

    def _log_structured(self, msg: str, level: str = "INFO", meta: Optional[dict] = None) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": msg,
            "subsystem": "model_router"
        }
        if meta:
            payload.update(meta)
        log_str = json.dumps(payload)
        if level in ["ERROR", "WARN"]:
            logger.error(log_str)
        else:
            logger.info(log_str)

    def _log_error(self, err: ModelRouterError) -> None:
        self._log_structured(err.message, "ERROR", {"code": err.code, "metadata": err.metadata, "stack": err.stack})

    async def shutdown(self) -> None:
        """Gracefully clears active connections, queues, and locks before termination."""
        self.is_shutting_down = True
        self._log_structured("Router shutting down. Cleaning up queues and resources.", "INFO")

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

        for request_id, event in list(self.active_request_controllers.items()):
            event.set()
            self.active_request_controllers.pop(request_id, None)

        while self.request_queue:
            ctx, future = self.request_queue.popleft()
            if not future.done():
                future.set_exception(ModelRouterError("System terminated during graceful shutdown processing loops.", "ERR_ROUTER_SHUTDOWN"))
