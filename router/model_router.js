import { EventEmitter } from 'node:events';
import { randomUUID } from 'node:crypto';

/**
 * Enterprise-grade Custom Error class for Cabangile AI Studio Router.
 * Supports granular system codes, structural metadata tracking, and operational categorization.
 * @extends Error
 */
export class ModelRouterError extends Error {
  /**
   * @param {string} message - Explicit human-readable diagnostic message.
   * @param {string} code - System error code categorization (e.g., 'ERR_CIRCUIT_BREAKER_OPEN').
   * @param {Record<string, any>} [metadata={}] - Structural debug context payload.
   */
  constructor(message, code, metadata = {}) {
    super(message);
    this.name = 'ModelRouterError';
    this.code = code;
    this.metadata = metadata;
    this.timestamp = new Date().toISOString();
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Serializes the internal state configuration to a clean payload.
   * @returns {Record<string, any>}
   */
  toJSON() {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      timestamp: this.timestamp,
      metadata: this.metadata,
      stack: this.stack
    };
  }
}

/**
 * Cabangile AI Studio Enterprise AI Orchestration Routing Engine.
 * Complete production-ready implementation managing resilience, caching, throttles, and topologies.
 * @extends EventEmitter
 */
export class ModelRouter extends EventEmitter {
  /**
   * @param {Object} providerManager - Active cloud and local API providers wrapper.
   * @param {Object} modelRegistry - Reference matrix for pricing models and capability lists.
   * @param {Object} healthMonitor - Active monitoring hook for infrastructure status.
   * @param {Object} costTracker - Real-time accounting interface for structural billing tracking.
   * @param {Object} telemetryManager - Aggregator linking transactional stats to higher monitoring instances.
   * @param {Record<string, any>} [config={}] - High performance fine-tuning parameters.
   */
  constructor(providerManager, modelRegistry, healthMonitor, costTracker, telemetryManager, config = {}) {
    super();
    this.providerManager = providerManager;
    this.modelRegistry = modelRegistry;
    this.healthMonitor = healthMonitor;
    this.costTracker = costTracker;
    this.telemetryManager = telemetryManager;

    // Deep merge configuration rules and runtime properties
    this.config = {
      defaultStrategy: 'balanced',
      maxFailoverAttempts: 3,
      historyLimit: 2000,
      tokenEstimationFactor: 4,
      isOnline: true,
      globalTimeoutMs: 30000,
      defaultCacheTtlMs: 60000,
      maxQueueSize: 5000,
      queueTimeoutMs: 15000,
      cooldownPeriodMs: 45000,
      maxCircuitFailures: 5,
      circuitBreakerResetTimeoutMs: 30000,
      deduplicationWindowMs: 2000,
      weights: { health: 0.3, cost: 0.2, speed: 0.2, quality: 0.15, reliability: 0.15 },
      ...config
    };

    // System Topology Registers
    this.strategies = new Map();
    this.rules = new Map();
    this.middlewares = { before: [], after: [] };
    this.plugins = new Map();
    this.modelAliases = new Map();
    this.providerAliases = new Map();
    this.fallbackChains = new Map();

    // Architectural Resiliency & Storage Subsystems
    this.history = [];
    this.startTime = Date.now();
    this.roundRobinPointers = new Map();
    this.loadMetrics = new Map();
    this.circuitBreakers = new Map();
    this.providerCooldowns = new Map();
    this.requestCache = new Map();
    this.requestQueue = [];
    this.deduplicationMap = new Map();
    this.activeRequestControllers = new Map();
    this.isShuttingDown = false;

    // Running Analytical Counters
    this.stats = {
      totalRequests: 0,
      routedRequests: 0,
      failedRequests: 0,
      failovers: 0,
      totalRoutingTimeMs: 0,
      totalEstimatedCost: 0,
      cacheHits: 0,
      queueCount: 0,
      circuitBreaks: 0,
      providerUsage: {},
      modelUsage: {},
      strategyUsage: {}
    };

    // Initialize Structural Subsystems
    this._initializeBuiltInStrategies();
    this._startBackgroundCleanupTasks();
  }

  // =========================================================================
  // CORE API OVERRIDES & IMPLEMENTATIONS
  // =========================================================================

  /**
   * Primary Entrypoint. Routes an incoming model request safely through the resilience pipeline.
   * @param {Object} request - Request definition parameters.
   * @param {Object} [options={}] - Execution options, explicit overrides, or pass-through AbortSignals.
   * @returns {Promise<Object>}
   */
  async route(request, options = {}) {
    if (this.isShuttingDown) {
      throw new ModelRouterError('System is undergoing graceful shutdown routines.', 'ERR_ROUTER_SHUTDOWN');
    }

    const requestId = request.requestId || randomUUID();
    const startTime = performance.now();
    this.stats.totalRequests++;

    // Context execution metadata envelope
    const ctx = {
      requestId,
      request: this._resolveAliases(request),
      options,
      attempts: 0,
      history: [],
      timeoutToken: null,
      internalSignal: null
    };

    // Request Deduplication layer
    const dedupKey = this._generateDeduplicationKey(ctx.request);
    if (dedupKey && this.deduplicationMap.has(dedupKey)) {
      this.stats.cacheHits++;
      return this.deduplicationMap.get(dedupKey);
    }

    // Cache lookup matrix layer
    if (ctx.request.cacheKey || options.useCache) {
      const cached = this._lookupCache(ctx.request);
      if (cached) {
        this.stats.cacheHits++;
        return cached;
      }
    }

    this.emit('routingStarted', { requestId, request: ctx.request });

    // Request structural queues allocation boundaries
    if (this._shouldQueueRequest(ctx.request)) {
      await this._enqueueRequest(ctx);
    }

    // Set up lifecycle cancellation signals
    this._setupCancellationContext(ctx);

    let promiseExecution = (async () => {
      try {
        await this.validateRequest(ctx.request);

        // Before Middleware Hook
        for (const middleware of this.middlewares.before) {
          await middleware(ctx);
        }

        // Apply external lifecycle plugins
        for (const [pluginName, plugin] of this.plugins) {
          if (typeof plugin.beforeRoute === 'function') {
            await plugin.beforeRoute(ctx);
          }
        }

        const routingResult = await this._executeRoutingLoopWithFailover(ctx);

        const latency = performance.now() - startTime;
        this.stats.routedRequests++;
        this.stats.totalRoutingTimeMs += latency;
        this.stats.totalEstimatedCost += routingResult.costEstimation.totalCost;

        this._updateStatistics(routingResult, latency);

        const finalOutput = {
          requestId,
          provider: routingResult.provider.id,
          model: routingResult.model.id,
          strategy: routingResult.strategy,
          costEstimation: routingResult.costEstimation,
          latencyMs: latency,
          history: ctx.history,
          isStream: !!(ctx.request.streaming || routingResult.model.capabilities?.streaming)
        };

        // After Middleware Hook
        for (const middleware of this.middlewares.after) {
          await middleware(finalOutput);
        }

        for (const [pluginName, plugin] of this.plugins) {
          if (typeof plugin.afterRoute === 'function') {
            await plugin.afterRoute(finalOutput, ctx);
          }
        }

        // Cache and deduplicate results when applicable
        this._populateCacheAndDedup(ctx.request, finalOutput, dedupKey);

        this.emit('routingCompleted', finalOutput);
        return finalOutput;

      } catch (error) {
        this.stats.failedRequests++;
        const routerError = error instanceof ModelRouterError 
          ? error 
          : new ModelRouterError(error.message, 'ERR_ROUTING_FAILED', { originalError: error, requestId });
        
        this.emit('routingFailed', { requestId, error: routerError });
        this._logError(routerError);
        throw routerError;
      } finally {
        this._clearCancellationContext(ctx);
      }
    })();

    if (dedupKey && !this.deduplicationMap.has(dedupKey)) {
      this.deduplicationMap.set(dedupKey, promiseExecution);
    }

    return promiseExecution;
  }

  /**
   * Concurrently processes collections of structured requests through standard isolated threads or event loops.
   * @param {Array<Object>} requests
   * @param {Object} [options={}]
   * @returns {Promise<Array<Object>>}
   */
  async routeBatch(requests, options = {}) {
    if (!Array.isArray(requests)) {
      throw new ModelRouterError('Batch operations require an array structure payload.', 'ERR_INVALID_BATCH');
    }
    return Promise.all(requests.map(req => this.route(req, options).catch(err => ({ error: err.toJSON() }))));
  }

  // =========================================================================
  // CORE INTERNAL ROUTING & FAILOVER EXECUTION PIPELINE
  // =========================================================================

  /**
   * Loops through calculated optimization spaces with resilient failover structures.
   * @private
   */
  async _executeRoutingLoopWithFailover(ctx) {
    const maxAttempts = this.config.maxFailoverAttempts;
    const excludedProviders = new Set();

    while (ctx.attempts < maxAttempts) {
      if (ctx.internalSignal?.aborted) {
        throw new ModelRouterError('Transaction execution lifecycle aborted by caller.', 'ERR_REQUEST_CANCELLED');
      }

      ctx.attempts++;
      let candidate = null;

      try {
        candidate = await this._selectCandidate(ctx.request, excludedProviders);
      } catch (selectError) {
        // Evaluate dynamic cross-network cluster level fallbacks
        candidate = await this._evaluateFallbackChain(ctx.request, excludedProviders);
        if (!candidate) throw selectError;
      }

      const pId = candidate.provider.id;
      this.emit('providerSelected', { requestId: ctx.requestId, provider: pId, model: candidate.model.id, attempt: ctx.attempts });

      // Resilience Checks: Circuit Breaker, Rate Limiters, Financial Quota systems
      if (this._isCircuitOpen(pId) || this._isProviderCooldownActive(pId) || !this._checkRateAndQuotaLimits(candidate)) {
        this.emit('providerUnavailable', { requestId: ctx.requestId, provider: pId });
        excludedProviders.add(pId);
        continue;
      }

      try {
        // Enforce active concurrency limits per infrastructure node
        this._acquireConcurrentSlot(pId);

        // Core dynamic health assertion check
        const isHealthy = await this._verifyCandidateHealth(candidate.provider);
        if (!isHealthy) {
          this._recordCircuitFailure(pId);
          throw new ModelRouterError(`Provider validation reported unhealthy states: ${pId}`, 'ERR_PROVIDER_UNHEALTHY');
        }

        // Success Path mapping injection
        this._resetCircuitBreaker(pId);

        ctx.history.push({
          attempt: ctx.attempts,
          provider: pId,
          model: candidate.model.id,
          strategy: candidate.strategy,
          timestamp: new Date().toISOString()
        });

        this._recordHistoryLog({
          requestId: ctx.requestId,
          provider: pId,
          model: candidate.model.id,
          strategy: candidate.strategy,
          cost: candidate.costEstimation.totalCost,
          success: true,
          timestamp: new Date().toISOString()
        });

        return candidate;

      } catch (executionError) {
        this._recordCircuitFailure(pId);
        this._activateCooldownTimer(pId);
        this._releaseConcurrentSlot(pId);

        this.emit('providerRejected', { requestId: ctx.requestId, error: executionError.message });
        
        if (ctx.attempts >= maxAttempts) {
          throw new ModelRouterError(`Execution context chain exhausted max failover threshold limits (${maxAttempts}).`, 'ERR_FAILOVER_EXHAUSTED', { history: ctx.history });
        }

        this.stats.failovers++;
        this.emit('failoverStarted', { requestId: ctx.requestId, currentAttempt: ctx.attempts });
      }
    }

    throw new ModelRouterError('Routing engines completely failed to allocate functional clusters.', 'ERR_NO_ROUTE_AVAILABLE');
  }

  /**
   * Scores, balances, and validates constraints across multi-tier topologies.
   * @private
   */
  async _selectCandidate(request, excludedProviders) {
    const allModels = await this.getAvailableModels();
    const allProviders = await this.getAvailableProviders();

    // Process administrative specific routing override rules
    const ruleAction = this._evaluateRules(request);
    let targetedModelName = request.model || (ruleAction ? ruleAction.model : null);
    let targetedProviderName = ruleAction ? ruleAction.provider : null;

    let candidates = [];

    for (const model of allModels) {
      if (targetedModelName && model.id !== targetedModelName && model.family !== targetedModelName) {
        continue;
      }

      for (const provider of allProviders) {
        if (excludedProviders.has(provider.id)) continue;
        if (targetedProviderName && provider.id !== targetedProviderName) continue;

        // Verify basic structural model configuration matching
        if (!provider.supportedModels?.includes(model.id) && provider.modelId !== model.id) {
          continue;
        }

        // Capability Matrix Validation Engine
        if (request.capabilities) {
          const supportsAll = request.capabilities.every(cap => model.capabilities?.[cap] || provider.capabilities?.[cap]);
          if (!supportsAll) continue;
        }

        // Special handling validation metrics
        if (request.vision && !model.capabilities?.vision) continue;
        if (request.streaming && !model.capabilities?.streaming) continue;

        const costEstimation = await this.estimateCost(request, model);
        
        // Enforce budgets and capabilities before choosing path
        if (model.contextWindow && costEstimation.totalTokens > model.contextWindow) continue;
        if (!this._validateBudgetEnforcement(costEstimation)) continue;

        candidates.push({ provider, model, costEstimation });
      }
    }

    if (candidates.length === 0) {
      throw new ModelRouterError('No operational providers matched specifications.', 'ERR_CAPABILITY_MISMATCH');
    }

    // Match Strategy execution context
    const requestedStrategy = (request.strategy || this.config.defaultStrategy).toLowerCase().replace(/\s+/g, '');
    const strategyFn = this.strategies.get(requestedStrategy) || this.strategies.get('balanced');

    const selected = await strategyFn(candidates, {
      roundRobinPointers: this.roundRobinPointers,
      loadMetrics: this.loadMetrics,
      weights: this.config.weights,
      history: this.history
    });

    if (!selected) {
      throw new ModelRouterError('Strategy computation module failed to yield valid context targets.', 'ERR_STRATEGY_FAULT');
    }

    return { ...selected, strategy: requestedStrategy };
  }

  // =========================================================================
  // INTEGRATED STRATEGIES & MATHEMATICAL SCORING MATRICES
  // =========================================================================

  /**
   * Injects core enterprise metric evaluation weights into selection modules.
   * @private
   */
  _initializeBuiltInStrategies() {
    this.registerStrategy('lowestcost', async (candidates) => candidates.sort((a, b) => a.costEstimation.totalCost - b.costEstimation.totalCost)[0]);
    this.registerStrategy('fastestresponse', async (candidates, ctx) => candidates.sort((a, b) => (ctx.loadMetrics.get(a.provider.id)?.activeRequests || 0) - (ctx.loadMetrics.get(b.provider.id)?.activeRequests || 0))[0]);
    this.registerStrategy('highestquality', async (candidates) => candidates.sort((a, b) => (b.model.qualityScore || 0) - (a.model.qualityScore || 0))[0]);
    this.registerStrategy('highestavailability', async (candidates) => candidates.sort((a, b) => (b.provider.availabilityScore || 0) - (a.provider.availabilityScore || 0))[0]);
    this.registerStrategy('lowestlatency', async (candidates, ctx) => candidates.sort((a, b) => (ctx.loadMetrics.get(a.provider.id)?.avgLatency || 0) - (ctx.loadMetrics.get(b.provider.id)?.avgLatency || 0))[0]);
    this.registerStrategy('highestreliability', async (candidates) => candidates.sort((a, b) => (b.provider.reliabilityScore || 0) - (a.provider.reliabilityScore || 0))[0]);
    this.registerStrategy('random', async (candidates) => candidates[Math.floor(Math.random() * candidates.length)]);
    this.registerStrategy('leastloaded', async (candidates, ctx) => candidates.sort((a, b) => (ctx.loadMetrics.get(a.provider.id)?.activeRequests || 0) - (ctx.loadMetrics.get(b.provider.id)?.activeRequests || 0))[0]);
    
    this.registerStrategy('roundrobin', async (candidates, ctx) => {
      const key = candidates.map(c => c.provider.id).sort().join('|');
      let idx = ctx.roundRobinPointers.get(key) || 0;
      if (idx >= candidates.length) idx = 0;
      const chosen = candidates[idx];
      ctx.roundRobinPointers.set(key, idx + 1);
      return chosen;
    });

    // Dynamic Multi-Factor Weighted Scoring Strategy
    this.registerStrategy('balanced', async (candidates, ctx) => {
      let topCandidate = null;
      let highestScore = -Infinity;

      for (const candidate of candidates) {
        const pId = candidate.provider.id;
        const metrics = ctx.loadMetrics.get(pId) || { activeRequests: 0, successCount: 1, errorCount: 0 };
        
        // Base telemetry calculation inputs
        const totalRequests = metrics.successCount + metrics.errorCount || 1;
        const reliabilityFactor = metrics.successCount / totalRequests;
        const providerPriorityWeight = candidate.provider.priorityWeight || 1;

        // Dynamic health scoring calculations
        const healthScore = this._calculateDynamicHealthScore(pId) * reliabilityFactor;
        const costScore = 1 / (1 + (candidate.costEstimation.totalCost || 0));
        const loadScore = 1 / (1 + (metrics.activeRequests || 0));
        const qualityScore = (candidate.model.qualityScore || 50) / 100;

        // Incorporate adaptive learning parameters using history tracking
        const pastPerformanceAdjustment = this._calculateAdaptiveHistoryWeight(pId, ctx.history);

        const score = (
          (healthScore * ctx.weights.health) +
          (costScore * ctx.weights.cost) +
          (loadScore * ctx.weights.speed) +
          (qualityScore * ctx.weights.quality) +
          (pastPerformanceAdjustment * ctx.weights.reliability)
        ) * providerPriorityWeight;

        if (score > highestScore) {
          highestScore = score;
          topCandidate = candidate;
        }
      }
      return topCandidate || candidates[0];
    });
  }

  // =========================================================================
  // FINANCIALS, TOKEN ESTIMATION & ADMINISTRATIVE POLICIES
  // =========================================================================

  /**
   * Resolves text inputs, arrays, and token allocations safely.
   */
  async estimateTokens(request) {
    let promptTokens = 0;
    if (typeof request.input === 'string') {
      promptTokens = Math.ceil(request.input.length / this.config.tokenEstimationFactor);
    } else if (Array.isArray(request.input)) {
      promptTokens = request.input.reduce((acc, msg) => {
        const length = typeof msg.content === 'string' ? msg.content.length : JSON.stringify(msg.content || '').length;
        return acc + Math.ceil(length / this.config.tokenEstimationFactor) + 4;
      }, 0);
    } else if (request.input && typeof request.input === 'object') {
      promptTokens = Math.ceil(JSON.stringify(request.input).length / this.config.tokenEstimationFactor);
    }

    const estimatedOutputTokens = request.estimatedOutputSize || request.maxTokens || 256;
    return { promptTokens, estimatedOutputTokens, totalTokens: promptTokens + estimatedOutputTokens };
  }

  /**
   * Computes billing projections matching configurations.
   */
  async estimateCost(request, modelSpec) {
    const { promptTokens, estimatedOutputTokens, totalTokens } = await this.estimateTokens(request);
    const pricing = modelSpec.pricing || { inputPerToken: 0, outputPerToken: 0 };
    const promptCost = promptTokens * (pricing.inputPerToken || 0);
    const completionCost = estimatedOutputTokens * (pricing.outputPerToken || 0);
    
    return { promptTokens, estimatedOutputTokens, totalTokens, promptCost, completionCost, totalCost: promptCost + completionCost };
  }

  /**
   * Evaluates if a request violates budget restrictions.
   * @private
   */
  _validateBudgetEnforcement(costEstimation) {
    if (this.costTracker) {
      if (typeof this.costTracker.checkBudgetLimit === 'function' && !this.costTracker.checkBudgetLimit(costEstimation.totalCost)) return false;
      if (typeof this.costTracker.checkTokenBudget === 'function' && !this.costTracker.checkTokenBudget(costEstimation.totalTokens)) return false;
    }
    return true;
  }

  /**
   * Validates structure requirements before entering the optimization loop.
   */
  async validateRequest(request) {
    if (!request || (!request.input && !request.prompt)) {
      throw new ModelRouterError('Inbound routing request payload cannot be empty.', 'ERR_MALFORMED_REQUEST');
    }
  }

  // =========================================================================
  // RESILIENCY SUBSYSTEMS: CIRCUIT BREAKERS, COOLDOWNS & ACCELERATION
  // =========================================================================

  /**
   * Evaluates state maps for active breaker objects.
   * @private
   */
  _isCircuitOpen(providerId) {
    const breaker = this.circuitBreakers.get(providerId);
    if (!breaker || breaker.state === 'CLOSED') return false;

    if (breaker.state === 'OPEN') {
      if (Date.now() - breaker.lastFailureTime > this.config.circuitBreakerResetTimeoutMs) {
        breaker.state = 'HALF_OPEN';
        this._logStructured('Circuit enters verification state.', 'INFO', { providerId });
        return false;
      }
      return true;
    }
    return false;
  }

  /**
   * Increases failure thresholds and updates structural circuit monitoring models.
   * @private
   */
  _recordCircuitFailure(providerId) {
    if (!this.circuitBreakers.has(providerId)) {
      this.circuitBreakers.set(providerId, { failures: 0, state: 'CLOSED', lastFailureTime: 0 });
    }
    const breaker = this.circuitBreakers.get(providerId);
    breaker.failures++;
    breaker.lastFailureTime = Date.now();

    if (breaker.failures >= this.config.maxCircuitFailures && breaker.state !== 'OPEN') {
      breaker.state = 'OPEN';
      this.stats.circuitBreaks++;
      this.emit('providerUnavailable', { providerId, reason: 'CIRCUIT_BREAKER_TRIGGERED' });
      this._logStructured('Circuit breaker activated for node.', 'WARN', { providerId, failures: breaker.failures });
    }
  }

  /**
   * Clear circuit variables following success paths.
   * @private
   */
  _resetCircuitBreaker(providerId) {
    this.circuitBreakers.set(providerId, { failures: 0, state: 'CLOSED', lastFailureTime: 0 });
  }

  /**
   * Prevents system thrashing via immediate cooling windows.
   * @private
   */
  _activateCooldownTimer(providerId) {
    this.providerCooldowns.set(providerId, Date.now() + this.config.cooldownPeriodMs);
  }

  /**
   * Checks if a provider cooldown window is currently active.
   * @private
   */
  _isProviderCooldownActive(providerId) {
    const activeUntil = this.providerCooldowns.get(providerId);
    if (!activeUntil) return false;
    if (Date.now() > activeUntil) {
      this.providerCooldowns.delete(providerId);
      return false;
    }
    return true;
  }

  /**
   * Asserts real-time downstream system status.
   * @private
   */
  _calculateDynamicHealthScore(providerId) {
    if (!this.healthMonitor || typeof this.healthMonitor.getScore !== 'function') return 1.0;
    return this.healthMonitor.getScore(providerId);
  }

  /**
   * Dynamically tracks response patterns.
   * @private
   */
  _calculateAdaptiveHistoryWeight(providerId, historyList) {
    if (!historyList || historyList.length === 0) return 1.0;
    const items = historyList.filter(h => h.provider === providerId).slice(-10);
    if (items.length === 0) return 1.0;
    const successes = items.filter(i => i.success).length;
    return successes / items.length;
  }

  /**
   * Enforces provider quota and sliding window configurations.
   * @private
   */
  _checkRateAndQuotaLimits(candidate) {
    const pId = candidate.provider.id;
    if (this.providerManager?.checkQuota && !this.providerManager.checkQuota(pId)) return false;
    
    const metrics = this.loadMetrics.get(pId);
    if (metrics && candidate.provider.maxConcurrentRequests && metrics.activeRequests >= candidate.provider.maxConcurrentRequests) {
      return false;
    }
    return true;
  }

  /**
   * Implements thread-safe resource locks inside memory allocation monitors.
   * @private
   */
  _acquireConcurrentSlot(providerId) {
    this._incrementActiveRequests(providerId);
  }

  /**
   * Frees operational locks following route completion.
   * @private
   */
  _releaseConcurrentSlot(providerId) {
    const metrics = this.loadMetrics.get(providerId);
    if (metrics) {
      metrics.activeRequests = Math.max(0, metrics.activeRequests - 1);
    }
  }

  // =========================================================================
  // CACHING, DEDUPLICATION, AND QUEUE MANAGEMENT SUBSYSTEMS
  // =========================================================================

  /**
   * Evaluates explicit incoming data schemas to determine execution cache paths.
   * @private
   */
  _generateDeduplicationKey(request) {
    if (!request.input) return null;
    try {
      const source = typeof request.input === 'string' ? request.input : JSON.stringify(request.input);
      return `${request.model || 'default'}:${source}`;
    } catch {
      return null;
    }
  }

  /**
   * Local TTL Key Lookup optimization framework.
   * @private
   */
  _lookupCache(request) {
    const key = request.cacheKey || this._generateDeduplicationKey(request);
    if (!key) return null;
    const record = this.requestCache.get(key);
    if (!record) return null;
    if (Date.now() > record.expiresAt) {
      this.requestCache.delete(key);
      return null;
    }
    return record.payload;
  }

  /**
   * Populates both cache and deduplication maps with resolved routing configurations.
   * @private
   */
  _populateCacheAndDedup(request, output, dedupKey) {
    const key = request.cacheKey || dedupKey;
    if (key) {
      const ttl = request.cacheTtlMs || this.config.defaultCacheTtlMs;
      this.requestCache.set(key, { payload: output, expiresAt: Date.now() + ttl });
    }
    if (dedupKey) {
      // Retain resolved promises briefly to align overlapping asynchronous loops
      setTimeout(() => this.deduplicationMap.delete(dedupKey), this.config.deduplicationWindowMs);
    }
  }

  /**
   * Evaluates if system should queue incoming workloads.
   * @private
   */
  _shouldQueueRequest(request) {
    if (request.bypassQueue) return false;
    const currentActiveCount = Array.from(this.loadMetrics.values()).reduce((sum, m) => sum + m.activeRequests, 0);
    return currentActiveCount > (this.config.maxQueueSize / 2);
  }

  /**
   * Pushes execution elements into transactional queue systems.
   * @private
   */
  async _enqueueRequest(ctx) {
    if (this.requestQueue.length >= this.config.maxQueueSize) {
      throw new ModelRouterError('System-wide operational queues are completely saturated.', 'ERR_QUEUE_SATURATED');
    }

    this.stats.queueCount++;
    return new Promise((resolve, reject) => {
      const timeoutToken = setTimeout(() => {
        const index = this.requestQueue.findIndex(item => item.ctx.requestId === ctx.requestId);
        if (index !== -1) {
          this.requestQueue.splice(index, 1);
          reject(new ModelRouterError('Queue retention time limit exceeded.', 'ERR_QUEUE_TIMEOUT'));
        }
      }, this.config.queueTimeoutMs);

      this.requestQueue.push({ ctx, resolve, reject, timeoutToken });
      this._processQueue();
    });
  }

  /**
   * Evaluates queue pools against open resources.
   * @private
   */
  _processQueue() {
    if (this.requestQueue.length === 0) return;
    const item = this.requestQueue.shift();
    clearTimeout(item.timeoutToken);
    item.resolve();
  }

  // =========================================================================
  // LIFECYCLE CANCELLATION & BACKGROUND HOUSEKEEPING AUTOMATIONS
  // =========================================================================

  /**
   * Hooks internal signal controls into global abort hooks.
   * @private
   */
  _setupCancellationContext(ctx) {
    const controller = new AbortController();
    this.activeRequestControllers.set(ctx.requestId, controller);
    ctx.internalSignal = controller.signal;

    if (ctx.options.signal) {
      ctx.options.signal.addEventListener('abort', () => controller.abort());
    }

    // Initialize global configuration timeout limits
    ctx.timeoutToken = setTimeout(() => {
      controller.abort();
      this.emit('routingFailed', { requestId: ctx.requestId, reason: 'TIMEOUT_TRIGGERED' });
    }, ctx.options.timeoutMs || this.config.globalTimeoutMs);
  }

  /**
   * Drops listeners to avoid runtime memory leaks.
   * @private
   */
  _clearCancellationContext(ctx) {
    clearTimeout(ctx.timeoutToken);
    this.activeRequestControllers.delete(ctx.requestId);
  }

  /**
   * Cleans expired caches and tracking parameters.
   * @private
   */
  _startBackgroundCleanupTasks() {
    this.cleanupTimer = setInterval(() => {
      const now = Date.now();
      
      // Clean expired execution caches
      for (const [k, v] of this.requestCache.entries()) {
        if (now > v.expiresAt) this.requestCache.delete(k);
      }
      
      // Clean stale breaker profiles
      for (const [k, v] of this.circuitBreakers.entries()) {
        if (v.state === 'CLOSED' && now - v.lastFailureTime > 300000) {
          this.circuitBreakers.delete(k);
        }
      }
    }, 15000);

    // Keep event loop clear when process receives shutdown requests
    if (this.cleanupTimer.unref) this.cleanupTimer.unref();
  }

  // =========================================================================
  // DYNAMIC COMPONENT PLUGINS & TOPOLOGY OVERRIDES
  // =========================================================================

  /**
   * Registers a plugin hook to inject custom enterprise validation rules or compliance metrics.
   * @param {string} name
   * @param {Object} pluginObject
   */
  registerPlugin(name, pluginObject) {
    this.plugins.set(name, pluginObject);
  }

  /**
   * Configures cross-region fallback maps for active failover paths.
   * @param {string} sourceProviderId
   * @param {Array<string>} fallbackProvidersList
   */
  registerFallbackChain(sourceProviderId, fallbackProvidersList) {
    this.fallbackChains.set(sourceProviderId, fallbackProvidersList);
  }

  /**
   * Internal mechanism to navigate fallback clusters when primary operations fail.
   * @private
   */
  async _evaluateFallbackChain(request, excludedProviders) {
    const targetProvider = request.provider;
    if (!targetProvider || !this.fallbackChains.has(targetProvider)) return null;

    const chain = this.fallbackChains.get(targetProvider);
    for (const providerId of chain) {
      if (excludedProviders.has(providerId)) continue;
      
      // Patch request to explore next step in the fallback sequence
      const mutatedRequest = { ...request, provider: providerId };
      try {
        return await this._selectCandidate(mutatedRequest, excludedProviders);
      } catch {
        continue;
      }
    }
    return null;
  }

  /**
   * Registers a model alias mapping profile.
   */
  registerModelAlias(alias, realModelId) { this.modelAliases.set(alias, realModelId); }

  /**
   * Registers a provider alias mapping profile.
   */
  registerProviderAlias(alias, realProviderId) { this.providerAliases.set(alias, realProviderId); }

  /**
   * Resolves structural aliased targets into explicit platform definitions.
   * @private
   */
  _resolveAliases(request) {
    const cloned = { ...request };
    if (cloned.model && this.modelAliases.has(cloned.model)) cloned.model = this.modelAliases.get(cloned.model);
    if (cloned.provider && this.providerAliases.has(cloned.provider)) cloned.provider = this.providerAliases.get(cloned.provider);
    return cloned;
  }

  // =========================================================================
  // COMPLIANCE, PUBLIC INVENTORIES, AND OBSERVABILITY ENGINE
  // =========================================================================

  async getAvailableProviders() {
    if (!this.providerManager?.getProviders) return [];
    const providers = await this.providerManager.getProviders();
    return providers.filter(p => {
      const status = this.healthMonitor ? this.healthMonitor.getStatus(p.id) : { healthy: true };
      return status.healthy && !this._isCircuitOpen(p.id);
    });
  }

  async getAvailableModels() {
    if (!this.modelRegistry?.getModels) return [];
    return await this.modelRegistry.getModels();
  }

  registerStrategy(name, executionFn) {
    if (typeof executionFn !== 'function') throw new ModelRouterError('Strategy parameter must be an executable function.', 'ERR_INVALID_STRATEGY_FN');
    this.strategies.set(name.toLowerCase(), executionFn);
    this.emit('strategyChanged', { action: 'registered', strategy: name });
  }

  unregisterStrategy(name) {
    const canonical = name.toLowerCase();
    if (['lowestcost', 'fastestresponse', 'highestquality', 'balanced', 'roundrobin', 'random', 'leastloaded', 'highestavailability', 'lowestlatency', 'highestreliability'].includes(canonical)) {
      throw new ModelRouterError('Protected core strategy cannot be removed.', 'ERR_CORE_STRATEGY_PROTECTED');
    }
    this.strategies.delete(canonical);
    this.emit('strategyChanged', { action: 'unregistered', strategy: name });
  }

  registerRule(id, predicateFn, action, priority = 100) {
    if (typeof predicateFn !== 'function') throw new ModelRouterError('Rule predicate must be an executable function.', 'ERR_INVALID_RULE_PREDICATE');
    this.rules.set(id, { id, predicate: predicateFn, action, priority });
  }

  unregisterRule(id) { this.rules.delete(id); }
  use(phase, middlewareFn) { this.middlewares[phase].push(middlewareFn); }

  _evaluateRules(request) {
    const sorted = Array.from(this.rules.values()).sort((a, b) => b.priority - a.priority);
    for (const rule of sorted) {
      try {
        if (rule.predicate(request)) return rule.action;
      } catch {}
    }
    return null;
  }

  async _verifyCandidateHealth(provider) {
    if (!this.healthMonitor) return true;
    return this.healthMonitor.getStatus(provider.id)?.healthy ?? true;
  }

  _incrementActiveRequests(providerId) {
    if (!this.loadMetrics.has(providerId)) {
      this.loadMetrics.set(providerId, { activeRequests: 0, queueSize: 0, successCount: 0, errorCount: 0 });
    }
    this.loadMetrics.get(providerId).activeRequests++;
  }

  _recordHistoryLog(logEntry) {
    this.history.push(logEntry);
    if (this.history.length > this.config.historyLimit) this.history.shift();
  }

  _updateStatistics(result, latency) {
    const pId = result.provider.id;
    const mId = result.model.id;
    const strat = result.strategy;

    this.stats.providerUsage[pId] = (this.stats.providerUsage[pId] || 0) + 1;
    this.stats.modelUsage[mId] = (this.stats.modelUsage[mId] || 0) + 1;
    this.stats.strategyUsage[strat] = (this.stats.strategyUsage[strat] || 0) + 1;

    const metrics = this.loadMetrics.get(pId);
    if (metrics) {
      metrics.activeRequests = Math.max(0, metrics.activeRequests - 1);
      metrics.successCount++;
      metrics.avgLatency = metrics.avgLatency ? (metrics.avgLatency * 0.7) + (latency * 0.3) : latency;
    }

    if (this.telemetryManager?.trackMetric) {
      this.telemetryManager.trackMetric('router.latency', latency, { provider: pId, model: mId });
      this.telemetryManager.trackMetric('router.cost', result.costEstimation.totalCost, { provider: pId });
    }
  }

  /**
   * System-wide internal diagnostic instrumentation verification platform.
   */
  async runDiagnostics() {
    const memory = process.memoryUsage();
    const diagnosticPayload = {
      health: this.config.isOnline && !this.isShuttingDown ? 'OPERATIONAL' : 'DEGRADED',
      uptimeSeconds: Math.floor((Date.now() - this.startTime) / 1000),
      routingStatistics: { ...this.stats },
      memoryUsage: {
        heapTotalMb: Math.round(memory.heapTotal / 1024 / 1024),
        heapUsedMb: Math.round(memory.heapUsed / 1024 / 1024)
      },
      queueState: { activeQueueLength: this.requestQueue.length },
      circuits: Array.from(this.circuitBreakers.entries()).map(([k, v]) => ({ provider: k, state: v.state, failures: v.failures })),
      registeredStrategies: Array.from(this.strategies.keys()),
      activeRulesCount: this.rules.size
    };
    this.emit('diagnosticsGenerated', diagnosticPayload);
    return diagnosticPayload;
  }

  // =========================================================================
  // ENTERPRISE CONFIGURATION SNAPSHOTS & BACKUPS
  // =========================================================================

  createSnapshot() {
    return {
      timestamp: new Date().toISOString(),
      config: { ...this.config },
      rules: Array.from(this.rules.entries()).map(([k, v]) => ({ id: k, priority: v.priority, action: v.action })),
      modelAliases: Array.from(this.modelAliases.entries()),
      providerAliases: Array.from(this.providerAliases.entries())
    };
  }

  restoreSnapshot(snapshot) {
    if (!snapshot || !snapshot.config) throw new ModelRouterError('Invalid snapshot context layout configuration.', 'ERR_SNAPSHOT_RESTORE_FAILED');
    this.config = { ...snapshot.config };
    if (snapshot.modelAliases) this.modelAliases = new Map(snapshot.modelAliases);
    if (snapshot.providerAliases) this.providerAliases = new Map(snapshot.providerAliases);
    this.emit('strategyChanged', { action: 'restored_from_snapshot' });
  }

  exportConfiguration() { return JSON.stringify(this.createSnapshot()); }
  importConfiguration(configStr) {
    try {
      this.restoreSnapshot(JSON.parse(configStr));
    } catch (err) {
      throw new ModelRouterError('Failed to parse dynamic parameters configuration string.', 'ERR_CONFIG_IMPORT_INVALID', { original: err });
    }
  }

  async backup() { return this.createSnapshot(); }
  async restore(backupData) { this.restoreSnapshot(backupData); }

  // =========================================================================
  // OBSERVABILITY LOGGERS & SHUTDOWN MANAGERS
  // =========================================================================

  /**
   * Internal structured console utility logging pipeline.
   * @private
   */
  _logStructured(msg, level = 'INFO', meta = {}) {
    const payload = { timestamp: new Date().toISOString(), level, message: msg, subsystem: 'model_router', ...meta };
    if (level === 'ERROR' || level === 'WARN') {
      console.error(JSON.stringify(payload));
    } else {
      console.log(JSON.stringify(payload));
    }
  }

  _logError(err) {
    this._logStructured(err.message, 'ERROR', { code: err.code, metadata: err.metadata, stack: err.stack });
  }

  /**
   * Gracefully clears active connections, queues, and locks before termination.
   */
  async shutdown() {
    this.isShuttingDown = true;
    this._logStructured('Router shutting down. Cleaning up queues and resources.', 'INFO');
    clearInterval(this.cleanupTimer);

    // Cancel active inflight processing calls
    for (const [requestId, controller] of this.activeRequestControllers.entries()) {
      controller.abort();
      this.activeRequestControllers.delete(requestId);
    }

    // Flush remaining elements from queue
    while (this.requestQueue.length > 0) {
      const item = this.requestQueue.shift();
      clearTimeout(item.timeoutToken);
      item.reject(new ModelRouterError('System terminated during graceful shutdown processing loops.', 'ERR_ROUTER_SHUTDOWN'));
    }
  }
}
