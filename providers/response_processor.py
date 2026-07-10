import json
import math
import time
from datetime import datetime
from typing import Any, AsyncIterable, AsyncGenerator, Dict, List, Optional


class EventEmitter:
    """A lightweight Python implementation of Node's EventEmitter."""
    def __init__(self):
        self._listeners: Dict[str, List[Any]] = {}

    def on(self, event: str, listener: Any):
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)
        return self

    def emit(self, event: str, *args, **kwargs):
        if event in self._listeners:
            for listener in self._listeners[event]:
                listener(*args, **kwargs)


class ResponseProcessorError(Exception):
    """Custom error class for ResponseProcessor operations."""
    def __init__(self, message: str, code: str, metadata: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.name = 'ResponseProcessorError'


class ResponseProcessor(EventEmitter):
    """
    Enterprise response processor for unifying, validating, and enriching
    AI provider outputs within the Cabangile AI Studio orchestration engine.
    """
    def __init__(self, dependencies: Optional[Dict] = None, config: Optional[Dict] = None):
        super().__init__()
        dependencies = dependencies or {}
        config = config or {}

        self.provider_manager = dependencies.get("providerManager")
        self.model_registry = dependencies.get("modelRegistry")
        self.model_router = dependencies.get("modelRouter")
        self.request_executor = dependencies.get("requestExecutor")

        self.config = {
            "enableJsonRepair": True,
            "maxHistorySize": 1000,
            "defaultTokenCostPerK": 0.002,
            "validationStrict": False,
            **config
        }

        self.stats = {
            "totalProcessed": 0,
            "successfulResponses": 0,
            "failedResponses": 0,
            "totalProcessingTimeMs": 0.0,
            "totalInputTokens": 0,
            "totalOutputTokens": 0,
            "totalEstimatedCost": 0.0,
            "providerPerformance": {}
        }

        self.history: List[Dict] = []
        self.errors: List[ResponseProcessorError] = []

    def normalize(self, raw_response: Any, provider_id: str, model_id: str) -> Dict:
        """Normalizes an external raw provider response into a standard internal format."""
        start_time = time.perf_counter_ns()
        self.emit('responseReceived', providerId=provider_id, modelId=model_id, timestamp=datetime.utcnow().isoformat() + "Z")

        if raw_response is None:
            error = ResponseProcessorError(
                'Received null or undefined raw response', 
                'EMPTY_RESPONSE', 
                {"providerId": provider_id, "modelId": model_id}
            )
            self._track_failure(error, provider_id)
            self.emit('validationFailed', error=error, providerId=provider_id, modelId=model_id)
            raise error

        try:
            self._detect_provider_error(raw_response, provider_id)

            normalized = None
            p_id = provider_id.lower()

            if p_id == 'openai':
                normalized = self._normalize_openai(raw_response, model_id)
            elif p_id == 'anthropic':
                normalized = self._normalize_anthropic(raw_response, model_id)
            elif p_id in ('google', 'gemini'):
                normalized = self._normalize_gemini(raw_response, model_id)
            elif p_id == 'mistral':
                normalized = self._normalize_mistral(raw_response, model_id)
            elif p_id == 'openrouter':
                normalized = self._normalize_openrouter(raw_response, model_id)
            elif p_id == 'ollama':
                normalized = self._normalize_ollama(raw_response, model_id)
            else:
                normalized = self._normalize_custom(raw_response, provider_id, model_id)

            self._validate_schema(normalized)

            if normalized.get("toolCalls") and len(normalized["toolCalls"]) > 0:
                self.emit('toolCallDetected', toolCalls=normalized["toolCalls"], providerId=provider_id, modelId=model_id)

            enrichments = self._calculate_token_metrics(normalized, model_id)
            normalized["usage"] = {**normalized.get("usage", {}), **enrichments["usage"]}
            normalized["metadata"] = {**normalized.get("metadata", {}), "cost": enrichments["cost"]}

            end_time = time.perf_counter_ns()
            processing_time = float(end_time - start_time) / 1_000_000.0

            self._track_success(normalized, processing_time)
            self.emit('responseProcessed', normalized)

            return normalized
        except Exception as error:
            processed_error = error if isinstance(error, ResponseProcessorError) else ResponseProcessorError(
                str(error), 
                'PROCESSING_FAILED', 
                {"originalError": str(error), "providerId": provider_id, "modelId": model_id}
            )
            self._track_failure(processed_error, provider_id)
            raise processed_error

    async def process_stream(self, stream_source: AsyncIterable, provider_id: str, model_id: str) -> AsyncGenerator[Dict, None]:
        """Processes standard streaming chunks using an async generator approach."""
        self.emit('streamStarted', providerId=provider_id, modelId=model_id, timestamp=datetime.utcnow().isoformat() + "Z")

        accumulated_content = ''
        accumulated_tool_calls = []
        final_usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        final_finish_reason = None
        original_id = f"stream_{int(time.time()*1000)}"

        try:
            async for raw_chunk in stream_source:
                chunk_data = self._extract_stream_chunk(raw_chunk, provider_id)

                if chunk_data.get("id"):
                    original_id = chunk_data["id"]
                if chunk_data.get("content"):
                    accumulated_content += chunk_data["content"]
                if chunk_data.get("finishReason"):
                    final_finish_reason = chunk_data["finishReason"]
                if chunk_data.get("usage"):
                    final_usage = {**final_usage, **chunk_data["usage"]}

                if chunk_data.get("toolCalls") and len(chunk_data["toolCalls"]) > 0:
                    accumulated_tool_calls = self._merge_tool_calls(accumulated_tool_calls, chunk_data["toolCalls"])

                yield {
                    "id": original_id,
                    "providerId": provider_id,
                    "modelId": model_id,
                    "type": 'stream_chunk',
                    "content": chunk_data.get("content") or '',
                    "choices": [{"text": chunk_data.get("content") or '', "index": 0}],
                    "usage": None,
                    "finishReason": chunk_data.get("finishReason"),
                    "toolCalls": chunk_data.get("toolCalls") or [],
                    "metadata": {"partial": True},
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }

            complete_normalized = {
                "id": original_id,
                "providerId": provider_id,
                "modelId": model_id,
                "type": 'text',
                "content": accumulated_content,
                "choices": [{"text": accumulated_content, "index": 0}],
                "usage": final_usage,
                "finishReason": final_finish_reason or 'stop',
                "toolCalls": accumulated_tool_calls,
                "metadata": {"assembledFromStream": True},
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            enrichments = self._calculate_token_metrics(complete_normalized, model_id)
            complete_normalized["usage"] = {**complete_normalized["usage"], **enrichments["usage"]}
            complete_normalized["metadata"]["cost"] = enrichments["cost"]

            self._track_success(complete_normalized, 0.0)
            self.emit('streamCompleted', complete_normalized)

            yield complete_normalized

        except Exception as error:
            stream_error = ResponseProcessorError(
                f"Stream handling interruption: {str(error)}", 
                'STREAM_PROCESSING_ERROR', 
                {"originalError": str(error), "providerId": provider_id, "modelId": model_id}
            )
            self.emit('responseFailed', stream_error)
            raise stream_error

    def extract_content_payload(self, normalized_response: Dict) -> Dict:
        """Extracts content from structured properties and handles internal variations like JSON payloads."""
        payload = {
            "text": normalized_response.get("content"),
            "json": None,
            "images": normalized_response.get("metadata", {}).get("images") or [],
            "audio": normalized_response.get("metadata", {}).get("audio") or [],
            "embeddings": normalized_response.get("metadata", {}).get("embeddings") or [],
            "toolCalls": normalized_response.get("toolCalls") or []
        }

        content = normalized_response.get("content")
        if content:
            stripped = content.strip()
            if stripped.startswith('{') or stripped.startswith('['):
                try:
                    payload["json"] = json.loads(content)
                except Exception:
                    if self.config.get("enableJsonRepair"):
                        payload["json"] = self._repair_and_parse_json(content)

        return payload

    def _normalize_openai(self, raw: Any, model_id: str) -> Dict:
        choices = raw.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        
        return {
            "id": raw.get("id") or f"omni_{int(time.time()*1000)}",
            "providerId": 'openai',
            "modelId": raw.get("model") or model_id,
            "type": 'text' if message.get("content") else ('tool_call' if message.get("tool_calls") else 'unknown'),
            "content": message.get("content") or '',
            "choices": [{"text": c.get("message", {}).get("content") or '', "index": c.get("index")} for c in choices],
            "usage": {
                "inputTokens": raw.get("usage", {}).get("prompt_tokens") or 0,
                "outputTokens": raw.get("usage", {}).get("completion_tokens") or 0,
                "totalTokens": raw.get("usage", {}).get("total_tokens") or 0
            },
            "finishReason": choice.get("finish_reason"),
            "toolCalls": self._parse_openai_tool_calls(message.get("tool_calls")),
            "metadata": {"systemFingerprint": raw.get("system_fingerprint")},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _normalize_anthropic(self, raw: Any, model_id: str) -> Dict:
        raw_content = raw.get("content", [])
        text_content = '\n'.join([c.get("text") for c in raw_content if c.get("type") == 'text'])
        
        tool_calls = []
        for c in raw_content:
            if c.get("type") == 'tool_use':
                tool_calls.append({
                    "id": c.get("id"),
                    "type": 'function',
                    "function": {
                        "name": c.get("name"),
                        "arguments": json.dumps(c.get("input") or {})
                    }
                })

        usage = raw.get("usage", {})
        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0

        return {
            "id": raw.get("id") or f"anth_{int(time.time()*1000)}",
            "providerId": 'anthropic',
            "modelId": raw.get("model") or model_id,
            "type": 'tool_call' if len(tool_calls) > 0 else 'text',
            "content": text_content,
            "choices": [{"text": text_content, "index": 0}],
            "usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": input_tokens + output_tokens
            },
            "finishReason": raw.get("stop_reason"),
            "toolCalls": tool_calls,
            "metadata": {"role": raw.get("role")},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _normalize_gemini(self, raw: Any, model_id: str) -> Dict:
        candidates = raw.get("candidates", [])
        candidate = candidates[0] if candidates else {}
        parts = candidate.get("content", {}).get("parts", [])

        content = "\n".join(
            p.get("text", "")
            for p in parts
            if p.get("text")
        )
        tool_calls = []

        raw_tool_calls = [p for p in parts if p.get("functionCall")]
        if len(raw_tool_calls) > 0:
            for idx, tc in enumerate(raw_tool_calls):
                tool_calls.append({
                    "id": f"gemini_tc_{idx}_{int(time.time()*1000)}",
                    "type": 'function',
                    "function": {
                        "name": tc["functionCall"].get("name"),
                        "arguments": json.dumps(tc["functionCall"].get("args") or {})
                    }
                })

        usage = raw.get("usageMetadata", {})
        return {
            "id": f"gemini_{int(time.time()*1000)}",
            "providerId": 'google',
            "modelId": model_id,
            "type": 'tool_call' if len(tool_calls) > 0 else 'text',
            "content": content,
            "choices": [{"text": content, "index": 0}],
            "usage": {
                "inputTokens": usage.get("promptTokenCount") or 0,
                "outputTokens": usage.get("candidatesTokenCount") or 0,
                "totalTokens": usage.get("totalTokenCount") or 0
            },
            "finishReason": candidate.get("finishReason"),
            "toolCalls": tool_calls,
            "metadata": {},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _normalize_mistral(self, raw: Any, model_id: str) -> Dict:
        choices = raw.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})
        
        return {
            "id": raw.get("id") or f"mistral_{int(time.time()*1000)}",
            "providerId": 'mistral',
            "modelId": raw.get("model") or model_id,
            "type": 'text' if message.get("content") else 'unknown',
            "content": message.get("content") or '',
            "choices": [{"text": c.get("message", {}).get("content") or '', "index": c.get("index")} for c in choices],
            "usage": {
                "inputTokens": raw.get("usage", {}).get("prompt_tokens") or 0,
                "outputTokens": raw.get("usage", {}).get("completion_tokens") or 0,
                "totalTokens": raw.get("usage", {}).get("total_tokens") or 0
            },
            "finishReason": choice.get("finish_reason"),
            "toolCalls": self._parse_openai_tool_calls(message.get("tool_calls")),
            "metadata": {},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _normalize_openrouter(self, raw: Any, model_id: str) -> Dict:
        if isinstance(raw, dict) and "choices" in raw:
            mapped = self._normalize_openai(raw, model_id)
            mapped["providerId"] = 'openrouter'
            return mapped
        return self._normalize_custom(raw, 'openrouter', model_id)

    def _normalize_ollama(self, raw: Any, model_id: str) -> Dict:
        input_eval = raw.get("prompt_eval_count") or 0
        output_eval = raw.get("eval_count") or 0
        return {
            "id": f"ollama_{int(time.time()*1000)}",
            "providerId": 'ollama',
            "modelId": raw.get("model") or model_id,
            "type": 'text',
            "content": raw.get("response") or '',
            "choices": [{"text": raw.get("response") or '', "index": 0}],
            "usage": {
                "inputTokens": input_eval,
                "outputTokens": output_eval,
                "totalTokens": input_eval + output_eval
            },
            "finishReason": 'stop' if raw.get("done") else None,
            "toolCalls": [],
            "metadata": {},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _normalize_custom(self, raw: Any, provider_id: str, model_id: str) -> Dict:
        text = ""
        if isinstance(raw, dict):
            text = raw.get("text") or raw.get("content") or raw.get("output") or json.dumps(raw)
        else:
            text = str(raw)
        return {
            "id": f"custom_{int(time.time()*1000)}",
            "providerId": provider_id,
            "modelId": model_id,
            "type": 'text',
            "content": text,
            "choices": [{"text": text, "index": 0}],
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "finishReason": 'unknown',
            "toolCalls": [],
            "metadata": {"customProviderRaw": raw},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _extract_stream_chunk(self, raw_chunk: Any, provider_id: str) -> Dict:
        p_id = provider_id.lower()

        if isinstance(raw_chunk, dict) and p_id in ('openai', 'mistral', 'openrouter'):
            choices = raw_chunk.get("choices", [])
            choice = choices[0] if choices else {}
            return {
                "id": raw_chunk.get("id"),
                "content": choice.get("delta", {}).get("content") or '',
                "finishReason": choice.get("finish_reason"),
                "toolCalls": self._parse_openai_tool_calls(choice.get("delta", {}).get("tool_calls")),
                "usage": {
                    "inputTokens": raw_chunk["usage"].get("prompt_tokens") or 0,
                    "outputTokens": raw_chunk["usage"].get("completion_tokens") or 0,
                    "totalTokens": raw_chunk["usage"].get("total_tokens") or 0
                } if raw_chunk.get("usage") else None
            }

        if isinstance(raw_chunk, dict) and p_id == 'anthropic':
            content = ''
            tool_calls = []

            chunk_type = raw_chunk.get("type")
            if chunk_type == 'content_block_delta' and raw_chunk.get("delta", {}).get("text"):
                content = raw_chunk["delta"]["text"]
            
            if chunk_type == 'content_block_start' and raw_chunk.get("content_block", {}).get("type") == 'tool_use':
                tool_calls.append({
                    "id": raw_chunk["content_block"].get("id"),
                    "type": 'function',
                    "function": {"name": raw_chunk["content_block"].get("name"), "arguments": ''}
                })
            
            if chunk_type == 'content_block_delta' and raw_chunk.get("delta", {}).get("partial_json"):
                tool_calls.append({
                    "index": raw_chunk.get("index"),
                    "function": {"arguments": raw_chunk["delta"]["partial_json"]}
                })

            usage = None
            if chunk_type == 'message_start' and raw_chunk.get("message", {}).get("usage"):
                anth_usage = raw_chunk["message"]["usage"]
                in_t = anth_usage.get("input_tokens") or 0
                out_t = anth_usage.get("output_tokens") or 0
                usage = {"inputTokens": in_t, "outputTokens": out_t, "totalTokens": in_t + out_t}

            return {
                "id": None,
                "content": content,
                "finishReason": raw_chunk.get("delta", {}).get("stop_reason") if chunk_type == 'message_delta' else None,
                "toolCalls": tool_calls,
                "usage": usage
            }

        if isinstance(raw_chunk, dict) and p_id in ('google', 'gemini'):
            candidates = raw_chunk.get("candidates", [])
            candidate = candidates[0] if candidates else {}
            parts = candidate.get("content", {}).get("parts", [])
            part_content = "\n".join(p.get("text", "") for p in parts if p.get("text"))
            return {
                "id": None,
                "content": part_content,
                "finishReason": candidate.get("finishReason"),
                "toolCalls": []
            }

        if isinstance(raw_chunk, dict) and p_id == 'ollama':
            return {
                "id": None,
                "content": raw_chunk.get("response") or '',
                "finishReason": 'stop' if raw_chunk.get("done") else None,
                "toolCalls": []
            }

        return {
            "id": None,
            "content": raw_chunk if isinstance(raw_chunk, str) else json.dumps(raw_chunk),
            "finishReason": None,
            "toolCalls": []
        }

    def _parse_openai_tool_calls(self, raw_tool_calls: Any) -> List[Dict]:
        if not raw_tool_calls or not isinstance(raw_tool_calls, list):
            return []
        parsed = []
        for tc in raw_tool_calls:
            if not isinstance(tc, dict) or not tc.get("function"):
                continue
            parsed.append({
                "id": tc.get("id"),
                "type": tc.get("type") or 'function',
                "function": {
                    "name": tc["function"].get("name"),
                    "arguments": tc["function"].get("arguments")
                }
            })
        return parsed

    def _merge_tool_calls(self, existing: List[Dict], incoming: List[Dict]) -> List[Dict]:
        base = list(existing)
        for inc in incoming:
            if not isinstance(inc, dict):
                continue
            if inc.get("id"):
                base.append(inc)
            elif isinstance(inc.get("index"), int) and inc["index"] < len(base):
                idx = inc["index"]
                base[idx].setdefault("function", {})
                base[idx]["function"]["arguments"] = (
                    base[idx]["function"].get("arguments", "")
                    + inc["function"].get("arguments", "")
                )
        return base

    def _validate_schema(self, struct: Dict):
        required = ['id', 'providerId', 'modelId', 'type', 'choices', 'timestamp']
        for field in required:
            if struct.get(field) is None:
                error = ResponseProcessorError(
                    f'Missing required standard response property: "{field}"', 
                    'SCHEMA_VALIDATION_FAILED', 
                    {"schema": struct}
                )
                self.emit('validationFailed', error=error, payload=struct)
                raise error

        if self.config.get("validationStrict") and (not struct.get("content") and (not struct.get("toolCalls") or len(struct["toolCalls"]) == 0)):
            error = ResponseProcessorError(
                'Strict mode enforcement payload failure: missing extractable content output or active functions.', 
                'EMPTY_PAYLOAD_VALIDATION_FAILED', 
                {"schema": struct}
            )
            self.emit('validationFailed', error=error, payload=struct)
            raise error

    def _detect_provider_error(self, raw: Any, provider_id: str):
        p_id = provider_id.lower()

        if isinstance(raw, dict) and p_id == 'openai' and raw.get("error"):
            throw_err = raw["error"] if isinstance(raw["error"], dict) else {"message": str(raw["error"])}
            raise ResponseProcessorError(throw_err.get("message") or 'OpenAI backend exception context received', 'PROVIDER_ERROR_OPENAI', throw_err)
        
        if isinstance(raw, dict) and p_id == 'anthropic' and raw.get("type") == 'error':
            throw_err = raw.get("error") or {}
            raise ResponseProcessorError(throw_err.get("message") or 'Anthropic API processing fault', 'PROVIDER_ERROR_ANTHROPIC', throw_err)
        
        if isinstance(raw, dict) and p_id in ('google', 'gemini'):
            if raw.get("promptFeedback", {}).get("blockReason"):
                raise ResponseProcessorError(f"Google request blocks generated: {raw['promptFeedback']['blockReason']}", 'PROVIDER_ERROR_GEMINI_BLOCKED', raw["promptFeedback"])

    def _calculate_token_metrics(self, normalized: Dict, model_id: str) -> Dict:
        input_tokens = normalized.get("usage", {}).get("inputTokens") or 0
        output_tokens = normalized.get("usage", {}).get("outputTokens") or 0

        if input_tokens == 0 and normalized.get("content"):
            input_tokens = math.ceil(len(normalized["content"]) / 4)
        if output_tokens == 0 and normalized.get("content"):
            output_tokens = math.ceil(len(normalized["content"]) / 4)

        base_pricing_k = self.config.get("defaultTokenCostPerK", 0.002)
        if self.model_registry and hasattr(self.model_registry, 'get_model_pricing'):
            model_pricing = self.model_registry.get_model_pricing(model_id)
            if model_pricing:
                calculated_cost = (input_tokens * model_pricing.get("inputPerToken", 0.0)) + (output_tokens * model_pricing.get("outputPerToken", 0.0))
                return {
                    "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens, "totalTokens": input_tokens + output_tokens},
                    "cost": calculated_cost
                }

        estimated_cost = ((input_tokens + output_tokens) / 1000.0) * base_pricing_k
        return {
            "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens, "totalTokens": input_tokens + output_tokens},
            "cost": estimated_cost
        }

    def _repair_and_parse_json(self, malformed_str: str) -> Any:
        clean_str = malformed_str.strip()

        if '```json' in clean_str:
            clean_str = clean_str.split('```json')[1].split('```')[0].strip()
        elif '```' in clean_str:
            clean_str = clean_str.split('```')[1].split('```')[0].strip()

        try:
            return json.loads(clean_str)
        except Exception:
            try:
                if not clean_str.endswith('}') and not clean_str.endswith(']'):
                    brackets = []
                    for char in clean_str:
                        if char in ('{', '['):
                            brackets.append(char)
                        elif char == '}':
                            if brackets and brackets[-1] == '{':
                                brackets.pop()
                        elif char == ']':
                            if brackets and brackets[-1] == '[':
                                brackets.pop()
                    while brackets:
                        target = brackets.pop()
                        clean_str += '}' if target == '{' else ']'
                return json.loads(clean_str)
            except Exception:
                raise ResponseProcessorError(
                    'Execution target JSON extraction strategy parameters unresolvable.', 
                    'MALFORMED_JSON_PARSE_FAILURE', 
                    {"payloadSource": malformed_str}
                )

    def _track_success(self, normalized: Dict, duration_ms: float):
        self.stats["totalProcessed"] += 1
        self.stats["successfulResponses"] += 1
        self.stats["totalProcessingTimeMs"] += duration_ms
        self.stats["totalInputTokens"] += normalized.get("usage", {}).get("inputTokens") or 0
        self.stats["totalOutputTokens"] += normalized.get("usage", {}).get("outputTokens") or 0
        self.stats["totalEstimatedCost"] += normalized.get("metadata", {}).get("cost") or 0.0

        provider = normalized.get("providerId", "unknown")
        if provider not in self.stats["providerPerformance"]:
            self.stats["providerPerformance"][provider] = {"count": 0, "totalTime": 0.0, "errors": 0}
        
        self.stats["providerPerformance"][provider]["count"] += 1
        self.stats["providerPerformance"][provider]["totalTime"] += duration_ms

        self.history.append({
            "id": normalized.get("id"),
            "status": 'success',
            "providerId": normalized.get("providerId"),
            "modelId": normalized.get("modelId"),
            "timestamp": normalized.get("timestamp")
        })

        if len(self.history) > self.config.get("maxHistorySize", 1000):
            self.history.pop(0)

    def _track_failure(self, error: ResponseProcessorError, provider_id: Optional[str]):
        self.stats["totalProcessed"] += 1
        self.stats["failedResponses"] += 1

        if provider_id:
            if provider_id not in self.stats["providerPerformance"]:
                self.stats["providerPerformance"][provider_id] = {"count": 0, "totalTime": 0.0, "errors": 0}
            self.stats["providerPerformance"][provider_id]["errors"] += 1

        self.errors.append(error)
        if len(self.errors) > 100:
            self.errors.pop(0)

        self.history.append({
            "id": error.metadata.get("id") or f"err_{int(time.time()*1000)}",
            "status": 'failed',
            "providerId": provider_id,
            "modelId": error.metadata.get("modelId"),
            "timestamp": error.timestamp,
            "errorCode": error.code
        })

        if len(self.history) > self.config.get("maxHistorySize", 1000):
            self.history.pop(0)

        self.emit('responseFailed', error)

    def export_configuration(self) -> str:
        return json.dumps({"config": self.config})

    def import_configuration(self, config_json: str):
        parsed = json.loads(config_json)
        if parsed and parsed.get("config"):
            self.config = {**self.config, **parsed["config"]}

    def create_snapshot(self) -> Dict:
        return {
            "stats": {**self.stats},
            "history": list(self.history),
            "errors": [
                {"message": e.message, "code": e.code, "metadata": e.metadata, "timestamp": e.timestamp}
                for e in self.errors
            ],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def restore_snapshot(self, snapshot: Dict):
        if snapshot:
            self.stats = {**snapshot.get("stats", {})}
            self.history = list(snapshot.get("history", []))
            self.errors = []
            for e in snapshot.get("errors", []):
                err = ResponseProcessorError(e["message"], e["code"], e["metadata"])
                err.timestamp = e["timestamp"]
                self.errors.append(err)

    def backup(self) -> str:
        return json.dumps(self.create_snapshot())

    def restore(self, backup_str: str):
        self.restore_snapshot(json.loads(backup_str))

    def run_diagnostics(self) -> Dict:
        active_providers = self.stats["providerPerformance"].keys()
        calculations = {}

        for prov in active_providers:
            entry = self.stats["providerPerformance"][prov]
            total_runs = entry["count"] + entry["errors"]
            calculations[prov] = {
                "avgResponseTimeMs": entry["totalTime"] / entry["count"] if entry["count"] > 0 else 0.0,
                "errorRate": entry["errors"] / total_runs if total_runs > 0 else 0.0
            }

        failure_ratio = self.stats["failedResponses"] / max(self.stats["totalProcessed"], 1)
        status = 'DEGRADED' if failure_ratio > 0.5 else 'HEALTHY'

        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processedCount": self.stats["totalProcessed"],
            "failureCount": self.stats["failedResponses"],
            "memoryStatistics": {
                "rssMs": "N/A (Platform Dependent)", 
                "note": "Use psutil package in python for precise RSS memory analysis"
            },
            "performanceMetrics": {
                "overallAvgTimeMs": self.stats["totalProcessingTimeMs"] / self.stats["successfulResponses"] if self.stats["successfulResponses"] > 0 else 0.0,
                "providerBreakdown": calculations,
                "accumulatedCost": self.stats["totalEstimatedCost"]
            }
        }


__all__ = [
    "ResponseProcessor",
    "ResponseProcessorError",
    "EventEmitter"
]
