# src/goes_natural_science_kg/eval/experiment.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Version and measure evidence-grounded prompt artifacts.
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.schemas.base import Contract, canonical_json, content_hash
from goes_natural_science_kg.schemas.prompt_evaluation import (
    EvaluationObservation,
    ExperimentSettings,
    PromptRegistry,
    RequestSettings,
)

# Published Vertex AI standard list prices in USD per million tokens for prompts <= 200K
# tokens: (input, cached input, output incl. reasoning). Estimates, never invoices.
MODEL_LIST_PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-2.5-flash": (0.30, 0.03, 2.50),
    "gemini-2.5-pro": (1.25, 0.125, 10.0),
    "gemini-2.5-flash-lite": (0.10, 0.01, 0.40),
    "gemini-3-flash-preview": (0.50, 0.05, 3.00),
    "gemini-3.1-pro-preview": (2.00, 0.20, 12.00),
}


async def evaluate_request(  # noqa: C901 - provider transport, deterministic cache and failure accounting
    client: genai.Client | None,
    limiter: asyncio.Semaphore,
    cache: Path,
    settings: RequestSettings,
    registry: PromptRegistry,
    prompt_id: str,
    payload: dict[str, Any],
    model: str,
    contract: type[Contract],
    replicate: int,
    revision: int,
    request_locks: dict[str, asyncio.Lock],
    *,
    prompt_version: str = "1.0.0",
    location: str | None = None,
    json_mode: bool = False,
    schema_override: dict[str, Any] | None = None,
    retry_attempt: int = 0,
) -> EvaluationObservation:
    if location is None:
        # Only the historical single-location experiment omits it; every plan names it.
        if not isinstance(settings, ExperimentSettings):
            raise ValueError("this experiment must name its Vertex location per request")
        location = settings.location
    artifact = registry.get(prompt_id, prompt_version)
    rendered = registry.render(prompt_id, prompt_version, {"input": canonical_json(payload)})
    request = {
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "prompt_sha256": artifact.sha256,
        "prompt": rendered,
        "input": payload,
        "model": model,
        "schema": contract.model_json_schema(),
        "replicate": replicate,
        "revision": revision,
        "seed": settings.seed + replicate,
        "temperature": settings.temperature,
        "thinking_budget": settings.thinking_budget,
        "cache_version": "prompt-experiment/1.0",
        "location": location,
    }
    if json_mode:
        request["output_mode"] = "json-with-local-contract-validation/1.0"
    if schema_override is not None:
        request["transport_schema"] = schema_override
    if retry_attempt:
        request["retry_attempt"] = retry_attempt
    key = content_hash(request)
    async with request_locks.setdefault(key, asyncio.Lock()):
        path = cache / (key + ".json")
        if path.exists() or path.with_suffix(".json.gz").exists():
            record = read_observation(cache, key)
            if record.request_sha256 != key:
                raise ValueError("evaluation cache mismatch")
            return record
        if client is None:
            raise FileNotFoundError("offline evaluation cache miss: " + key)
        async with limiter:
            start = time.perf_counter()
            response = None
            error = None
            result = None
            try:
                result = await client.aio.models.generate_content(
                    model=model,
                    contents=rendered,
                    config=types.GenerateContentConfig(
                        temperature=settings.temperature,
                        seed=settings.seed + replicate,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=settings.thinking_budget
                        ),
                        response_mime_type="application/json",
                        response_json_schema=(
                            None
                            if json_mode
                            else schema_override
                            if schema_override is not None
                            else contract.model_json_schema()
                        ),
                        max_output_tokens=32768,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
                response = result.text
                if response is None:
                    error = "Provider returned no text"
            except Exception as exception:
                # Transport failures are measured separately, never silently assigned zero cost.
                error = type(exception).__name__ + ": " + str(exception)[:300]
            elapsed = time.perf_counter() - start
        usage = result.usage_metadata if result else None
        incoming = usage.prompt_token_count if usage else None
        outgoing = usage.candidates_token_count if usage else None
        reasoning = usage.thoughts_token_count if usage else None
        cached = usage.cached_content_token_count if usage else None
        cost = estimate_cost(model, incoming, outgoing, reasoning, cached)
        record = EvaluationObservation(
            request_sha256=key,
            request=request,
            response=response,
            response_sha256=content_hash(response) if response is not None else None,
            status="ok" if error is None else "failed",
            error=error,
            model_version=result.model_version if result else None,
            latency_seconds=elapsed,
            input_tokens=incoming,
            output_tokens=outgoing,
            reasoning_tokens=reasoning,
            cached_input_tokens=cached,
            estimated_usd=cost,
        )
        atomic_bytes(path, (canonical_json(record) + "\n").encode())
        return record


def estimate_cost(
    model: str,
    incoming: int | None,
    outgoing: int | None,
    reasoning: int | None,
    cached: int | None,
) -> float | None:
    """Published standard USD list prices <=200K prompt tokens; output includes reasoning."""
    if incoming is None or outgoing is None:
        return None
    if model not in MODEL_LIST_PRICES:
        raise ValueError("no list price registered for model " + model)
    if incoming > 200000:
        raise ValueError("pricing tier unsupported by experiment")
    rates = MODEL_LIST_PRICES[model]
    return (
        (incoming - (cached or 0)) * rates[0]
        + (cached or 0) * rates[1]
        + (outgoing + (reasoning or 0)) * rates[2]
    ) / 1e6


def read_observation(directory: Path, key: str) -> EvaluationObservation:
    """Read either live JSON or deterministic compressed audit observations."""
    import gzip

    path = directory / (key + ".json")
    raw = (
        path.read_bytes()
        if path.exists()
        else gzip.decompress(path.with_suffix(".json.gz").read_bytes())
    )
    record = EvaluationObservation.model_validate_json(raw)
    if record.request_sha256 != key:
        raise ValueError("observation filename/request digest mismatch")
    return record
