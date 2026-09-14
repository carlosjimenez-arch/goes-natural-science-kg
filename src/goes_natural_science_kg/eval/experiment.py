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
)


async def evaluate_request(
    client: genai.Client | None,
    limiter: asyncio.Semaphore,
    cache: Path,
    settings: ExperimentSettings,
    registry: PromptRegistry,
    prompt_id: str,
    payload: dict[str, Any],
    model: str,
    contract: type[Contract],
    replicate: int,
    revision: int,
    request_locks: dict[str, asyncio.Lock],
) -> EvaluationObservation:
    artifact = registry.get(prompt_id, "1.0.0")
    rendered = registry.render(prompt_id, "1.0.0", {"input": canonical_json(payload)})
    request = {
        "prompt_id": prompt_id,
        "prompt_version": "1.0.0",
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
        "location": settings.location,
    }
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
                        response_json_schema=contract.model_json_schema(),
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
        cost = None
        if incoming is not None and outgoing is not None:
            # Published standard USD list prices <=200K prompt tokens; output includes reasoning.
            rates = (0.30, 0.03, 2.50) if model == "gemini-2.5-flash" else (1.25, 0.125, 10.0)
            if incoming > 200000:
                raise ValueError("pricing tier unsupported by experiment")
            cost = (
                (incoming - (cached or 0)) * rates[0]
                + (cached or 0) * rates[1]
                + (outgoing + (reasoning or 0)) * rates[2]
            ) / 1e6
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
