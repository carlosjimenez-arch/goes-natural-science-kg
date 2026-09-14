# src/goes_natural_science_kg/agents/transport.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
import asyncio
import json
from typing import Any

from google.genai import types

from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.eval.registry import load_registry
from goes_natural_science_kg.schemas.base import Contract, canonical_json, content_hash
from goes_natural_science_kg.schemas.orchestration import AgentRuntime, CachedGeneration


async def generate(
    runtime: AgentRuntime,
    prompt_name: str,
    model: str,
    payload: dict[str, object],
    contract: type[Contract],
) -> Contract:
    registry = load_registry(runtime.prompts, recursive=False)
    artifact = next(
        a
        for a in registry.artifacts
        if a.path == str(runtime.prompts / (prompt_name + ".v1.prompt"))
    )
    metadata = artifact.metadata
    rendered = registry.render(
        metadata.id,
        metadata.version,
        {
            "output_schema_version": str(contract.model_fields["schema_version"].default),
            "input": canonical_json(payload),
        },
    )
    request = {
        "prompt_id": metadata.id,
        "prompt_version": metadata.version,
        "prompt": rendered,
        "input": payload,
        "model": model,
        "schema": contract.model_json_schema(),
        "seed": runtime.settings.seed,
        "temperature": 0,
        "cache_version": runtime.settings.cache_version,
        "location": runtime.settings.location,
    }
    key = content_hash(request)
    lock = runtime.request_locks.setdefault(key, asyncio.Lock())
    async with lock:
        record = await generation_record(runtime, request, key, model, rendered, contract)
    return contract.model_validate(json.loads(record.response))


async def generation_record(
    runtime: AgentRuntime,
    request: dict[str, Any],
    key: str,
    model: str,
    rendered: str,
    contract: type[Contract],
) -> CachedGeneration:
    path = runtime.cache / (key + ".json")
    if path.exists():
        record = CachedGeneration.model_validate_json(path.read_bytes())
        if record.request_sha256 != key:
            raise ValueError("cache request mismatch")
    else:
        if runtime.client is None:
            raise FileNotFoundError("offline generation cache miss: " + key)
        async with runtime.limiter:
            result = await runtime.client.aio.models.generate_content(
                model=model,
                contents=rendered,
                config=types.GenerateContentConfig(
                    temperature=0,
                    seed=runtime.settings.seed,
                    response_mime_type="application/json",
                    response_json_schema=contract.model_json_schema(),
                    max_output_tokens=16384,
                ),
            )
        response = result.text
        if response is None:
            raise ValueError("provider returned no structured content")
        record = CachedGeneration(
            request=request,
            request_sha256=key,
            response=response,
            response_sha256=content_hash(response),
            provider_model_version=result.model_version or model,
        )
        # Preserve invalid model output as evidence too; validation belongs to the local loop.
        atomic_bytes(path, (canonical_json(record) + "\n").encode())
    return record
