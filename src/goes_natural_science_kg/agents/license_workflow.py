# src/goes_natural_science_kg/agents/license_workflow.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Execute the real license gate before exposing an ingestion-ready set.
"""A deterministic LangGraph workflow; no model, fetcher or domain inference."""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from goes_natural_science_kg.corpus.license_gate import evaluate_license
from goes_natural_science_kg.corpus.manifest import write_manifest
from goes_natural_science_kg.schemas.corpus import GateStatus
from goes_natural_science_kg.schemas.workflow import LicenseGateState


def license_gate_node(state: LicenseGateState) -> dict[str, object]:
    results = tuple(
        evaluate_license(doc, state.allowlist, state.checked_at)
        for doc in sorted(state.documents, key=lambda d: d.id)
    )
    return {
        "results": results,
        "ingestion_ready_ids": tuple(
            doc.id for doc in results if doc.license_gate_status == GateStatus.ACCEPTED
        ),
    }


def record_manifest_node(state: LicenseGateState) -> dict[str, object]:
    write_manifest(state.manifest_path, state.results)
    return {}


def build_license_workflow() -> CompiledStateGraph[LicenseGateState, Any, Any, Any]:
    builder = StateGraph(LicenseGateState)
    builder.add_node("license_gate", license_gate_node)
    builder.add_node("record_manifest", record_manifest_node)
    builder.add_edge(START, "license_gate")
    builder.add_edge("license_gate", "record_manifest")
    builder.add_edge("record_manifest", END)
    return builder.compile()
