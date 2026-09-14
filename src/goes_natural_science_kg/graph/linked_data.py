# src/goes_natural_science_kg/graph/linked_data.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Export context-free nodes and typed edges as lossless JSON-LD 1.1.
from typing import Any

from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.graph import EvidenceGraphSnapshot, GraphSnapshot
from goes_natural_science_kg.schemas.sequencing import LinkedGraph
from goes_natural_science_kg.schemas.skills import Edge, MicroSkill


def export_jsonld(graph: GraphSnapshot) -> LinkedGraph:
    prefix = "urn:goes:science:node:"
    nodes: list[dict[str, Any]] = [
        {
            "@id": "urn:goes:science:graph:" + content_hash(graph),
            "@type": "GraphVersion",
            "payload": {
                "schema_version": graph.schema_version,
                "version": graph.version,
                "source_ids": list(graph.source_ids),
            },
        }
    ]
    if isinstance(graph, EvidenceGraphSnapshot):
        nodes[0]["payload"]["chunk_sources"] = graph.chunk_sources
    for entity in (*graph.skills, *graph.micro_skills, *graph.edges):
        node = {
            "@id": prefix + entity.id,
            "@type": type(entity).__name__,
            "payload": entity.model_dump(mode="json"),
        }
        if isinstance(entity, Edge):
            node.update(
                source=prefix + entity.source,
                target=prefix + entity.target,
                relation=entity.type.value,
            )
        elif isinstance(entity, MicroSkill):
            node.update(parent=prefix + entity.parent_skill_id)
        if hasattr(entity, "source_refs"):
            node["evidence"] = [prefix + ref for ref in entity.source_refs]
        nodes.append(node)
    if isinstance(graph, EvidenceGraphSnapshot):
        nodes.extend(
            {"@id": prefix + chunk, "@type": "EvidenceChunkReference", "source": prefix + source}
            for chunk, source in sorted(graph.chunk_sources.items())
        )
    nodes.extend(
        {"@id": prefix + source, "@type": "SourceDocumentReference"} for source in graph.source_ids
    )
    return LinkedGraph.model_validate(
        {
            "@context": {
                "@version": 1.1,
                "@vocab": "urn:goes:science:vocab:",
                "payload": {"@type": "@json"},
                "source": {"@type": "@id"},
                "target": {"@type": "@id"},
                "parent": {"@type": "@id"},
                "evidence": {"@type": "@id", "@container": "@set"},
            },
            "@graph": nodes,
        }
    )


def import_jsonld(document: LinkedGraph) -> GraphSnapshot:
    metadata = next(n["payload"] for n in document.nodes if n["@type"] == "GraphVersion")
    values = {**metadata, "skills": [], "micro_skills": [], "edges": []}
    fields = {
        "Skill": "skills",
        "MicroSkill": "micro_skills",
        "EvidenceReadyMicroSkill": "micro_skills",
        "Edge": "edges",
    }
    for node in document.nodes:
        if node["@type"] in fields:
            values[fields[node["@type"]]].append(node["payload"])
    return (
        EvidenceGraphSnapshot.model_validate(values)
        if values["schema_version"] == "graph-snapshot/2.0"
        else GraphSnapshot.model_validate(values)
    )
