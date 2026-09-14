# src/goes_natural_science_kg/corpus/normalize.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Normalize whitespace without losing the original character mapping.
"""No dehyphenation, Unicode folding or OCR correction silently changes evidence."""

import re

from goes_natural_science_kg.schemas.ingestion import NormalizedText


def normalize(text: str) -> NormalizedText:
    values: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\s+|\S", text):
        value = match.group()
        values.append(" " if value.isspace() else value)
        spans.append(match.span())
    return NormalizedText(
        text="".join(values), original_spans=tuple(spans), original_length=len(text)
    )
