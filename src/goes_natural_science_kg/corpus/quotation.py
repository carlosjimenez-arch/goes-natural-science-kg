# src/goes_natural_science_kg/corpus/quotation.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Locate quotations across PDF whitespace while preserving exact source offsets.
import re


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Permit whitespace and explicit PDF line-wrap continuations; never fuzzy-match words."""
    exact = text.find(quote)
    if exact >= 0:
        return exact, exact + len(quote)
    # Some provider JSON strings double-escape PDF line breaks. Decode only these
    # whitespace tokens, never arbitrary escape sequences or source characters.
    quote = quote.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    tokens = list(re.finditer(r"\S+", text))
    normalized = " ".join(token.group() for token in tokens)
    wanted = " ".join(quote.split())
    if not wanted:
        return None
    offset = normalized.find(wanted)
    if offset < 0:
        # Restore only word continuations explicitly split by a PDF line ending.
        breaks = list(re.finditer(r"(?<=[A-Za-zÀ-ÿ])-\s*\n\s*(?=[a-zà-ÿ])", text))
        if not breaks:
            return None
        removed = {i for match in breaks for i in range(match.start(), match.end())}
        original_positions = [i for i in range(len(text)) if i not in removed]
        folded = "".join(text[i] for i in original_positions)
        located = locate_quote(folded, quote)
        if located is None:
            return None
        return original_positions[located[0]], original_positions[located[1] - 1] + 1
    positions = []
    for i, token in enumerate(tokens):
        if i:
            positions.append(token.start() - 1)
        positions.extend(range(token.start(), token.end()))
    return positions[offset], positions[offset + len(wanted) - 1] + 1
