# tests/property/test_quotation.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Prove whitespace-only quote repair maps back to unchanged source words.
from hypothesis import given
from hypothesis import strategies as st

from goes_natural_science_kg.corpus.quotation import locate_quote


@given(st.lists(st.sampled_from([" ", "  ", "\n", "\t", "\r\n"]), min_size=3, max_size=3))
def test_source_offsets_survive_pdf_whitespace(separators):
    words = ["Evidence", "supports", "bounded", "claims."]
    text = (
        "prefix\n"
        + "".join(word + (separators[i] if i < 3 else "") for i, word in enumerate(words))
        + "\nsuffix"
    )
    quote = " ".join(words)
    start, end = locate_quote(text, quote)
    assert " ".join(text[start:end].split()) == quote
    assert locate_quote(text, "Evidence proves every claim.") is None


def test_double_escaped_pdf_newline_is_not_a_fabricated_word():
    text = "Plants transport\nwater through stems."
    start, end = locate_quote(text, "transport\\nwater")
    assert text[start:end] == "transport\nwater"
    assert locate_quote(text, "transport\\nrocks") is None


def test_pdf_word_wrap_preserves_source_offsets_and_rejects_changed_words():
    text = "Enfermedades infec-\nciosas y meta-\nbolismo."
    start, end = locate_quote(text, "infecciosas y metabolismo")
    assert text[start:end] == "infec-\nciosas y meta-\nbolismo"
    assert locate_quote(text, "infecciosas sin metabolismo") is None
