# src/goes_natural_science_kg/schemas/configuration.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Define provider configuration values without credentials.
"""ADC supplies credentials; no secret-bearing fields enter serialized contracts."""

from goes_natural_science_kg.schemas.base import Contract, Text


class GCPSettings(Contract):
    project_id: Text | None = None
    location: Text = "us-central1"
