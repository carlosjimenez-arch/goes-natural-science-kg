# src/goes_natural_science_kg/config.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Read namespaced configuration without requiring credentials at import.
"""Instantiation is explicit; legacy sibling .env values are ignored."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from goes_natural_science_kg.schemas.base import PositiveMinutes
from goes_natural_science_kg.schemas.configuration import GCPSettings
from goes_natural_science_kg.schemas.sequencing import SequencingSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GOES_NATURAL_SCIENCE_KG_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )
    data_dir: Path = Path("data")
    session_minutes: PositiveMinutes | None = None
    gcp: GCPSettings = Field(default_factory=GCPSettings)
    sequencing: SequencingSettings = Field(default_factory=SequencingSettings)
