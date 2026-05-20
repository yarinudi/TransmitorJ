"""Thin Streamlit-cached wrapper around WHSPreprocessor + feature extractors."""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime
from typing import Any

import numpy as np
import streamlit as st

from whs_preprocessor import (
    MacroFeatureExtractor,
    MicroFeatureExtractor,
    PreprocessConfig,
    WHSPreprocessor,
)


def default_config_dict() -> dict[str, Any]:
    """A plain dict snapshot of PreprocessConfig defaults — used to seed the UI."""
    return asdict(PreprocessConfig())


def config_field_specs() -> list[tuple[str, type, Any]]:
    """(name, type, default) for every PreprocessConfig field. Drives the editor UI.

    We derive the type from the default value's runtime type instead of
    ``dataclasses.Field.type`` because the preprocessor uses
    ``from __future__ import annotations`` — that turns every annotation into a
    string ("int", "float", ...), which breaks ``type_ is int`` checks.
    """
    cfg = PreprocessConfig()
    return [(f.name, type(getattr(cfg, f.name)), getattr(cfg, f.name)) for f in fields(cfg)]


@st.cache_data(show_spinner="Running preprocessor...")
def run_pipeline(
    subject_id: str,
    _data: np.ndarray,
    start_datetime: datetime,
    config_dict: dict,
) -> dict:
    """Run the full pipeline + feature extractors.

    The leading underscore on ``_data`` tells Streamlit's hasher to skip it —
    the cache key is (subject_id, start_datetime, config_dict).
    """
    cfg = PreprocessConfig(**config_dict)
    pre = WHSPreprocessor(cfg)
    result = pre.process(_data, start_datetime=start_datetime, subject_id=subject_id)
    if result["valid"]:
        result["macro_features"] = MacroFeatureExtractor(cfg).extract_all(result["macro"])
        result["micro_features"] = MicroFeatureExtractor(cfg).extract_all(result["micro"])
    return result
