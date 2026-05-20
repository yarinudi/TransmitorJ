"""Pipeline orchestration shared by the Streamlit and Dash apps.

``run_pipeline_core`` is the framework-agnostic implementation. The cached
``run_pipeline`` wrapper sits on top of it via Streamlit's ``@cache_data`` —
when Streamlit is missing the decorator becomes a no-op and the Dash app
adds its own caching at the call site.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime
from typing import Any

import numpy as np

from whs_preprocessor import (
    MacroFeatureExtractor,
    MicroFeatureExtractor,
    PreprocessConfig,
    WHSPreprocessor,
)

from ._streamlit_compat import cache_data


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


def run_pipeline_core(
    subject_id: str,
    data: np.ndarray,
    start_datetime: datetime,
    config_dict: dict,
) -> dict:
    """Framework-agnostic pipeline runner.

    Both ``app.py`` (via the Streamlit-cached ``run_pipeline`` below) and
    ``app_dash.py`` (via its own ``lru_cache`` wrapper) call into this.
    """
    cfg = PreprocessConfig(**config_dict)
    pre = WHSPreprocessor(cfg)
    result = pre.process(data, start_datetime=start_datetime, subject_id=subject_id)
    if result["valid"]:
        result["macro_features"] = MacroFeatureExtractor(cfg).extract_all(result["macro"])
        result["micro_features"] = MicroFeatureExtractor(cfg).extract_all(result["micro"])
    return result


@cache_data(show_spinner="Running preprocessor...")
def run_pipeline(
    subject_id: str,
    _data: np.ndarray,
    start_datetime: datetime,
    config_dict: dict,
) -> dict:
    """Streamlit-cached entry point. The underscore on ``_data`` tells Streamlit's
    hasher to skip the large numpy array — the cache key is
    ``(subject_id, start_datetime, config_dict)``.
    """
    return run_pipeline_core(subject_id, _data, start_datetime, config_dict)
