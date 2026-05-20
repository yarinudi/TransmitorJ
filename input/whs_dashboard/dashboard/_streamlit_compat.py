"""Streamlit decorator shims so the dashboard package imports cleanly without Streamlit.

When Streamlit is installed, ``cache_data`` / ``cache_resource`` forward to
``streamlit.cache_data`` / ``streamlit.cache_resource``. Otherwise they
no-op — the wrapped function runs uncached, and any downstream framework
(e.g. Dash) is expected to add its own caching at the call site.

The ``WHS_DASHBOARD_NO_STREAMLIT_CACHE`` env var forces the no-op path even
when Streamlit *is* importable — this is how ``app_dash.py`` opts out so it
doesn't trigger Streamlit's "No runtime found" warnings on every cached call.
"""
from __future__ import annotations

import os

_OPT_OUT = bool(os.environ.get("WHS_DASHBOARD_NO_STREAMLIT_CACHE"))

try:
    import streamlit as _st
    HAS_STREAMLIT = True and not _OPT_OUT
except ImportError:
    _st = None
    HAS_STREAMLIT = False


def cache_data(**decorator_kwargs):
    """Drop-in for ``@st.cache_data``; pass-through when Streamlit is missing or opted-out."""
    if HAS_STREAMLIT:
        return _st.cache_data(**decorator_kwargs)
    def _passthrough(fn):
        return fn
    return _passthrough


def cache_resource(**decorator_kwargs):
    """Drop-in for ``@st.cache_resource``; pass-through when Streamlit is missing or opted-out."""
    if HAS_STREAMLIT:
        return _st.cache_resource(**decorator_kwargs)
    def _passthrough(fn):
        return fn
    return _passthrough
