"""
softstump_shape.py
==================

Adapter that wraps WarmStartGAM's per-feature soft-stumps behind the ShapeFunction
interface, plus `AdditiveGAM` -- a single object holding a heterogeneous mix of
soft-stump and constrained-spline shapes under one bias.

This is the "both bases in one model" answer: keep soft-stumps for the features you
don't want to constrain, swap the S/U features to a constrained SplineShape, and sum
everything into one log-risk vector that your Breslow head and Cox / composite loss
consume UNCHANGED.

Also provides an automated shape-function audit: `AdditiveGAM.audit_registry` (and the
registry=None path of `from_warmstart_gam`) classifies each feature's XGBoost shape
into a suggested ShapeClass. Treat the output as a *first guess from one fold* -- review
against domain priors and fold stability before committing any feature to CONVEX.

Requires spline_nam.py (ShapeFunction, ShapeClass, SplineShape) on the path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spline_nam import ShapeFunction, ShapeClass, SplineShape


def _soft_stump_eval(thresholds: torch.Tensor, weights: torch.Tensor,
                     temperature: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """f(x) = sum_t w_t * sigmoid(T_t * (x - c_t)); x:(N,) -> (N,). Same algebra as
    WarmStartGAM.forward, restricted to one feature's stumps."""
    s = torch.sigmoid(temperature * (x.unsqueeze(-1) - thresholds))   # (N, S)
    return (weights * s).sum(dim=-1)                                  # (N,)


def suggest_shape_class(g: np.ndarray, *, mono_thresh: float = 0.85,
                        arm_frac: float = 0.15, flat_tol: float = 1e-3,
                        edge_margin: float = 0.05):
    """Classify a shape function sampled on a (data-supported) grid into a ShapeClass.

    Robust, conservative heuristic -- proposes a *constraint* only when the evidence is
    clean, otherwise FREE:
      * amplitude < flat_tol                              -> FREE (negligible / unused)
      * net displacement / total variation >= mono_thresh -> MONO_INC / MONO_DEC
        (near-monotone; a small tail wiggle does NOT flip it)
      * single dominant interior extremum, both arms >= arm_frac of amplitude
                                                          -> CONVEX (U) / CONCAVE (cap)
      * anything else (multi-modal / ambiguous)           -> FREE

    Returns (ShapeClass, diagnostics dict). Validated on canonical shapes incl. the
    phantom-tail case (monotone + tiny uptick -> stays monotone, not CONVEX).
    """
    g = np.asarray(g, dtype=float)
    n = len(g)
    amp = float(g.max() - g.min())
    diag = {"amp": amp}
    if amp < flat_tol:
        diag["reason"] = "flat"
        return ShapeClass.FREE, diag

    tv = float(np.abs(np.diff(g)).sum())
    net = float(abs(g[-1] - g[0]))
    mono_ratio = net / (tv + 1e-12)
    diag["mono_ratio"] = mono_ratio
    if mono_ratio >= mono_thresh:
        diag["reason"] = "monotone"
        return (ShapeClass.MONO_INC if g[-1] >= g[0] else ShapeClass.MONO_DEC), diag

    i_min, i_max = int(np.argmin(g)), int(np.argmax(g))
    m = max(1, int(edge_margin * n))
    interior = lambda i: m <= i <= n - 1 - m
    u_arms = min(g[0] - g[i_min], g[-1] - g[i_min])     # rise from the (interior) min
    c_arms = min(g[i_max] - g[0], g[i_max] - g[-1])     # fall from the (interior) max
    is_U = interior(i_min) and u_arms >= arm_frac * amp
    is_C = interior(i_max) and c_arms >= arm_frac * amp
    if is_U and not is_C:
        diag["reason"] = "U: single interior minimum"
        return ShapeClass.CONVEX, diag
    if is_C and not is_U:
        diag["reason"] = "cap: single interior maximum"
        return ShapeClass.CONCAVE, diag
    diag["reason"] = "ambiguous / multi-modal"
    return ShapeClass.FREE, diag


class SoftStumpShape(ShapeFunction):
    """Per-feature soft-stump shape -- exactly WarmStartGAM restricted to one feature,
    then centered for additive identifiability. Trainability mirrors WarmStartGAM:
    thresholds frozen, weights optional, temperature trainable.
    """

    def __init__(self, thresholds: torch.Tensor, weights: torch.Tensor,
                 temperature: torch.Tensor, train_weights: bool = False,
                 train_temp: bool = True, train_thresh: bool = False):
        super().__init__()
        self.thresholds = nn.Parameter(thresholds.detach().clone(), requires_grad=train_thresh)
        self.weights = nn.Parameter(weights.detach().clone(), requires_grad=train_weights)
        self.temperature = nn.Parameter(temperature.detach().clone(), requires_grad=train_temp)
        self.register_buffer("center", torch.zeros(()))

    def forward(self, x_col: torch.Tensor) -> torch.Tensor:
        return _soft_stump_eval(self.thresholds, self.weights, self.temperature, x_col) - self.center

    @torch.no_grad()
    def warm_start(self, x_grid: np.ndarray, y_ref=None) -> None:
        g = torch.as_tensor(np.asarray(x_grid), dtype=self.thresholds.dtype,
                            device=self.thresholds.device)
        self.center.copy_(_soft_stump_eval(self.thresholds, self.weights, self.temperature, g).mean())


class AdditiveGAM(nn.Module):
    """
    log-risk(x) = bias + sum_j f_j(x_j), each f_j ANY ShapeFunction (SoftStumpShape or
    SplineShape). One object, one bias, heterogeneous bases. Drop-in for WarmStartGAM:
    forward returns the (N,) log-risk vector for the Breslow head + Cox / composite loss.
    """

    def __init__(self, shapes: dict, bias: float, train_bias: bool = False):
        super().__init__()
        self.feat_idx = sorted(shapes)
        self.shapes = nn.ModuleList(shapes[j] for j in self.feat_idx)
        self.bias = nn.Parameter(torch.tensor(float(bias)), requires_grad=train_bias)
        self.audit_ = None              # set when registry was auto-suggested
        self.suggested_registry_ = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r = self.bias.expand(X.shape[0]).clone()
        for j, shape in zip(self.feat_idx, self.shapes):
            r = r + shape(X[:, j])
        return r                                       # (N,) -> Breslow head

    def penalty(self) -> torch.Tensor:
        """Roughness penalty over FREE spline features only; add to the loss with a weight."""
        terms = [s.penalty() for s in self.shapes
                 if isinstance(s, SplineShape) and s.shape == ShapeClass.FREE]
        return torch.stack(terms).sum() if terms else self.bias.new_zeros(())

    # -- automated shape-function audit ------------------------------------
    @staticmethod
    @torch.no_grad()
    def audit_registry(wsgam, X_train: np.ndarray, n_grid: int = 200,
                       audit_quantile_range: tuple = (0.025, 0.975),
                       include_free: bool = False, **classify_kw):
        """Classify every feature XGBoost split on into a suggested ShapeClass.

        Classification uses each feature's XGBoost soft-stump shape sampled on the
        central [q_lo, q_hi] quantile band (default 2.5-97.5%), so sparse-tail phantoms
        cannot drive the call. Returns (registry, audit_records):
          * registry : {feat -> ShapeClass} -- by default only features given a
            confident constraint (MONO_*/CONVEX/CONCAVE); FREE features are omitted so
            they stay as soft-stumps. Pass include_free=True to register them as well.
          * audit_records : list of dicts (one per audited feature, incl. FREE) with
            the suggested class and diagnostics, for review (`pd.DataFrame(records)`).
        """
        feat = wsgam.feat_idx.detach().cpu()
        thr = wsgam.thresholds.detach().cpu()
        w = wsgam.weights.detach().cpu()
        temp = wsgam.temperature.detach().cpu()
        q_lo, q_hi = audit_quantile_range

        registry, records = {}, []
        for j in sorted(feat.unique().tolist()):
            mask = (feat == j)
            grid = np.quantile(X_train[:, j], np.linspace(q_lo, q_hi, n_grid))
            g = _soft_stump_eval(thr[mask], w[mask], temp[mask],
                                 torch.as_tensor(grid, dtype=thr.dtype)).numpy()
            cls_j, diag = suggest_shape_class(g, **classify_kw)
            records.append({"feat": j, "suggested": cls_j.value,
                            "n_stumps": int(mask.sum()), **diag})
            if cls_j != ShapeClass.FREE or include_free:
                registry[j] = cls_j
        return registry, records

    @classmethod
    @torch.no_grad()
    def from_warmstart_gam(cls, wsgam, shape_registry=None, X_train: np.ndarray = None,
                           n_grid: int = 200, train_weights: bool = False,
                           train_temp: bool = True, audit_quantile_range: tuple = (0.025, 0.975),
                           include_free: bool = False, **spline_kw) -> "AdditiveGAM":
        """Build one mixed object from an *already-initialised* WarmStartGAM.

        If `shape_registry is None`, the registry is auto-suggested via `audit_registry`
        (the audit is stashed on the returned model as `.audit_` and `.suggested_registry_`
        for review). Features in the registry get a constrained SplineShape warm-started
        from that feature's XGBoost shape; all other features XGBoost split on keep their
        soft-stumps. WarmStartGAM's temperature init is reused verbatim.
        """
        if X_train is None:
            raise ValueError("X_train is required (used for knot placement and the audit).")

        auto = shape_registry is None
        if auto:
            shape_registry, audit_records = cls.audit_registry(
                wsgam, X_train, n_grid=n_grid, audit_quantile_range=audit_quantile_range,
                include_free=include_free)
        else:
            audit_records = None

        feat = wsgam.feat_idx.detach().cpu()
        thr = wsgam.thresholds.detach().cpu()
        w = wsgam.weights.detach().cpu()
        temp = wsgam.temperature.detach().cpu()
        orig_bias = float(wsgam.bias.detach().cpu())

        shapes: dict = {}
        all_feats = sorted(set(feat.unique().tolist()) | set(shape_registry))
        for j in all_feats:
            mask = (feat == j)
            grid = np.quantile(X_train[:, j], np.linspace(0.0, 1.0, n_grid))
            cls_j = shape_registry.get(j)
            if cls_j is None:                                    # keep soft-stumps
                shp = SoftStumpShape(thr[mask], w[mask], temp[mask],
                                     train_weights=train_weights, train_temp=train_temp)
                shp.warm_start(grid)
            else:                                                # replace with spline
                shp = SplineShape(cls_j, X_train[:, j], **spline_kw)
                if mask.any():
                    g = torch.as_tensor(grid, dtype=thr.dtype)
                    ref = _soft_stump_eval(thr[mask], w[mask], temp[mask], g).numpy()
                else:
                    ref = np.zeros_like(grid)                    # feature unused by XGBoost -> flat
                shp.warm_start(grid, ref)
            shapes[j] = shp

        total_center = sum(float(s.center) for s in shapes.values())
        model = cls(shapes, bias=orig_bias + total_center, train_bias=False)
        if auto:
            model.audit_ = audit_records
            model.suggested_registry_ = dict(shape_registry)
        return model


# ---------------------------------------------------------------------------
# Usage
#
# A) one-shot, auto-suggested registry:
#   model = AdditiveGAM.from_warmstart_gam(wsgam, shape_registry=None, X_train=X_train)
#   import pandas as pd; print(pd.DataFrame(model.audit_))     # review the suggestions
#
# B) audit -> review/edit -> build (recommended before committing CONVEX):
#   registry, audit = AdditiveGAM.audit_registry(wsgam, X_train)
#   # ...inspect `audit`, override any feature against your domain prior / fold stability...
#   registry[FEAT["sleep_hrs"]] = ShapeClass.CONVEX
#   model = AdditiveGAM.from_warmstart_gam(wsgam, registry, X_train)
#
#   r = model(X_train)        # same (N,) log-risk -> Breslow head + composite loss
# ---------------------------------------------------------------------------
