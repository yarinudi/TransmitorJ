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
        # Already warm (initialised from XGBoost); just fix the identifiability center.
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

    @classmethod
    @torch.no_grad()
    def from_warmstart_gam(cls, wsgam, shape_registry: dict, X_train: np.ndarray,
                           n_grid: int = 200, train_weights: bool = False,
                           train_temp: bool = True, **spline_kw) -> "AdditiveGAM":
        """Build one mixed object from an *already-initialised* WarmStartGAM.

        Features listed in `shape_registry` (feat_idx -> ShapeClass) are replaced by a
        constrained SplineShape warm-started from that feature's XGBoost soft-stump
        shape; every other feature XGBoost split on keeps its soft-stumps. WarmStartGAM's
        own temperature init is reused verbatim -- we only slice its tensors.
        """
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
                shp.warm_start(grid)                             # sets center
            else:                                                # replace with spline
                shp = SplineShape(cls_j, X_train[:, j], **spline_kw)
                if mask.any():
                    g = torch.as_tensor(grid, dtype=thr.dtype)
                    ref = _soft_stump_eval(thr[mask], w[mask], temp[mask], g).numpy()
                else:
                    ref = np.zeros_like(grid)                    # feature unused by XGBoost -> flat
                shp.warm_start(grid, ref)                        # fits coeffs + sets center
            shapes[j] = shp

        # one bias preserving the warm-start level: forward subtracts each shape's center,
        # so add the centers back here  =>  r(x) reproduces the original stump model at init.
        total_center = sum(float(s.center) for s in shapes.values())
        return cls(shapes, bias=orig_bias + total_center, train_bias=False)


# ---------------------------------------------------------------------------
# Usage -- one object, mixed bases, built from your existing WarmStartGAM:
#
#   wsgam = WarmStartGAM(stumps, frozen=False)        # your model, already warm
#   registry = {                                       # from the shape-function audit
#       FEAT["RMS_stds"]:  ShapeClass.MONO_DEC,        # S-shaped, decreasing
#       FEAT["sleep_hrs"]: ShapeClass.CONVEX,          # U-shaped: interior optimum
#   }
#   model = AdditiveGAM.from_warmstart_gam(wsgam, registry, X_train,
#                                          train_weights=False, train_temp=True)
#   r = model(X_train)                                 # same (N,) log-risk as before
#   loss = composite_loss(r, durations, events, w_current=...) + lam * model.penalty()
# ---------------------------------------------------------------------------
