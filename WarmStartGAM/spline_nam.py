"""
spline_nam.py
=============

Constrained, warm-started spline shape functions for the WarmStartGAM additive model.

WarmStartGAM already factorises the cause-specific risk score as

    r(x) = bias + sum_j  f_j(x_j)

where each f_j is currently the sum of the soft-stumps that split on feature j.
This module supplies an alternative f_j: a constrained cubic B-spline subnet whose
shape class (monotone / convex-unimodal / free) is fixed in advance from a domain
prior, and whose coefficients are warm-started by least-squares fitting to the
existing XGBoost / soft-stump shape.

Drop-in contract
----------------
Anything implementing `ShapeFunction` maps ONE feature column (N,) to a centered
log-risk contribution (N,). WarmStartGAM only ever sees r(x), so the Breslow head,
the Cox partial-likelihood loss and WarmStartCompositeLoss are untouched -- the
spline subnet is a parameterisation swap behind f_j, not a new training pipeline.

Constraints are imposed by *reparameterisation* (softplus-of-raw), so training
stays unconstrained first-order: no projected gradient, no QP in the loop. The
constraint enters only the one-off warm-start projection and the forward
reconstruction of the coefficients.

Deps: numpy, torch, scipy>=1.8 (BSpline.design_matrix), scikit-learn (isotonic).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from scipy.interpolate import BSpline
from sklearn.isotonic import isotonic_regression


class ShapeClass(str, Enum):
    MONO_INC = "mono_inc"
    MONO_DEC = "mono_dec"
    CONVEX   = "convex"     # U-shaped: single interior minimum
    CONCAVE  = "concave"    # single interior maximum
    FREE     = "free"


def _softplus_inv(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse of softplus, for recovering raw params from a positive target."""
    return torch.log(torch.expm1(y.clamp_min(eps)))


class ShapeFunction(nn.Module):
    """f_j: feature column (N,) -> centered contribution (N,) in log-risk space.

    Implement this and WarmStartGAM/SplineNAM can hold a mix of bases per feature
    (e.g. keep your SoftStumpShape for some features, SplineShape for others).
    """

    def forward(self, x_col: torch.Tensor) -> torch.Tensor:        # noqa: D401
        raise NotImplementedError

    @torch.no_grad()
    def warm_start(self, x_grid: np.ndarray, y_ref: np.ndarray) -> None:
        """Initialise params to reproduce a reference shape sampled on x_grid."""
        raise NotImplementedError

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)


class SplineShape(ShapeFunction):
    """
    Constrained cubic B-spline shape function.

        f(x) = B(x) @ theta - center

    `theta` is reconstructed from raw params so the shape constraint holds by
    construction (optimiser stays unconstrained):

        FREE     : theta = raw                                  (+ 2nd-diff penalty)
        MONO_INC : theta = level + cumsum(softplus(raw))        (non-decreasing coeffs)
        MONO_DEC : -MONO_INC
        CONVEX   : 1st diffs non-decreasing
                   fd = slope + cumsum(softplus(curv)); theta = level + cumsum(fd)
        CONCAVE  : -CONVEX

    For a B-spline, monotone coefficients => monotone f, and convex coefficient
    sequence => convex f (exact for uniform knots; near-exact for the quantile
    knots used here -- tighten n_knots if a constrained fit looks biased).
    """

    def __init__(self, shape: ShapeClass, x_train: np.ndarray,
                 n_knots: int = 10, degree: int = 3, ridge: float = 1e-4):
        super().__init__()
        self.shape = ShapeClass(shape)
        self.degree = int(degree)
        self.ridge = float(ridge)

        # interior knots at feature quantiles; clamped (repeated) boundary knots
        interior = np.quantile(np.asarray(x_train, dtype=np.float64),
                               np.linspace(0.0, 1.0, n_knots))
        lo, hi = interior[0], interior[-1]
        t = np.concatenate([[lo] * degree, interior, [hi] * degree])
        self.register_buffer("knots", torch.tensor(t, dtype=torch.float32))
        self.n_basis = len(t) - degree - 1
        self.register_buffer("center", torch.zeros(()))   # set in warm_start

        if self.shape == ShapeClass.FREE:
            self.raw = nn.Parameter(torch.zeros(self.n_basis))
        elif self.shape in (ShapeClass.MONO_INC, ShapeClass.MONO_DEC):
            self.level = nn.Parameter(torch.zeros(()))
            self.raw = nn.Parameter(torch.zeros(self.n_basis - 1))   # softplus -> steps
        else:  # CONVEX / CONCAVE
            self.level = nn.Parameter(torch.zeros(()))
            self.slope = nn.Parameter(torch.zeros(()))
            self.raw = nn.Parameter(torch.zeros(self.n_basis - 2))   # softplus -> curvature

    # -- coefficient reconstruction (constraint baked in) -------------------
    def _theta(self) -> torch.Tensor:
        s, dev = self.shape, self.raw.device
        if s == ShapeClass.FREE:
            return self.raw
        if s in (ShapeClass.MONO_INC, ShapeClass.MONO_DEC):
            steps = F.softplus(self.raw)
            theta = self.level + torch.cat([torch.zeros(1, device=dev), torch.cumsum(steps, 0)])
            return theta if s == ShapeClass.MONO_INC else -theta
        # CONVEX / CONCAVE
        curv = F.softplus(self.raw)                                   # >= 0  (2nd diffs)
        fd = self.slope + torch.cat([torch.zeros(1, device=dev), torch.cumsum(curv, 0)])
        theta = self.level + torch.cat([torch.zeros(1, device=dev), torch.cumsum(fd, 0)])
        return theta if s == ShapeClass.CONVEX else -theta

    # -- fixed B-spline design matrix (knots fixed -> no grad through x) -----
    def _basis_np(self, x: np.ndarray) -> np.ndarray:
        t = self.knots.detach().cpu().numpy().astype(np.float64)
        lo, hi = t[0], t[-1]
        x = np.clip(np.asarray(x, dtype=np.float64), lo, np.nextafter(hi, lo))
        return BSpline.design_matrix(x, t, self.degree, extrapolate=False).toarray()

    def forward(self, x_col: torch.Tensor) -> torch.Tensor:
        # Full-batch: recomputing the basis per step is one small matmul. For GPU
        # minibatch training, precompute B once and index it instead.
        B = torch.as_tensor(self._basis_np(x_col.detach().cpu().numpy()),
                            dtype=x_col.dtype, device=x_col.device)
        f = B @ self._theta()
        return f - self.center

    def penalty(self) -> torch.Tensor:
        """Second-difference roughness penalty -- meaningful only for FREE features."""
        th = self._theta()
        d2 = th[2:] - 2.0 * th[1:-1] + th[:-2]
        return (d2 ** 2).sum()

    @torch.no_grad()
    def warm_start(self, x_grid: np.ndarray, y_ref: np.ndarray) -> None:
        y_ref = np.asarray(y_ref, dtype=np.float64)
        B = self._basis_np(x_grid)
        # 1) unconstrained ridge LS fit of coefficients to the reference shape
        A = B.T @ B + self.ridge * np.eye(B.shape[1])
        theta = np.linalg.solve(A, B.T @ y_ref)                       # (n_basis,)

        # 2) project onto the constraint set, then invert the reparameterisation
        s = self.shape
        if s == ShapeClass.FREE:
            self.raw.copy_(torch.as_tensor(theta, dtype=self.raw.dtype))
        elif s in (ShapeClass.MONO_INC, ShapeClass.MONO_DEC):
            th = -theta if s == ShapeClass.MONO_DEC else theta
            th = isotonic_regression(th, increasing=True)             # PAVA projection
            steps = np.clip(np.diff(th), 1e-6, None)
            self.level.copy_(torch.as_tensor(th[0], dtype=self.level.dtype))
            self.raw.copy_(_softplus_inv(torch.as_tensor(steps, dtype=self.raw.dtype)))
        else:  # CONVEX / CONCAVE
            th = -theta if s == ShapeClass.CONCAVE else theta
            fd = np.diff(th)                                           # first differences
            curv = np.clip(np.diff(fd), 1e-6, None)                   # clamp 2nd diffs >= 0
            self.level.copy_(torch.as_tensor(th[0], dtype=self.level.dtype))
            self.slope.copy_(torch.as_tensor(fd[0], dtype=self.slope.dtype))
            self.raw.copy_(_softplus_inv(torch.as_tensor(curv, dtype=self.raw.dtype)))

        # 3) centering constant for additive identifiability (mean over the grid)
        f0 = self._basis_np(x_grid) @ self._theta().detach().cpu().numpy()
        self.center.copy_(torch.as_tensor(f0.mean(), dtype=self.center.dtype))


class SplineNAM(nn.Module):
    """
    r(x) = bias + sum_j f_j(x_j), each f_j a ShapeFunction (default SplineShape).

    Mirrors WarmStartGAM's additive contract exactly, so the existing Breslow head,
    Cox partial-likelihood loss and WarmStartCompositeLoss attach UNCHANGED -- they
    only consume r(x). Mix bases freely by putting any ShapeFunction in `self.shapes`
    (e.g. a SoftStumpShape adapter around your current stumps for the FREE features).
    """

    def __init__(self, shape_registry: dict[int, ShapeClass], X_train: np.ndarray,
                 bias_init: float = 0.0, **spline_kw):
        super().__init__()
        self.feat_idx = sorted(shape_registry)
        self.shapes = nn.ModuleList(
            SplineShape(shape_registry[j], X_train[:, j], **spline_kw)
            for j in self.feat_idx
        )
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r = self.bias.expand(X.shape[0]).clone()
        for j, shape in zip(self.feat_idx, self.shapes):
            r = r + shape(X[:, j])
        return r                                  # (N,) log-risk score -> Breslow head

    @torch.no_grad()
    def warm_start_from_xgb(self, shape_fns: dict[int, "callable"],
                            grids: dict[int, np.ndarray]) -> None:
        """shape_fns[j](grid) -> the XGBoost / soft-stump contribution for feature j
        on `grids[j]` (your existing per-feature shape sampler / viz code)."""
        for j, shape in zip(self.feat_idx, self.shapes):
            shape.warm_start(grids[j], shape_fns[j](grids[j]))

    def penalty(self) -> torch.Tensor:
        """Sum roughness penalties over FREE features only; add to the loss with a weight."""
        return sum((s.penalty() for s in self.shapes if s.shape == ShapeClass.FREE),
                   start=torch.zeros((), device=self.bias.device))


# ---------------------------------------------------------------------------
# Sketch of use (one cause; loop causes exactly as you do now):
#
#   registry = {  # from your shape-function audit + domain prior
#       FEAT["RMS_stds"]:   ShapeClass.MONO_DEC,   # S-shaped, decreasing
#       FEAT["sleep_hrs"]:  ShapeClass.CONVEX,     # U-shaped: interior optimum
#       FEAT["age"]:        ShapeClass.MONO_INC,
#       FEAT["lab_x"]:      ShapeClass.FREE,
#       ...
#   }
#   model = SplineNAM(registry, X_train, bias_init=float(xgb_bias))
#   model.warm_start_from_xgb(xgb_shape_fns, quantile_grids)   # start at XGBoost shapes
#
#   # identical to your current loop: feed r=model(X) to the Breslow head + Cox loss
#   r = model(X_train)
#   loss = composite_loss(r, durations, events, w_current=..., ) \
#          + lam_smooth * model.penalty()
#   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# ---------------------------------------------------------------------------
