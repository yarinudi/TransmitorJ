"""WarmStartGAM vs XGBoost depth-1: shape-function and stump-level comparison.

Three artifacts, all dropped under OUT_DIR:
  1. shape_functions/<feature>.png   - per-feature overlay (XGB step + GAM smooth + density)
  2. divergence_summary.png          - per-feature density-weighted L1 divergence + XGB importance
  3. threshold_migration.png         - per-stump scatter of original vs trained threshold (quantile rank)

Run directly to demo on the POC pipeline, or `from viz_xgb_vs_torch import make_report`
and call with your own (booster, model, X_train, feat_cols).
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import KernelDensity

OUT_DIR_DEFAULT = "plots/warmstart_vs_xgb"
N_GRID = 300

# steelblue density colormap (matches xgbGAMView convention: white -> #467CB4)
_SB_CDICT = {
    "red":   [(0.0, 1.0, 1.0), (1.0, 0.275, 0.275)],
    "green": [(0.0, 1.0, 1.0), (1.0, 0.510, 0.510)],
    "blue":  [(0.0, 1.0, 1.0), (1.0, 0.706, 0.706)],
}
STEELBLUE_MAP = LinearSegmentedColormap("steelblue", _SB_CDICT, 256)


# ----------------------------- Shape extraction -----------------------------
def get_xgb_stumps(booster, feat_cols):
    """One dict per depth-1 tree: {feat, feat_idx, c, left, right}."""
    tdf = booster.trees_to_dataframe()
    stumps = []
    for _, sub in tdf.groupby("Tree"):
        root = sub[sub["Node"] == 0].iloc[0]
        if root["Feature"] == "Leaf":
            continue
        yes_row = sub[sub["ID"] == root["Yes"]].iloc[0]
        no_row = sub[sub["ID"] == root["No"]].iloc[0]
        stumps.append({
            "feat": root["Feature"],
            "feat_idx": feat_cols.index(root["Feature"]),
            "c": float(root["Split"]),
            "left": float(yes_row["Gain"]),
            "right": float(no_row["Gain"]),
        })
    return stumps


def xgb_shape(stumps_xgb, feat_name, x_grid):
    """f^XGB_j(x) = Σ_{t: feat_t=j} (left_t if x<c_t else right_t)."""
    y = np.zeros_like(x_grid, dtype=float)
    for s in stumps_xgb:
        if s["feat"] == feat_name:
            y += np.where(x_grid < s["c"], s["left"], s["right"])
    return y


def torch_shape(model, feat_idx, x_grid):
    """f^GAM_j(x) = Σ_{t: feat_idx_t=j} w_t · sigmoid(T · (x − c_t)).

    Handles scalar OR per-feature temperature (length-F vector indexed by feat_idx).
    """
    fi = model.feat_idx.cpu().numpy()
    mask = fi == feat_idx
    if not mask.any():
        return np.zeros_like(x_grid, dtype=float)
    c = model.thresholds.detach().cpu().numpy()[mask]
    w = model.weights.detach().cpu().numpy()[mask]
    T_raw = model.temperature.detach().cpu().numpy()
    if T_raw.ndim == 0:
        T_used = float(T_raw)
    elif T_raw.ndim == 1 and T_raw.size > feat_idx:
        T_used = float(T_raw[feat_idx])    # per-feature temperature variant
    else:
        T_used = float(T_raw[mask][0])     # fallback: per-stump
    x = x_grid[:, None]                     # (G, 1)
    return (w * 1.0 / (1.0 + np.exp(-T_used * (x - c)))).sum(axis=1)


# ----------------------------- Plot 1: per-feature overlay -----------------------------
def plot_shape_overlay(feat_name, x_grid, xgb_y, trn_y, x_train_j,
                       xgb_thresh_j, trn_thresh_j, save_path):
    # Center at training-data median (robust, always in-distribution)
    ref = float(np.median(x_train_j))
    xgb_yc = xgb_y - np.interp(ref, x_grid, xgb_y)
    trn_yc = trn_y - np.interp(ref, x_grid, trn_y)

    # KDE density band (Silverman bandwidth, guarded)
    bw = max(1e-6, 1.06 * np.std(x_train_j) * max(1, len(x_train_j)) ** (-0.2))
    kde = KernelDensity(bandwidth=bw).fit(x_train_j.reshape(-1, 1))
    dens = np.exp(kde.score_samples(x_grid.reshape(-1, 1)))

    ymin = min(xgb_yc.min(), trn_yc.min())
    ymax = max(xgb_yc.max(), trn_yc.max())
    pad = 0.1 * (ymax - ymin + 1e-9)
    ymin -= pad; ymax += pad

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(dens.reshape(1, -1), aspect="auto", cmap=STEELBLUE_MAP,
              extent=[x_grid.min(), x_grid.max(), ymin, ymax],
              alpha=0.55, zorder=0)
    ax.plot(x_grid, xgb_yc, color="black", linestyle="--", linewidth=1.4,
            label="XGBoost depth-1", zorder=2)
    ax.plot(x_grid, trn_yc, color="C3", linewidth=2.0,
            label="WarmStartGAM", zorder=3)
    # threshold ticks: XGB at bottom, trained at top
    for c in xgb_thresh_j:
        ax.axvline(c, ymin=0.00, ymax=0.04, color="black", linewidth=0.8)
    for c in trn_thresh_j:
        ax.axvline(c, ymin=0.96, ymax=1.00, color="C3", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.5, zorder=1)
    ax.set_xlim(x_grid.min(), x_grid.max()); ax.set_ylim(ymin, ymax)
    ax.set_xlabel(feat_name); ax.set_ylabel("shape contribution (centered)")
    ax.set_title(f"{feat_name}  ({len(xgb_thresh_j)} stumps)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


# ----------------------------- Plot 2: divergence summary -----------------------------
def plot_divergence_summary(divergences, importances, save_path):
    feats = list(divergences.keys())
    div = np.array([divergences[f] for f in feats])
    imp = np.array([importances[f] for f in feats])
    order = np.argsort(div)[::-1]
    feats_s = [feats[i] for i in order]
    div_s, imp_s = div[order], imp[order]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(8, len(feats) * 0.32), 6.5), sharex=True
    )
    ax1.bar(range(len(feats)), div_s, color="C3", edgecolor="black", linewidth=0.4)
    ax1.set_ylabel(r"$\int |f_{\mathrm{GAM}}-f_{\mathrm{XGB}}|\,p(x)\,dx$")
    ax1.set_title("Shape-function divergence (sorted desc)")
    ax2.bar(range(len(feats)), imp_s, color="steelblue", edgecolor="black", linewidth=0.4)
    ax2.set_ylabel("XGB importance\n(range of shape fn)")
    ax2.set_xticks(range(len(feats)))
    ax2.set_xticklabels(feats_s, rotation=45, ha="right", fontsize=9)
    ax2.set_xlabel("feature")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


# ----------------------------- Plot 3: threshold migration -----------------------------
def plot_threshold_migration(stumps_xgb, model, X_train, save_path):
    """Quantile-rank thresholds within each feature so scales become comparable."""
    trained = model.thresholds.detach().cpu().numpy()
    weights = model.weights.detach().cpu().numpy()
    xs, ys, sizes, feat_ids = [], [], [], []
    for i, s in enumerate(stumps_xgb):
        col = X_train[:, s["feat_idx"]]
        xs.append((col < s["c"]).mean())            # quantile rank of XGB threshold
        ys.append((col < trained[i]).mean())        # quantile rank of trained threshold
        sizes.append(abs(weights[i]))
        feat_ids.append(s["feat_idx"])
    xs = np.array(xs); ys = np.array(ys)
    sizes = np.array(sizes); feat_ids = np.array(feat_ids)
    s_pt = 20 + 200 * (sizes / (sizes.max() + 1e-9))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="no movement", zorder=1)
    ax.scatter(xs, ys, c=feat_ids, cmap="tab20", s=s_pt, alpha=0.75,
               edgecolor="black", linewidth=0.4, zorder=2)
    ax.set_xlabel("XGB threshold  (quantile rank in train)")
    ax.set_ylabel("Trained threshold  (quantile rank in train)")
    ax.set_title(f"Stump threshold migration  ({len(xs)} stumps)\n"
                 f"size ∝ |w_t|,  color = feature")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


# ----------------------------- Orchestration -----------------------------
def _density_weighted_l1(col, x_grid, xgb_y, trn_y, n_bins=50):
    """∫ |f_GAM − f_XGB| · p(x) dx, with both shape fns mean-zero over p."""
    hist, edges = np.histogram(col, bins=n_bins, density=True)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    a = np.interp(ctr, x_grid, xgb_y)
    b = np.interp(ctr, x_grid, trn_y)
    norm = hist.sum() + 1e-9
    a -= (a * hist).sum() / norm
    b -= (b * hist).sum() / norm
    widths = np.diff(edges)
    return float((np.abs(a - b) * hist * widths).sum())


def make_report(booster, model, X_train, feat_cols,
                out_dir=OUT_DIR_DEFAULT, n_grid=N_GRID):
    """Produce all three figure types under out_dir/. Returns (divergences, importances)."""
    shapes_dir = os.path.join(out_dir, "shape_functions")
    os.makedirs(shapes_dir, exist_ok=True)

    stumps_xgb = get_xgb_stumps(booster, feat_cols)
    trained_thresh = model.thresholds.detach().cpu().numpy()
    feats_used = sorted({s["feat"] for s in stumps_xgb})

    divergences, importances = {}, {}
    for feat_name in feats_used:
        j = feat_cols.index(feat_name)
        col = X_train[:, j]
        lo, hi = np.percentile(col, [1, 99])
        pad = 0.05 * (hi - lo + 1e-9)
        x_grid = np.linspace(lo - pad, hi + pad, n_grid)

        xgb_y = xgb_shape(stumps_xgb, feat_name, x_grid)
        trn_y = torch_shape(model, j, x_grid)

        divergences[feat_name] = _density_weighted_l1(col, x_grid, xgb_y, trn_y)
        importances[feat_name] = float(xgb_y.max() - xgb_y.min())

        xgb_thresh_j = [s["c"] for s in stumps_xgb if s["feat"] == feat_name]
        trn_thresh_j = [trained_thresh[i] for i, s in enumerate(stumps_xgb)
                        if s["feat"] == feat_name]
        plot_shape_overlay(
            feat_name, x_grid, xgb_y, trn_y, col,
            xgb_thresh_j, trn_thresh_j,
            os.path.join(shapes_dir, f"{feat_name}.png"),
        )

    plot_divergence_summary(
        divergences, importances, os.path.join(out_dir, "divergence_summary.png")
    )
    plot_threshold_migration(
        stumps_xgb, model, X_train,
        os.path.join(out_dir, "threshold_migration.png")
    )
    print(f"  shape plots: {shapes_dir}/  ({len(feats_used)} features)")
    print(f"  summary    : {out_dir}/divergence_summary.png")
    print(f"  thresholds : {out_dir}/threshold_migration.png")
    return divergences, importances


# ----------------------------- Demo entrypoint -----------------------------
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from xgb_gam_to_torch_poc import (
        load_data, split, train_xgb, extract_stumps,
        WarmStartGAM, fit_pytorch, SEED, TEMP_INIT,
    )
    torch.manual_seed(SEED); np.random.seed(SEED)
    X, y, feat_cols = load_data()
    X_tr, X_te, y_tr, y_te = split(X, y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr["event"], random_state=SEED
    )
    booster = train_xgb(X_fit, y_fit)
    stumps = extract_stumps(booster, feat_cols)
    model = WarmStartGAM(stumps, temp_init=TEMP_INIT)
    fit_pytorch(model, X_fit, y_fit, X_val, y_val)
    make_report(booster, model, X_fit, feat_cols)
