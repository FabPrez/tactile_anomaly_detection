
# ------------------------------------------------------------
# Validazione 2 (GF = 0.00, 0.05, 0.10, 0.15) + confronto con
# "GF stabile" preso dalla Validazione 1 (es. 0.60).
#
# Output (per ogni metrica e pezzo):
# 1) OVERVIEW: strip/scatter (seed chiari) + media piena (stesso colore e size)
# 2) ZOOM "fine": boxplot + seed + media su [0.05, 0.10, 0.15, 0.60]
# 3) ZOOM GF=0.00: boxplot + seed + media solo su GF=0.00
# ------------------------------------------------------------

# ===== IMPORT DA VALIDAZIONE 1 (INREACH) =====
from valid1_inreach_seed import good_fractions as good_fractions_full
from valid1_inreach_seed import results_roc_by_method as results_roc_full
from valid1_inreach_seed import results_pro_by_method as results_pro_full
from valid1_inreach_seed import results_pr_by_method as results_pr_full

import matplotlib.pyplot as plt
import numpy as np


# ===========================
# ASCISSE VALIDAZIONE 2 (% GOOD)
# ===========================
good_fractions = np.array([0.00, 0.05, 0.10, 0.15], dtype=float)


# ===========================
# ROC (Pixel AUROC) PER SEED
# INREACH – PZ1, PZ3
# ===========================
results_roc_by_method = {
    "INREACH": {
        "PZ1": [
            [0.7861, 0.9597, 0.9613, 0.9619],
            [0.7835, 0.9606, 0.9618, 0.9619],
            [0.7912, 0.9605, 0.9621, 0.9621],
            [0.7880, 0.9610, 0.9625, 0.9623],
            [0.7954, 0.9609, 0.9617, 0.9622],
            [0.7844, 0.9587, 0.9617, 0.9623],
            [0.7839, 0.9608, 0.9618, 0.9624],
            [0.7858, 0.9606, 0.9625, 0.9625],
            [0.7891, 0.9606, 0.9611, 0.9622],
            [0.7877, 0.9615, 0.9619, 0.9626],
        ],
        "PZ3": [
            [0.9473, 0.9666, 0.9756, 0.9754],
            [0.9494, 0.9745, 0.9754, 0.9753],
            [0.9494, 0.9740, 0.9754, 0.9754],
            [0.9444, 0.9747, 0.9755, 0.9751],
            [0.9488, 0.9750, 0.9757, 0.9750],
            [0.9464, 0.9711, 0.9754, 0.9748],
            [0.9485, 0.9751, 0.9761, 0.9760],
            [0.9486, 0.9760, 0.9756, 0.9757],
            [0.9465, 0.9748, 0.9753, 0.9757],
            [0.9464, 0.9755, 0.9752, 0.9755],
        ],
    }
}


# ===========================
# PRO (Pixel AUC-PRO) PER SEED
# INREACH – PZ1, PZ3
# ===========================
results_pro_by_method = {
    "INREACH": {
        "PZ1": [
            [0.0270, 0.0630, 0.0629, 0.0647],
            [0.0262, 0.0634, 0.0631, 0.0631],
            [0.0312, 0.0640, 0.0635, 0.0646],
            [0.0275, 0.0641, 0.0653, 0.0627],
            [0.0311, 0.0625, 0.0629, 0.0638],
            [0.0262, 0.0636, 0.0611, 0.0619],
            [0.0266, 0.0642, 0.0622, 0.0646],
            [0.0271, 0.0624, 0.0646, 0.0641],
            [0.0296, 0.0617, 0.0640, 0.0664],
            [0.0283, 0.0654, 0.0645, 0.0633],
        ],
        "PZ3": [
            [0.2260, 0.1972, 0.1990, 0.1924],
            [0.2308, 0.2004, 0.2042, 0.1988],
            [0.2241, 0.2037, 0.1968, 0.1955],
            [0.2218, 0.1916, 0.1990, 0.1977],
            [0.2246, 0.2014, 0.2041, 0.2025],
            [0.2264, 0.2071, 0.1990, 0.1904],
            [0.2263, 0.2074, 0.2031, 0.2040],
            [0.2255, 0.1996, 0.1982, 0.1934],
            [0.2279, 0.1992, 0.1974, 0.2029],
            [0.2253, 0.2007, 0.1933, 0.2011],
        ],
    }
}


# ===========================
# PR (Pixel AUPRC) PER SEED
# INREACH – PZ1, PZ3
# ===========================
results_pr_by_method = {
    "INREACH": {
        "PZ1": [
            [0.0804, 0.2395, 0.2416, 0.2444],
            [0.0821, 0.2369, 0.2373, 0.2366],
            [0.0873, 0.2356, 0.2403, 0.2407],
            [0.0842, 0.2375, 0.2511, 0.2383],
            [0.0875, 0.2381, 0.2400, 0.2423],
            [0.0818, 0.2338, 0.2376, 0.2414],
            [0.0823, 0.2366, 0.2415, 0.2422],
            [0.0838, 0.2425, 0.2547, 0.2514],
            [0.0863, 0.2395, 0.2414, 0.2565],
            [0.0846, 0.2519, 0.2513, 0.2479],
        ],
        "PZ3": [
            [0.2758, 0.2800, 0.2825, 0.2783],
            [0.2826, 0.2800, 0.2736, 0.2778],
            [0.2800, 0.2862, 0.2784, 0.2757],
            [0.2671, 0.2811, 0.2822, 0.2763],
            [0.2738, 0.2767, 0.2798, 0.2801],
            [0.2707, 0.2833, 0.2801, 0.2719],
            [0.2753, 0.2895, 0.2787, 0.2773],
            [0.2722, 0.2803, 0.2780, 0.2736],
            [0.2747, 0.2806, 0.2821, 0.2790],
            [0.2748, 0.2861, 0.2785, 0.2788],
        ],
    }
}


# =================================================
# UTILITY: Estrae (num_seeds,) dalla Validazione 1 al GF target (es. 0.60)
# =================================================
def extract_seed_values_at_gf(results_by_method, method, piece, gf_full, target_gf):
    gf_full = np.asarray(gf_full, dtype=float)

    idx = np.where(np.isclose(gf_full, target_gf, atol=1e-9))[0]
    if len(idx) == 0:
        raise ValueError(f"target_gf={target_gf} non trovato in gf_full={gf_full}")
    idx = int(idx[0])

    curves_full = np.asarray(results_by_method[method][piece], dtype=float)  # (num_seeds, num_gf_full)
    if curves_full.ndim != 2:
        raise ValueError(
            f"Val1 {method}-{piece}: atteso array 2D (seeds x gf). Ottenuto ndim={curves_full.ndim}"
        )
    if idx >= curves_full.shape[1]:
        raise ValueError(f"Indice GF={idx} fuori range: shape={curves_full.shape}")

    return curves_full[:, idx]  # (num_seeds,)


# =================================================
# Plot helpers (robusti anche se matplotlib non supporta tick_labels)
# =================================================
def _boxplot(ax, data, labels, showfliers=True):
    try:
        return ax.boxplot(data, tick_labels=labels, showfliers=showfliers)
    except TypeError:
        return ax.boxplot(data, labels=labels, showfliers=showfliers)


def _set_ylim_from_percentiles(ax, vals, prc=(5, 95), pad=0.15):
    vals = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(vals, prc)
    span = max(hi - lo, 1e-9)
    ax.set_ylim(lo - pad * span, hi + pad * span)


def build_data(values_4gf, gf_4, static_vals, static_gf):
    curves4 = np.asarray(values_4gf, dtype=float)  # (num_seeds, 4)
    if curves4.ndim != 2 or curves4.shape[1] != len(gf_4):
        raise ValueError(f"Atteso (num_seeds x {len(gf_4)}). Ottenuto {curves4.shape}")

    static_vals = np.asarray(static_vals, dtype=float)  # (num_seeds,)
    if static_vals.ndim != 1:
        raise ValueError(f"static_vals deve essere 1D. Ottenuto {static_vals.shape}")

    if static_vals.shape[0] != curves4.shape[0]:
        raise ValueError(f"Mismatch seeds: Val2={curves4.shape[0]} vs static={static_vals.shape[0]}")

    data = [curves4[:, j] for j in range(len(gf_4))] + [static_vals]
    labels = [f"{g:.2f}" for g in gf_4] + [f"{static_gf:.2f}"]
    return data, labels


# =================================================
# FIGURA 1: OVERVIEW (solo punti + media)
# =================================================
def plot_overview_scatter(
    data, labels, title, ylabel,
    seed_alpha=0.35, mean_alpha=1.0,
    s=26, jitter=0.06
):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rng = np.random.default_rng(0)

    for i, vals in enumerate(data, start=1):
        c = palette[(i - 1) % len(palette)]
        vals = np.asarray(vals, dtype=float)

        x = rng.normal(loc=i, scale=jitter, size=len(vals))
        ax.scatter(x, vals, alpha=seed_alpha, s=s, color=c, edgecolors="none", zorder=2)
        ax.scatter(i, float(np.mean(vals)), alpha=mean_alpha, s=s, color=c, edgecolors="none", zorder=5)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Good Fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in data])
    y0, y1 = float(np.min(all_vals)), float(np.max(all_vals))
    span = max(y1 - y0, 1e-9)
    ax.set_ylim(y0 - 0.08 * span, y1 + 0.08 * span)

    plt.tight_layout()
    plt.show()


# =================================================
# FIGURA 2: ZOOM "fine" (boxplot su 0.05/0.10/0.15/static)
# =================================================
def plot_zoom_fine_box(
    data, labels, select_idxs,
    title, ylabel,
    prc=(5, 95), pad=0.15,
    seed_alpha=0.35, mean_alpha=1.0,
    s=26, jitter=0.06
):
    data_sel = [data[i] for i in select_idxs]
    labels_sel = [labels[i] for i in select_idxs]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    _boxplot(ax, data_sel, labels_sel, showfliers=True)

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rng = np.random.default_rng(0)

    for pos, orig_idx in enumerate(select_idxs, start=1):
        vals = np.asarray(data[orig_idx], dtype=float)
        c = palette[orig_idx % len(palette)]

        x = rng.normal(loc=pos, scale=jitter, size=len(vals))
        ax.scatter(x, vals, alpha=seed_alpha, s=s, color=c, edgecolors="none", zorder=2)
        ax.scatter(pos, float(np.mean(vals)), alpha=mean_alpha, s=s, color=c, edgecolors="none", zorder=5)

    ax.set_xlabel("Good Fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    all_sel = np.concatenate([np.asarray(v, dtype=float) for v in data_sel])
    _set_ylim_from_percentiles(ax, all_sel, prc=prc, pad=pad)

    plt.tight_layout()
    plt.show()


# =================================================
# FIGURA 3: ZOOM GF=0.00 (boxplot solo su 0.00)
# =================================================
def plot_zoom_gf0(
    data0, label0,
    title, ylabel,
    prc=(5, 95), pad=0.15,
    seed_alpha=0.35, mean_alpha=1.0,
    s=26, jitter=0.06
):
    vals0 = np.asarray(data0, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=140)
    _boxplot(ax, [vals0], [label0], showfliers=True)

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c0 = palette[0 % len(palette)]
    rng = np.random.default_rng(0)

    x0 = rng.normal(loc=1, scale=jitter, size=len(vals0))
    ax.scatter(x0, vals0, alpha=seed_alpha, s=s, color=c0, edgecolors="none", zorder=2)
    ax.scatter(1, float(np.mean(vals0)), alpha=mean_alpha, s=s, color=c0, edgecolors="none", zorder=5)

    ax.set_xlabel("Good Fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    _set_ylim_from_percentiles(ax, vals0, prc=prc, pad=pad)

    plt.tight_layout()
    plt.show()


# =================================================
# Runner: per pezzo+metrica crea le figure richieste
# =================================================
def run_piece_metric(
    method, piece, metric_short, ylabel,
    values_4gf, static_vals, static_gf,
    do_overview=True, do_zoom_fine=True, do_zoom_gf0=True
):
    data, labels = build_data(values_4gf, good_fractions, static_vals, static_gf)

    title_overview = f"{method} – {piece}: {metric_short} – Overview"
    title_zoomfine = f"{method} – {piece}: {metric_short} – Zoom GF=0.05–0.15 + {static_gf:.2f} (Val1)"
    title_gf0      = f"{method} – {piece}: {metric_short} – ZOOM GF=0.00 (Val2)"

    if do_overview:
        plot_overview_scatter(data, labels, title_overview, ylabel)

    if do_zoom_fine:
        # indici su data: 0->0.00, 1->0.05, 2->0.10, 3->0.15, 4->static
        plot_zoom_fine_box(
            data, labels, select_idxs=[1, 2, 3, 4],
            title=title_zoomfine, ylabel=ylabel
        )

    if do_zoom_gf0:
        plot_zoom_gf0(data[0], labels[0], title=title_gf0, ylabel=ylabel)


# =================================================
# RUN
# =================================================
METHOD = "INREACH"
STATIC_GF = 0.20

DO_OVERVIEW = True
DO_ZOOM_FINE = True
DO_ZOOM_GF0 = True

PIECES = ["PZ1", "PZ3"]

for piece in PIECES:
    # ROC
    roc_static = extract_seed_values_at_gf(results_roc_full, METHOD, piece, good_fractions_full, STATIC_GF)
    run_piece_metric(
        method=METHOD,
        piece=piece,
        metric_short="ROC",
        ylabel="Pixel AUROC (ROC)",
        values_4gf=results_roc_by_method[METHOD][piece],
        static_vals=roc_static,
        static_gf=STATIC_GF,
        do_overview=DO_OVERVIEW,
        do_zoom_fine=DO_ZOOM_FINE,
        do_zoom_gf0=DO_ZOOM_GF0,
    )

    # PRO
    pro_static = extract_seed_values_at_gf(results_pro_full, METHOD, piece, good_fractions_full, STATIC_GF)
    run_piece_metric(
        method=METHOD,
        piece=piece,
        metric_short="PRO",
        ylabel="Pixel AUC-PRO",
        values_4gf=results_pro_by_method[METHOD][piece],
        static_vals=pro_static,
        static_gf=STATIC_GF,
        do_overview=DO_OVERVIEW,
        do_zoom_fine=DO_ZOOM_FINE,
        do_zoom_gf0=DO_ZOOM_GF0,
    )

    # PR
    pr_static = extract_seed_values_at_gf(results_pr_full, METHOD, piece, good_fractions_full, STATIC_GF)
    run_piece_metric(
        method=METHOD,
        piece=piece,
        metric_short="PR",
        ylabel="Pixel AUPRC (PR)",
        values_4gf=results_pr_by_method[METHOD][piece],
        static_vals=pr_static,
        static_gf=STATIC_GF,
        do_overview=DO_OVERVIEW,
        do_zoom_fine=DO_ZOOM_FINE,
        do_zoom_gf0=DO_ZOOM_GF0,
    )
