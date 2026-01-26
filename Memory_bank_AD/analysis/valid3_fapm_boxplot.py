# val3_fapm_compare_boxplot.py
# ------------------------------------------------------------
# Confronto BOX PLOT tra:
# - Val1 (import da un tuo file: es. valid1_fapm_seed.py)
# - Val3 (hardcoded con i valori che mi hai mandato)
# Con ZOOM "vero" tramite BROKEN Y-AXIS (asse Y spezzato)
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ===========================
# CONFIG
# ===========================
METHOD = "FAPM"
TARGET_GF = 0.60
PIECES = ["PZ3", "PZ4", "PZ5"]

# Nome file Val1 SENZA ".py"
VAL1_MODULE = "valid1_fapm_seed"   # <-- cambia se serve

VAL1_LABEL = "Val1"
VAL3_LABEL = "Val3"

SHOW_FLIERS = True
SEED_ALPHA = 0.35
MEAN_ALPHA = 1.0
POINT_SIZE = 26
JITTER = 0.06
DPI = 140

# Boxplot options
NOTCH = False
SHOW_MEANS = True
MEANLINE = True
WHIS = (5, 95)  # baffi ai percentili 5–95

# ZOOM
USE_BROKEN_Y_AXIS = True     # <<< questo è lo "zoom vero"
ZOOM_PRC = (1, 99)           # percentili per definire la finestra
ZOOM_PAD = 0.35              # padding relativo (aumenta se vuoi più aria)
MIN_GAP = 1e-4               # distanza minima tra finestre per fare "break"


# ===========================
# VAL3: i tuoi dati (GF=0.60)
# ===========================
val3_good_fractions = np.array([0.60], dtype=float)

val3_results_roc_by_method = {
    "FAPM": {
        "PZ3": [[0.9934],[0.9935],[0.9935],[0.9935],[0.9935],[0.9935],[0.9937],[0.9937],[0.9936],[0.9935]],
        "PZ4": [[0.9979],[0.9980],[0.9979],[0.9979],[0.9979],[0.9980],[0.9979],[0.9979],[0.9979],[0.9979]],
        "PZ5": [[0.9982],[0.9982],[0.9982],[0.9981],[0.9982],[0.9982],[0.9982],[0.9981],[0.9983],[0.9982]],
    }
}

val3_results_pro_by_method = {
    "FAPM": {
        "PZ3": [[0.4978],[0.5124],[0.5139],[0.5009],[0.5020],[0.5054],[0.5178],[0.5028],[0.5094],[0.5056]],
        "PZ4": [[0.7650],[0.7722],[0.7668],[0.7685],[0.7677],[0.7777],[0.7675],[0.7657],[0.7703],[0.7657]],
        "PZ5": [[0.6788],[0.6790],[0.6632],[0.6548],[0.6727],[0.6777],[0.6810],[0.6473],[0.6848],[0.6758]],
    }
}

val3_results_pr_by_method = {
    "FAPM": {
        "PZ3": [[0.5113],[0.5181],[0.5159],[0.5116],[0.5112],[0.5178],[0.5172],[0.5158],[0.5159],[0.5166]],
        "PZ4": [[0.3899],[0.3966],[0.3955],[0.3935],[0.3945],[0.3983],[0.3880],[0.3966],[0.3968],[0.3915]],
        "PZ5": [[0.3281],[0.3363],[0.3321],[0.3220],[0.3320],[0.3340],[0.3330],[0.3257],[0.3348],[0.3344]],
    }
}


# ===========================
# IMPORT VAL1
# ===========================
try:
    val1_mod = __import__(VAL1_MODULE)
except Exception as e:
    raise ImportError(
        f"Non riesco a importare '{VAL1_MODULE}.py'. "
        f"Mettilo nella stessa cartella dello script o nel PYTHONPATH. Errore: {e}"
    )

val1_good_fractions = np.asarray(getattr(val1_mod, "good_fractions"), dtype=float)
val1_results_roc_by_method = getattr(val1_mod, "results_roc_by_method")
val1_results_pro_by_method = getattr(val1_mod, "results_pro_by_method")
val1_results_pr_by_method  = getattr(val1_mod, "results_pr_by_method")


# ===========================
# UTILS
# ===========================
def _as_2d_seedxgf(seed_lists, context=""):
    arr = np.asarray(seed_lists, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)   # assume 1 GF
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{context}: atteso 1D/2D, ottenuto shape={arr.shape}")

def extract_seed_values_at_gf(results_by_method, method, piece, gf_full, target_gf):
    gf_full = np.asarray(gf_full, dtype=float)
    idxs = np.where(np.isclose(gf_full, target_gf, atol=1e-9))[0]
    if len(idxs) == 0:
        raise ValueError(f"{method}-{piece}: target_gf={target_gf} non trovato in gf_full={gf_full}")
    j = int(idxs[0])

    curves = _as_2d_seedxgf(results_by_method[method][piece], context=f"{method}-{piece}")
    if j >= curves.shape[1]:
        raise ValueError(f"{method}-{piece}: indice GF={j} fuori range: curves.shape={curves.shape}")

    return curves[:, j].astype(float)  # (num_seeds,)

def _boxplot(ax, data, labels, showfliers=True):
    try:
        return ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=showfliers,
            notch=NOTCH,
            showmeans=SHOW_MEANS,
            meanline=MEANLINE,
            whis=WHIS
        )
    except TypeError:
        return ax.boxplot(
            data,
            labels=labels,
            showfliers=showfliers,
            notch=NOTCH,
            showmeans=SHOW_MEANS,
            meanline=MEANLINE,
            whis=WHIS
        )

def _range_from_percentiles(vals, prc=(1, 99), pad=0.35):
    vals = np.asarray(vals, dtype=float).ravel()
    lo, hi = np.percentile(vals, prc)
    span = max(hi - lo, 1e-12)
    ypad = pad * span
    return (lo - ypad, hi + ypad)

def _scatter_and_mean(ax, data):
    rng = np.random.default_rng(0)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, vals in enumerate(data, start=1):
        vals = np.asarray(vals, dtype=float).ravel()
        c = palette[(i - 1) % len(palette)]
        x = rng.normal(loc=i, scale=JITTER, size=len(vals))
        ax.scatter(x, vals, alpha=SEED_ALPHA, s=POINT_SIZE, color=c, edgecolors="none", zorder=2)
        ax.scatter(i, float(np.mean(vals)), alpha=MEAN_ALPHA, s=POINT_SIZE, color=c, edgecolors="none", zorder=5)

def plot_compare_box(vals_a, vals_b, title, ylabel, labels=(VAL1_LABEL, VAL3_LABEL)):
    vals_a = np.asarray(vals_a, dtype=float).ravel()
    vals_b = np.asarray(vals_b, dtype=float).ravel()
    data = [vals_a, vals_b]

    if not USE_BROKEN_Y_AXIS:
        fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=DPI)
        _boxplot(ax, data, list(labels), showfliers=SHOW_FLIERS)
        _scatter_and_mean(ax, data)

        # zoom "semplice" (qui però con Val1/Val3 lontani non basta)
        lo, hi = _range_from_percentiles(np.concatenate(data), prc=ZOOM_PRC, pad=ZOOM_PAD)
        ax.set_ylim(lo, hi)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    # ---- BROKEN Y-AXIS: due finestre, una per Val1 e una per Val3 ----
    mean_a = float(np.mean(vals_a))
    mean_b = float(np.mean(vals_b))

    # assegna "basso" e "alto" automaticamente
    if mean_a <= mean_b:
        low_vals, high_vals = vals_a, vals_b
        low_label, high_label = labels[0], labels[1]
        order = "A_low"
    else:
        low_vals, high_vals = vals_b, vals_a
        low_label, high_label = labels[1], labels[0]
        order = "B_low"

    low_ylim  = _range_from_percentiles(low_vals,  prc=ZOOM_PRC, pad=ZOOM_PAD)
    high_ylim = _range_from_percentiles(high_vals, prc=ZOOM_PRC, pad=ZOOM_PAD)

    # se le finestre si sovrappongono, niente break: fallback a singolo asse
    if low_ylim[1] + MIN_GAP >= high_ylim[0]:
        fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=DPI)
        _boxplot(ax, data, list(labels), showfliers=SHOW_FLIERS)
        _scatter_and_mean(ax, data)

        lo, hi = _range_from_percentiles(np.concatenate(data), prc=ZOOM_PRC, pad=ZOOM_PAD)
        ax.set_ylim(lo, hi)

        ax.set_title(title + " (no-break: range sovrapposti)")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(7.6, 5.4), dpi=DPI,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06}
    )

    # disegna lo stesso boxplot su entrambi gli assi
    _boxplot(ax_top, data, list(labels), showfliers=SHOW_FLIERS)
    _boxplot(ax_bot, data, list(labels), showfliers=SHOW_FLIERS)

    _scatter_and_mean(ax_top, data)
    _scatter_and_mean(ax_bot, data)

    # set limiti (basso e alto)
    if order == "A_low":
        ax_bot.set_ylim(*low_ylim)     # Val1 (basso)
        ax_top.set_ylim(*high_ylim)    # Val3 (alto)
    else:
        ax_bot.set_ylim(*low_ylim)     # Val3 (basso)
        ax_top.set_ylim(*high_ylim)    # Val1 (alto)

    # estetica break
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.008  # dimensione dei tagli diagonali
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_top.grid(True, axis="y", alpha=0.3)
    ax_bot.grid(True, axis="y", alpha=0.3)

    ax_top.set_title(title + " (broken y-axis zoom)")
    ax_bot.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()


# ===========================
# RUN: ROC / PRO / PR
# ===========================
for piece in PIECES:
    # ROC
    roc_val1 = extract_seed_values_at_gf(val1_results_roc_by_method, METHOD, piece, val1_good_fractions, TARGET_GF)
    roc_val3 = extract_seed_values_at_gf(val3_results_roc_by_method, METHOD, piece, val3_good_fractions, TARGET_GF)
    plot_compare_box(
        roc_val1, roc_val3,
        title=f"{METHOD} – {piece}: ROC (Pixel AUROC) @ GF={TARGET_GF:.2f}",
        ylabel="Pixel AUROC (ROC)"
    )

    # PRO
    pro_val1 = extract_seed_values_at_gf(val1_results_pro_by_method, METHOD, piece, val1_good_fractions, TARGET_GF)
    pro_val3 = extract_seed_values_at_gf(val3_results_pro_by_method, METHOD, piece, val3_good_fractions, TARGET_GF)
    plot_compare_box(
        pro_val1, pro_val3,
        title=f"{METHOD} – {piece}: PRO (Pixel AUC-PRO) @ GF={TARGET_GF:.2f}",
        ylabel="Pixel AUC-PRO"
    )

    # PR
    pr_val1 = extract_seed_values_at_gf(val1_results_pr_by_method, METHOD, piece, val1_good_fractions, TARGET_GF)
    pr_val3 = extract_seed_values_at_gf(val3_results_pr_by_method, METHOD, piece, val3_good_fractions, TARGET_GF)
    plot_compare_box(
        pr_val1, pr_val3,
        title=f"{METHOD} – {piece}: PR (Pixel AUPRC) @ GF={TARGET_GF:.2f}",
        ylabel="Pixel AUPRC (PR)"
    )
