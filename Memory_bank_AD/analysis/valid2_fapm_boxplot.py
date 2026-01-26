# val2_fapm_seed.py
# ------------------------------------------------------------
# Validazione 2 (GF = 0.00, 0.05, 0.10, 0.15) + confronto con
# "GF stabile" preso dalla Validazione 1 (es. 0.60).
#
# Output (per ogni metrica e pezzo):
# 1) OVERVIEW: strip/scatter (seed chiari) + media piena (stesso colore e size)
# 2) ZOOM "fine": boxplot + seed + media su [0.05, 0.10, 0.15, 0.60]
# 3) ZOOM GF=0.00: boxplot + seed + media solo su GF=0.00
# ------------------------------------------------------------

# ===== IMPORT DA VALIDAZIONE 1 =====
from valid1_fapm_seed import good_fractions as good_fractions_full
from valid1_fapm_seed import results_roc_by_method as results_roc_full
from valid1_fapm_seed import results_pro_by_method as results_pro_full
from valid1_fapm_seed import results_pr_by_method as results_pr_full

import matplotlib.pyplot as plt
import numpy as np


# ===========================
# ASCISSE VALIDAZIONE 2 (% GOOD)
# ===========================
good_fractions = np.array([0.00, 0.05, 0.10, 0.15], dtype=float)


# ===========================
# ROC (Pixel AUROC) PER SEED
# FAPM – PZ1, PZ3
# ===========================
results_roc_by_method = {
    "FAPM": {
        "PZ1": [
            [0.7502, 0.9620,               0.9628,              0.9632],
            [0.7509, 0.9616430532736453,   0.9626913431817049, 0.9633788414613481],
            [0.7511, 0.9616430532736453,   0.9626913431817049, 0.9633788414613481],
            [0.7502, 0.9618543463458,      0.963174121152078,  0.9630300367188988],
            [0.7505, 0.9620402355786135,   0.9630843500982582, 0.9631481172290135],
            [0.7510, 0.9624980802455289,   0.9628359173594819, 0.9631633611917629],
            [0.7505, 0.9614191092599451,   0.9623812207238769, 0.963156147343142],
            [0.7513, 0.9621259253076196,   0.9627338012601016, 0.9635486090377812],
            [0.7516, 0.961934310418132,    0.961924630955491,  0.9627905663201501],
            [0.7515, 0.9630368918114842,   0.9634051861464068, 0.9636208733270452],
        ],
        "PZ3": [
            [0.8834, 0.9742756180498299, 0.9746755658014843, 0.9748679931819074],
            [0.8892, 0.9744237584096592, 0.9754436845469231, 0.9754834148466498],
            [0.8878, 0.974682826451966,  0.9753089958722114, 0.9753156760883186],
            [0.8875, 0.9744314666171557, 0.975141134894109,  0.9752392447680658],
            [0.8889, 0.9746169804561842, 0.9748760211849484, 0.9753500864184387],
            [0.8881, 0.974219172374373,  0.9746315940900036, 0.9747485754309835],
            [0.8885, 0.9746604021077266, 0.9752692533103949, 0.975006471385377],
            [0.8870, 0.9746214786004256, 0.974682640656932,  0.9747256696500755],
            [0.8900, 0.974233146349221,  0.9748192971532689, 0.974900575534999],
            [0.8879, 0.9746288204756802, 0.9742835857915194, 0.9755514543000826],
        ],
    }
}


# ===========================
# PRO (Pixel AUC-PRO) PER SEED
# FAPM – PZ1, PZ3
# ===========================
results_pro_by_method = {
    "FAPM": {
        "PZ1": [
            [0.0037, 0.06511809717953648, 0.0647471453694562,  0.0642897119234744],
            [0.0037, 0.06428362031219936, 0.06430673580375751, 0.06399306863201319],
            [0.0035, 0.06512982181871016, 0.06436483297130087, 0.06395016196910719],
            [0.0032, 0.06462775620993476, 0.0636480820901414,  0.06324582541043715],
            [0.0036, 0.06540667382331136, 0.06412501038962272, 0.06379022269870278],
            [0.0032, 0.06480597035233136, 0.06413093690251467, 0.06374956077356034],
            [0.0036, 0.06527983441265925, 0.06511190098286956, 0.06390081467813193],
            [0.0036, 0.06471564185477434, 0.06459936237187874, 0.06389342442068206],
            [0.0038, 0.06410976585399608, 0.06407441420355285, 0.06377753453185916],
            [0.0038, 0.06441363390342998, 0.06508900954803606, 0.06495195345761075],
        ],
        "PZ3": [
            [0.0712, 0.20172735858703755, 0.20364842755245358, 0.20193868134924722],
            [0.1020, 0.2103783777338128,  0.2054393775633519,  0.20671555986935242],
            [0.1008, 0.2059415428377529,  0.2010878645149975,  0.20170118877891183],
            [0.1043, 0.20420506344450673, 0.19758700401881235, 0.20531387350310368],
            [0.1061, 0.20505701175011795, 0.20024907586111457, 0.20212056919016097],
            [0.0984, 0.200558674695798,   0.20237562509309814, 0.19961135582252007],
            [0.1017, 0.204255197470482,   0.20385308802017083, 0.20178208694167635],
            [0.1067, 0.21093367490241902, 0.208887129427938,   0.20764594403212228],
            [0.1050, 0.20579762198890938, 0.20380851703412842, 0.20813433498048792],
            [0.1015, 0.2046228176994478,  0.20493644722717924, 0.2028070677135662],
        ],
    }
}


# ===========================
# PR (Pixel AUPRC) PER SEED
# FAPM – PZ1, PZ3
# ===========================
results_pr_by_method = {
    "FAPM": {
        "PZ1": [
            [0.0340, 0.2456993318649111,  0.24687840522579282, 0.24599828903886226],
            [0.0341, 0.2422822082088868,  0.24155315354173476, 0.24130648782241998],
            [0.0336, 0.2420555923285774,  0.24390349038501624, 0.24704372785998888],
            [0.0324, 0.24367777614473762, 0.24762562978070413, 0.24521463488741152],
            [0.0338, 0.24251409710286215, 0.24693961966629058, 0.245631958745761],
            [0.0323, 0.24529054849817733, 0.24333937255399912, 0.2435723998385499],
            [0.0339, 0.2405619233061814,  0.24273183053105674, 0.24523017114189455],
            [0.0341, 0.24493153471255555, 0.2452476371035196,  0.24582156644578249],
            [0.0349, 0.244412783568728,   0.24290844239240855, 0.24675356295950673],
            [0.0348, 0.24773804371633282, 0.24815665337749876, 0.24826604753816459],
        ],
        "PZ3": [
            [0.0674, 0.2612442155731325,  0.2612014633839659,  0.2589240440014156],
            [0.0852, 0.2539875041724812,  0.2470150946083372,  0.24998575252146485],
            [0.0840, 0.2572796583455544,  0.2514521366110157,  0.2528644945287004],
            [0.0857, 0.26128391023914754, 0.25010705521331045, 0.24711479698679295],
            [0.0888, 0.25248146067135163, 0.2502306376600505,  0.24934520873427762],
            [0.0837, 0.24562108597354332, 0.2480481956801442,  0.24728480995290852],
            [0.0852, 0.24669115262875702, 0.24733261105675197, 0.2453957798448853],
            [0.0839, 0.25766488906950147, 0.25493728156901646, 0.2518633343423973],
            [0.0893, 0.2595520976570394,  0.25897956102545944, 0.25353267110040967],
            [0.0862, 0.2508224626319957,  0.2519351090300205,  0.24885281799295533],
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

        # seed "chiari"
        x = rng.normal(loc=i, scale=jitter, size=len(vals))
        ax.scatter(x, vals, alpha=seed_alpha, s=s, color=c, edgecolors="none", zorder=2)

        # media "piena" (stesso colore e stessa size)
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
    ax.set_title(title)  # <<< QUI: titolo esatto che passi dal runner
    ax.grid(True, axis="y", alpha=0.3)

    _set_ylim_from_percentiles(ax, vals0, prc=prc, pad=pad)

    plt.tight_layout()
    plt.show()


# =================================================
# Runner: per pezzo+metrica crea le figure richieste
# =================================================
def run_piece_metric(
    piece, metric_short, ylabel,
    values_4gf, static_vals, static_gf,
    do_overview=True, do_zoom_fine=True, do_zoom_gf0=True
):
    data, labels = build_data(values_4gf, good_fractions, static_vals, static_gf)

    # Titoli compatti (puoi cambiare lo stile qui e si propaga ovunque)
    title_overview = f"FAPM – {piece}: {metric_short} – Overview"
    title_zoomfine = f"FAPM – {piece}: {metric_short} – Zoom GF=0.05–0.15 + {static_gf:.2f} (Val1)"
    title_gf0      = f"FAPM – {piece}: {metric_short} – ZOOM GF=0.00 (Val2)"

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
METHOD = "FAPM"
STATIC_GF = 0.60

# Toggle figure (se vuoi ridurre il numero di finestre)
DO_OVERVIEW = True
DO_ZOOM_FINE = True
DO_ZOOM_GF0 = True

PIECES = ["PZ1", "PZ3"]

for piece in PIECES:
    # ROC
    roc_static = extract_seed_values_at_gf(results_roc_full, METHOD, piece, good_fractions_full, STATIC_GF)
    run_piece_metric(
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
