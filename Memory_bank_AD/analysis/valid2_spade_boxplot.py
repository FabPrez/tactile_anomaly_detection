# val2_spade_seed.py
# ------------------------------------------------------------
# Validazione 2 (GF = 0.00, 0.05, 0.10, 0.15) + confronto con
# "GF stabile" preso dalla Validazione 1 (es. 0.60).
#
# Output (per ogni metrica e pezzo):
# 1) OVERVIEW: scatter (seed chiari) + media piena
# 2) ZOOM "fine": boxplot + seed + media su [0.05, 0.10, 0.15, 0.60]
# 3) ZOOM GF=0.00: boxplot + seed + media solo su GF=0.00
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# ===== IMPORT DA VALIDAZIONE 1 (SPADE) =====
# Se il tuo file si chiama diversamente, cambia qui.
from valid1_spade_seed import good_fractions as good_fractions_full
from valid1_spade_seed import results_roc_by_method as results_roc_full
from valid1_spade_seed import results_pro_by_method as results_pro_full
from valid1_spade_seed import results_pr_by_method as results_pr_full


# ===========================
# ASCISSE VALIDAZIONE 2 (% GOOD)
# ===========================
good_fractions = np.array([0.00, 0.05, 0.10, 0.15], dtype=float)


# ===========================
# ROC (Pixel AUROC) PER SEED
# SPADE – PZ1, PZ3 (Val2)
# ===========================
results_roc_by_method = {
    "SPADE": {
        "PZ1": [
            [0.9562, 0.9764379293698949, 0.9746092223775805, 0.9743986183162644],
            [0.9562, 0.9753453643256644, 0.9744819732019336, 0.9744898599710651],
            [0.9562, 0.9757466312780938, 0.9758655609125192, 0.9751176943115992],
            [0.9562, 0.9751986261485392, 0.9763286592949088, 0.9762875114371196],
            [0.9562, 0.9768257128234056, 0.9751861306469718, 0.9733946374980187],
            [0.9562, 0.9749912893008035, 0.9753231033157840, 0.9754003891736901],
            [0.9562, 0.9755392527393044, 0.9746995474229246, 0.9742973441811620],
            [0.9562, 0.9747667091907849, 0.9758814260006098, 0.9744441826504899],
            [0.9562, 0.9745433675321916, 0.9748916793163351, 0.9768827943519941],
            [0.9562, 0.9767531298623053, 0.9767148132132340, 0.9744467466693404],
        ],
        "PZ3": [
            [0.9631, 0.9801378526453456, 0.9802495833373293, 0.9799945683877391],
            [0.9631, 0.9814109831454922, 0.9804301033894431, 0.9806290822679737],
            [0.9631, 0.9821174704500262, 0.9811844192060020, 0.9811407844952523],
            [0.9631, 0.9819432779447198, 0.9808725099810146, 0.9809285464470283],
            [0.9631, 0.9829532814657571, 0.9812033977981454, 0.9808413727031524],
            [0.9631, 0.9813648441358966, 0.9810185646624561, 0.9810033399853216],
            [0.9631, 0.9806646500788777, 0.9810652248633158, 0.9807883833509842],
            [0.9631, 0.9826724577734777, 0.9812386416664871, 0.9813414654592019],
            [0.9631, 0.9829000730930437, 0.9809355231378762, 0.9810066981008122],
            [0.9631, 0.9828158512667403, 0.9807654802360162, 0.9809456115731005],
        ],
    }
}


# ===========================
# PRO (Pixel AUC-PRO) PER SEED
# SPADE – PZ1, PZ3 (Val2)
# ===========================
results_pro_by_method = {
    "SPADE": {
        "PZ1": [
            [0.1286, 0.13314642028882734, 0.13264384537341165, 0.13209054304661616],
            [0.1286, 0.12819877226351280, 0.13328647950324687, 0.13066387941883362],
            [0.1286, 0.13032460854967584, 0.12854105716833755, 0.13204378582703420],
            [0.1286, 0.12845775319585886, 0.12679940295305137, 0.12634418936229003],
            [0.1286, 0.13281695540093208, 0.13272346507130456, 0.13218288014327412],
            [0.1286, 0.12550749497025182, 0.12638167075920953, 0.13275522286780644],
            [0.1286, 0.12823335137820024, 0.13027383420522020, 0.13102248605653255],
            [0.1286, 0.13145526905713428, 0.12663063331787014, 0.13166164512096860],
            [0.1286, 0.13210237195002272, 0.13098004928827633, 0.13143506249015297],
            [0.1286, 0.13151975552215622, 0.13164714979046496, 0.13007779885768547],
        ],
        "PZ3": [
            [0.3397, 0.31394197844841730, 0.31312885814892420, 0.32241105959860010],
            [0.3397, 0.33566074307873583, 0.32638357725977280, 0.33007675879640114],
            [0.3397, 0.33341642499468530, 0.32503662504017340, 0.33031855010008840],
            [0.3397, 0.33554966727500070, 0.33457911622971580, 0.33601234280559267],
            [0.3397, 0.32850702421404580, 0.33381823703883290, 0.33524499702177490],
            [0.3397, 0.31728724038085720, 0.32803837298954330, 0.33365379319197824],
            [0.3397, 0.32442019124877030, 0.33653759282147260, 0.33273825100197724],
            [0.3397, 0.32727544539170267, 0.33271024061774970, 0.32840970760481220],
            [0.3397, 0.34474242680925415, 0.33293745882778830, 0.33503966523334033],
            [0.3397, 0.33476380829662444, 0.32945431814176790, 0.33045253889444826],
        ],
    }
}


# ===========================
# PR (Pixel AUPRC) PER SEED
# SPADE – PZ1, PZ3 (Val2)
# ===========================
results_pr_by_method = {
    "SPADE": {
        "PZ1": [
            [0.3529, 0.39057668995263345, 0.38671060630891274, 0.38584603640372184],
            [0.3529, 0.37509290189101985, 0.38258176857440420, 0.38302814275871090],
            [0.3529, 0.37632566211723834, 0.37574304218897340, 0.38454803244822210],
            [0.3529, 0.36988890467204220, 0.37577835388834800, 0.37571295984133085],
            [0.3529, 0.39005961367673180, 0.38740953760416624, 0.38217934525584357],
            [0.3529, 0.37023631360214193, 0.36981876736142566, 0.38477223331843670],
            [0.3529, 0.37471093924243590, 0.38166944208319200, 0.38242091121117360],
            [0.3529, 0.36863828410092000, 0.37447511825448100, 0.38422662290355064],
            [0.3529, 0.37065019050012460, 0.37059995813158060, 0.38346112624942770],
            [0.3529, 0.38961369394491190, 0.38860196964337296, 0.38287183688415080],
        ],
        "PZ3": [
            [0.3712, 0.38381328575189780, 0.38666273396907470, 0.39023115674966920],
            [0.3712, 0.40163987003745560, 0.39663527975973190, 0.39582299860488710],
            [0.3712, 0.40128650428851714, 0.39167822443773240, 0.39411923700830710],
            [0.3712, 0.39656214000907890, 0.39668557538345120, 0.39709538091875940],
            [0.3712, 0.40462482592479565, 0.39777298637425170, 0.39589658509734604],
            [0.3712, 0.39372138294589640, 0.39511987489250670, 0.39789648787087760],
            [0.3712, 0.39240152914675697, 0.39928207773128743, 0.39985572382289390],
            [0.3712, 0.39695609609153126, 0.39759720610002336, 0.39582874166213990],
            [0.3712, 0.41103057994026640, 0.39822799403423570, 0.40260821264009490],
            [0.3712, 0.40809088107715347, 0.39679907955199534, 0.39624284626052580],
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
# Plot helpers (compatibili con versioni matplotlib diverse)
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
        # indici su data: 0->0.00, 1->0.05, 2->0.10, 3->0.15, 4->static (Val1)
        plot_zoom_fine_box(
            data, labels, select_idxs=[1, 2, 3, 4],
            title=title_zoomfine, ylabel=ylabel
        )

    if do_zoom_gf0:
        plot_zoom_gf0(data[0], labels[0], title=title_gf0, ylabel=ylabel)


# =================================================
# RUN
# =================================================
METHOD = "SPADE"
STATIC_GF = 1.0

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
