
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

# ===== IMPORT DA VALIDAZIONE 1 (PADIM) =====
# Se il tuo file si chiama diversamente, cambia qui.
from valid1_Padim_seed import good_fractions as good_fractions_full
from valid1_Padim_seed import results_roc_by_method as results_roc_full
from valid1_Padim_seed import results_pro_by_method as results_pro_full
from valid1_Padim_seed import results_pr_by_method as results_pr_full


# ===========================
# ASCISSE VALIDAZIONE 2 (% GOOD)
# ===========================
good_fractions = np.array([0.00, 0.05, 0.10, 0.15], dtype=float)


# ===========================
# ROC (Pixel AUROC) PER SEED
# PADIM – PZ1, PZ3   (Val2)
# ===========================
results_roc_by_method = {
    "PADIM": {
        "PZ1": [
            [0.7732, 0.9674132264168334, 0.9704851260615408, 0.9711197786622975],
            [0.7722, 0.9677367345978742, 0.9694584287583834, 0.9706876426603802],
            [0.7752, 0.9676314934806013, 0.9705305412361362, 0.9713544282293248],
            [0.7746, 0.9678969836666941, 0.9710025168824411, 0.9714604137448778],
            [0.7765, 0.9682409035756087, 0.9708535140379249, 0.9715899656730799],
            [0.7750, 0.9686972705877266, 0.9706564583297904, 0.9709777464050428],
            [0.7752, 0.9681047082483079, 0.9704166855020485, 0.9714139841660551],
            [0.7761, 0.9668422207296388, 0.9697280677272011, 0.9713874212402603],
            [0.7774, 0.9675567825953881, 0.9697816355889295, 0.9708893778597214],
            [0.7761, 0.9675393235489497, 0.9702173312055618, 0.9709704529817419],
        ],
        "PZ3": [
            [0.8920, 0.9780718538251473, 0.9800715836414127, 0.9806442186049910],
            [0.9054, 0.9811196822535898, 0.9820355052616824, 0.9822922834158337],
            [0.8956, 0.9794655424875806, 0.9813618041188822, 0.9816765781308714],
            [0.9040, 0.9798802039877400, 0.9814236466874218, 0.9820999932757786],
            [0.8917, 0.9800392017645385, 0.9812596663305307, 0.9816812609963655],
            [0.9051, 0.9808090358263576, 0.9815925863899980, 0.9818007276783960],
            [0.9062, 0.9804437901154305, 0.9812420651036783, 0.9819570768129783],
            [0.9053, 0.9806057734858070, 0.9811656015946688, 0.9814194022271161],
            [0.9059, 0.9803166170463950, 0.9811461580613604, 0.9817660661574876],
            [0.8908, 0.9807041059102107, 0.9813561720553355, 0.9818160406215940],
        ],
    }
}


# ===========================
# PRO (Pixel AUC-PRO) PER SEED
# PADIM – PZ1, PZ3  (Val2)
# ===========================
results_pro_by_method = {
    "PADIM": {
        "PZ1": [
            [0.0310, 0.10177066092505680, 0.10286799226702029, 0.10215865589244658],
            [0.0311, 0.10214759152539822, 0.10201336825996588, 0.10121675395237723],
            [0.0318, 0.10073509130660324, 0.10121909216478765, 0.10194950266587605],  # <- fix 0.x
            [0.0304, 0.10297782872646027, 0.10348621612116860, 0.10337961020952996],
            [0.0315, 0.10246019457349742, 0.10264697647836364, 0.10244079504633341],
            [0.0306, 0.10166511230027395, 0.10075837280227642, 0.10050126880862037],
            [0.0316, 0.10093469659773213, 0.10069739788851549, 0.10113406383527070],
            [0.0319, 0.10182411501001900, 0.10139804620430910, 0.10255364748948755],
            [0.0319, 0.10193871791877383, 0.10087630012133562, 0.10113062046577811],
            [0.0313, 0.10292846631100580, 0.10265688315335987, 0.10210014138959636],
        ],
        "PZ3": [
            [0.0651, 0.31092392167792815, 0.29897739595329740, 0.29818122578048480],
            [0.0861, 0.30298542869678424, 0.30437126407899173, 0.30186762834037770],
            [0.0683, 0.30535339943612916, 0.29766751277488024, 0.29623677771762240],
            [0.0873, 0.31177649633253635, 0.30662657728784370, 0.30138164883282015],
            [0.0685, 0.30726231245761065, 0.29650998311838010, 0.29314295666739437],
            [0.0881, 0.30111812550593053, 0.29893375253502300, 0.29735690940233833],
            [0.0886, 0.32091250189961845, 0.31783329979031230, 0.31103480882644940],
            [0.0880, 0.30224218330755986, 0.29541009223847160, 0.29231519122966165],
            [0.0873, 0.30544725666818223, 0.30053519107414620, 0.29584909322348120],
            [0.0670, 0.29342554326772230, 0.28989020424405465, 0.29149178415436840],
        ],
    }
}


# ===========================
# PR (Pixel AUPRC) PER SEED
# PADIM – PZ1, PZ3  (Val2)
# ===========================
results_pr_by_method = {
    "PADIM": {
        "PZ1": [
            [0.0788, 0.29631913102506560, 0.30221583513514577, 0.30241919904941330],
            [0.0798, 0.29754859245460180, 0.29897046467521876, 0.29941568440246470],
            [0.0812, 0.29671665974146790, 0.30192842172842080, 0.30489653140976240],
            [0.0787, 0.29792394448566260, 0.30325797985937110, 0.30248353776671466],
            [0.0811, 0.29943563479739010, 0.30428990856850580, 0.30373244180713040],
            [0.0798, 0.29833510375648753, 0.29896063059541894, 0.29917046533360425],
            [0.0813, 0.29587266343392876, 0.29772870055761746, 0.30122863440257200],
            [0.0816, 0.29768607970128640, 0.30111132974145777, 0.30416213270249940],
            [0.0820, 0.29768368952683777, 0.29834050678491420, 0.30188137024908623],
            [0.0804, 0.30091536468990226, 0.30406233529289690, 0.30441742878119693],
        ],
        "PZ3": [
            [0.0682, 0.34112544972755643, 0.34218657999561300, 0.34368276927605180],
            [0.1026, 0.34859258185827474, 0.34540145259576227, 0.34391871325780643],
            [0.0829, 0.33782081370474760, 0.33942539354224070, 0.33957561948891910],
            [0.1037, 0.34790073550989015, 0.34870222907909940, 0.34256132813687910],
            [0.0740, 0.34351912752376310, 0.33857778510873580, 0.33663315096817290],
            [0.1048, 0.34579116320999587, 0.34510950272557170, 0.34491356366618400],
            [0.1035, 0.34793574751305034, 0.34668362118912520, 0.34563253898454540],
            [0.1043, 0.34670798582555173, 0.34452075886594300, 0.34266506713062110],
            [0.1034, 0.34595598497139170, 0.34572338436066260, 0.34435487656651886],
            [0.0687, 0.34267543837823510, 0.34257204434953004, 0.33843216558927550],
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
    ax.set_title(title)  # titolo esatto passato dal runner
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
METHOD = "PADIM"
STATIC_GF = 0.40

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
