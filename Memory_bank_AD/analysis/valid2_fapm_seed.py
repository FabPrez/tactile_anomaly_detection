from val1_fapm_seed import good_fractions as good_fractions_full
from val1_fapm_seed import results_roc_by_method as results_roc_full
from val1_fapm_seed import results_pro_by_method as results_pro_full
from val1_fapm_seed import results_pr_by_method  as results_pr_full



import matplotlib.pyplot as plt
import numpy as np

# ===========================
# ASCISSE COMUNI (% GOOD)
# ===========================

good_fractions = np.array([
    0.00, 0.05, 0.10, 0.15, 
])

# ===========================
# ROC (Pixel AUROC) PER SEED
# SPADE – PZ1, PZ3
# ===========================

results_roc_by_method = {
    "FAPM": {

        "PZ1": [
            # ----- SEED 0 -----
            [   
                0.7502,0.9620, 0.9628, 
                0.9632
                
            ],

            # ----- SEED 1 -----
           
            [   0.7509,0.9616430532736453, 0.9626913431817049, 
             0.9633788414613481
            
            ],

            # ----- SEED 2 -----
            [
                0.7511,0.9616430532736453, 0.9626913431817049, 
                0.9633788414613481
            
            ],

            # ----- SEED 3 -----
            [
                0.7502, 0.9618543463458, 0.963174121152078, 
                0.9630300367188988
            ],
            # ----- SEED 4 -----
            [
                0.7505, 0.9620402355786135, 0.9630843500982582, 0.9631481172290135
            ],
            # ----- SEED 5 -----
            [
                0.7510, 0.9624980802455289, 0.9628359173594819, 0.9631633611917629
            ],
            # ----- SEED 6 -----
            [
                0.7505, 0.9614191092599451, 0.9623812207238769, 0.963156147343142
            ],
            # ----- SEED 7 -----
            [
                0.7513, 0.9621259253076196, 0.9627338012601016, 0.9635486090377812
            ],
            # ----- SEED 8 -----
            [
                0.7516,0.961934310418132, 0.961924630955491, 0.9627905663201501
            ],
            # ----- SEED 9 -----
            [
                0.7515, 0.9630368918114842, 0.9634051861464068, 0.9636208733270452
            ]
        ],

        "PZ3": [
            # ----- SEED 0 -----
            [
                0.8834, 0.9742756180498299, 0.9746755658014843, 
                0.9748679931819074
            ], 
            # ----- SEED 1 -----
            [
                0.8892, 0.9744237584096592, 0.9754436845469231, 
                0.9754834148466498
            ],
            # ----- SEED 2 -----
            [
                0.8878, 0.974682826451966, 0.9753089958722114, 
                0.9753156760883186
            ],
            # ----- SEED 3 -----
            [
                0.8875, 0.9744314666171557, 0.975141134894109, 
                0.9752392447680658
            ],
            # ----- SEED 4 -----
            [
                0.8889, 0.9746169804561842, 0.9748760211849484, 
                0.9753500864184387
            ],
            # ----- SEED 5 -----
            [
                0.8881, 0.974219172374373, 0.9746315940900036, 
                0.9747485754309835
            ],
            # ----- SEED 6 -----
            [
                0.8885, 0.9746604021077266, 0.9752692533103949, 
                0.975006471385377
            ],
            # ----- SEED 7 -----
            [
                0.8870, 0.9746214786004256, 0.974682640656932, 
                0.9747256696500755
            ],
            # ----- SEED 8 -----
            [
                0.8900, 0.974233146349221, 0.9748192971532689, 
                0.974900575534999
            ],
            # ----- SEED 9 -----
            [
                0.8879, 0.9746288204756802, 0.9742835857915194, 
                0.9755514543000826
            ]
        ],

    }
}


# ===========================
# PRO (Pixel AUC-PRO) PER SEED
# FAPM – PZ1..PZ5
# Ogni pezzo: lista di liste [seed0 ... seed9]
# ===========================

results_pro_by_method = {

    "FAPM": {

        "PZ1": [
            # ----- SEED 0 -----
            [
                0.0037, 0.06511809717953648, 0.0647471453694562, 
                0.0642897119234744
            ],
            # ----- SEED 1 -----
            [
                0.0037, 0.06428362031219936, 0.06430673580375751, 
                0.06399306863201319
            ],
            # ----- SEED 2 -----
            [
                0.0035, 0.06512982181871016, 0.06436483297130087, 
                0.06395016196910719
            ],
            # ----- SEED 3 -----
            [
                0.0032, 0.06462775620993476, 0.0636480820901414, 
                0.06324582541043715
            ],
            # ----- SEED 4 -----
            [
                0.0036, 0.06540667382331136, 0.06412501038962272, 
                0.06379022269870278
            ],
            # ----- SEED 5 -----
            [
                0.0032, 0.06480597035233136, 0.06413093690251467, 
                0.06374956077356034
            ],
            # ----- SEED 6 -----
            [
                0.0036, 0.06527983441265925, 0.06511190098286956, 
                0.06390081467813193
            ],
            # ----- SEED 7 -----
            [
                0.0036, 0.06471564185477434, 0.06459936237187874, 
                0.06389342442068206
            ],
            # ----- SEED 8 -----
            [
                0.0038, 0.06410976585399608, 0.06407441420355285, 
                0.06377753453185916
            ],
            # ----- SEED 9 -----
            [
                0.0038, 0.06441363390342998, 0.06508900954803606, 
                0.06495195345761075
            ],
        ],

        "PZ3": [
            # ----- SEED 0 -----
            [
                0.0712, 0.20172735858703755, 0.20364842755245358, 
                0.20193868134924722
            ],
            # ----- SEED 1 -----
            [
                0.1020, 0.2103783777338128, 0.2054393775633519, 
                0.20671555986935242
            ],
            # ----- SEED 2 -----
            [
                0.1008, 0.2059415428377529, 0.2010878645149975, 
                0.20170118877891183
            ],
            # ----- SEED 3 -----
            [
                0.1043, 0.20420506344450673, 0.19758700401881235, 
                0.20531387350310368
            ],
            # ----- SEED 4 -----
            [
                0.1061, 0.20505701175011795, 0.20024907586111457, 
                0.20212056919016097
            ],
            # ----- SEED 5 -----
            [
                0.0984, 0.200558674695798, 0.20237562509309814, 
                0.19961135582252007
            ],
            # ----- SEED 6 -----
            [
                0.1017, 0.204255197470482, 0.20385308802017083, 
                0.20178208694167635
            ],
            # ----- SEED 7 -----
            [
                0.1067, 0.21093367490241902, 0.208887129427938, 
                0.20764594403212228
            ],
            # ----- SEED 8 -----
            [
                0.1050, 0.20579762198890938, 0.20380851703412842, 
                0.20813433498048792
            ],
            # ----- SEED 9 -----
            [
                0.1015, 0.2046228176994478, 0.20493644722717924, 
                0.2028070677135662
            ],
        ],

    }
}

# ===========================
# PR (Pixel AUPRC) PER SEED
# SPADE – PZ1..PZ5
# ===========================

results_pr_by_method = {
    
    "FAPM": {

        "PZ1": [
            # ----- SEED 0 -----
            [
                0.0340, 0.2456993318649111, 0.24687840522579282, 
                0.24599828903886226
            ],

            # ----- SEED 1 -----
            [   
                0.0341, 0.2422822082088868, 0.24155315354173476, 
                0.24130648782241998
            ],

            # ----- SEED 2 -----
            [
                0.0336, 0.2420555923285774, 0.24390349038501624, 
                0.24704372785998888
            ],

            # ----- SEED 3 -----
            [
                0.0324, 0.24367777614473762, 0.24762562978070413, 
                0.24521463488741152
            ],

            # ----- SEED 4 -----
            [
                0.0338, 0.24251409710286215, 0.24693961966629058, 
                0.245631958745761
            ],

            # ----- SEED 5 -----
            [
                0.0323, 0.24529054849817733, 0.24333937255399912, 
                0.2435723998385499
            ],

            # ----- SEED 6 -----
            [
                0.0339, 0.2405619233061814, 0.24273183053105674, 
                0.24523017114189455
            ],

            # ----- SEED 7 -----
            [
                0.0341, 0.24493153471255555, 0.2452476371035196, 
                0.24582156644578249
            ],

            # ----- SEED 8 -----
            [
                0.0349, 0.244412783568728, 0.24290844239240855, 
                0.24675356295950673
            ],

            # ----- SEED 9 -----
            [
                0.0348, 0.24773804371633282, 0.24815665337749876, 
                0.24826604753816459
            ],
        ],

            "PZ3": [
            # ----- SEED 0 -----
            [
                0.0674,0.2612442155731325, 0.2612014633839659, 
                0.2589240440014156
            ],

            # ----- SEED 1 -----
            [
                0.0852, 0.2539875041724812, 0.2470150946083372, 
                0.24998575252146485
            ],

            # ----- SEED 2 -----
            [
                0.0840, 0.2572796583455544, 0.2514521366110157, 
                0.2528644945287004
            ],

            # ----- SEED 3 -----
            [
                0.0857, 0.26128391023914754, 0.25010705521331045, 
                0.24711479698679295
            ],

            # ----- SEED 4 -----
            [
                0.0888,0.25248146067135163, 0.2502306376600505, 
                0.24934520873427762
            ],

            # ----- SEED 5 -----
            [
                0.0837, 0.24562108597354332, 0.2480481956801442, 
                0.24728480995290852
            ],

            # ----- SEED 6 -----
            [
                0.0852, 0.24669115262875702, 0.24733261105675197, 
                0.2453957798448853
            ],

            # ----- SEED 7 -----
            [
                0.0839, 0.25766488906950147, 0.25493728156901646, 
                0.2518633343423973
            ],

            # ----- SEED 8 -----
            [
                0.0893, 0.2595520976570394, 0.25897956102545944, 
                0.25353267110040967
            ],

            # ----- SEED 9 -----
            [
                0.0862, 0.2508224626319957, 0.2519351090300205, 
                0.24885281799295533
            ],
        ],

    }
}

# Colori per i pezzi
colors_pieces = {
    "PZ1": "blue",
    "PZ2": "orange",
    "PZ3": "green",
    "PZ4": "red",
    "PZ5": "purple",
}

# =================================================
# FUNZIONE: ROC – Pixel-level AUROC
# GRAFICO SINGOLO PER CIASCUN PEZZO
# =================================================

def plot_method_roc(method_name, pieces_dict):
    print(f"\n=== {method_name} – Pixel-level AUROC (ROC) ===")

    for piece_name, values in pieces_dict.items():
        # FIGURA DEDICATA AL SINGOLO PEZZO
        plt.figure(figsize=(8, 5))

        arr = np.array(values, dtype=float)
        color = colors_pieces.get(piece_name, None)

        if arr.ndim == 1:
            curves = arr.reshape(1, -1)
        elif arr.ndim == 2:
            curves = arr
        else:
            raise ValueError(
                f"{method_name} – {piece_name}: 'values' ha dim = {arr.ndim}, "
                "mi aspetto 1D oppure 2D (num_seeds x num_frazioni)."
            )

        num_seeds, num_frac = curves.shape
        if num_frac != len(good_fractions):
            raise ValueError(
                f"{method_name} – {piece_name}: colonne = {num_frac}, "
                f"ma len(good_fractions) = {len(good_fractions)}"
            )

        mean_y = curves.mean(axis=0)
        var_y  = curves.var(axis=0)
        std_y  = curves.std(axis=0)

        print(f"\n{method_name} – {piece_name}: {num_seeds} seed per ciascuna Good Fraction (ROC)")
        for j, gf in enumerate(good_fractions):
            vals   = curves[:, j]
            mean_j = vals.mean()
            var_j  = vals.var()
            std_j  = vals.std()
            cv_pct = 100.0 * std_j / mean_j if mean_j != 0 else np.nan
            print(
                f"  GF = {gf:.2f} -> mean = {mean_j:.6f}, "
                f"var = {var_j:.6e}, std = {std_j:.6f}, CV = {cv_pct:.2f}%"
            )

        # tutti i punti (tutti i seed) per ogni GF
        for j, gf in enumerate(good_fractions):
            plt.scatter(
                np.full(num_seeds, gf),
                curves[:, j],
                alpha=0.4,
                s=20,
                color=color
            )

        # CURVA MEDIA
        plt.plot(
            good_fractions,
            mean_y,
            marker="o",
            linewidth=2,
            label=f"{piece_name} (media)",
            color=color
        )

        plt.title(f"{method_name} – {piece_name}: Pixel-level AUROC (ROC) vs Good Fraction")
        plt.xlabel("Good Fraction")
        plt.ylabel("Pixel AUROC (ROC)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# =================================================
# FUNZIONE: PRO – PUNTI TUTTI I SEED + CURVA MEDIA
# GRAFICO SINGOLO PER CIASCUN PEZZO
# =================================================

def plot_method_pro(method_name, pieces_dict):
    print(f"\n=== {method_name} – Pixel-level AUC-PRO ===")

    for piece_name, values in pieces_dict.items():
        # FIGURA DEDICATA AL SINGOLO PEZZO
        plt.figure(figsize=(8, 5))

        arr = np.array(values, dtype=float)
        color = colors_pieces.get(piece_name, None)

        if arr.ndim == 1:
            curves = arr.reshape(1, -1)
        elif arr.ndim == 2:
            curves = arr
        else:
            raise ValueError(
                f"{method_name} – {piece_name}: 'values' ha dim = {arr.ndim}, "
                "mi aspetto 1D oppure 2D (num_seeds x num_frazioni)."
            )

        num_seeds, num_frac = curves.shape
        if num_frac != len(good_fractions):
            raise ValueError(
                f"{method_name} – {piece_name}: colonne = {num_frac}, "
                f"ma len(good_fractions) = {len(good_fractions)}"
            )

        mean_y = curves.mean(axis=0)
        var_y  = curves.var(axis=0)
        std_y  = curves.std(axis=0)

        print(f"\n{method_name} – {piece_name}: {num_seeds} seed per ciascuna Good Fraction")
        for j, gf in enumerate(good_fractions):
            vals   = curves[:, j]
            mean_j = vals.mean()
            var_j  = vals.var()
            std_j  = vals.std()
            cv_pct = 100.0 * std_j / mean_j if mean_j != 0 else np.nan
            print(
                f"  GF = {gf:.2f} -> mean = {mean_j:.6f}, "
                f"var = {var_j:.6e}, std = {std_j:.6f}, CV = {cv_pct:.2f}%"
            )

        # tutti i punti (tutti i seed) per ogni GF
        for j, gf in enumerate(good_fractions):
            plt.scatter(
                np.full(num_seeds, gf),
                curves[:, j],
                alpha=0.4,
                s=20,
                color=color
            )

        # CURVA MEDIA
        plt.plot(
            good_fractions,
            mean_y,
            marker="o",
            linewidth=2,
            label=f"{piece_name} (media)",
            color=color
        )

        plt.title(f"{method_name} – {piece_name}: Pixel-level AUC-PRO vs Good Fraction")
        plt.xlabel("Good Fraction")
        plt.ylabel("Pixel AUC-PRO")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# =================================================
# FUNZIONE: PR – AUPRC
# GRAFICO SINGOLO PER CIASCUN PEZZO
# =================================================

def plot_method_pr(method_name, pieces_dict):
    print(f"\n=== {method_name} – Pixel-level AUPRC (PR) ===")

    for piece_name, values in pieces_dict.items():
        # FIGURA DEDICATA AL SINGOLO PEZZO
        plt.figure(figsize=(8, 5))

        arr = np.array(values, dtype=float)
        color = colors_pieces.get(piece_name, None)

        if arr.ndim == 1:
            curves = arr.reshape(1, -1)
        elif arr.ndim == 2:
            curves = arr
        else:
            raise ValueError(
                f"{method_name} – {piece_name}: 'values' ha dim = {arr.ndim}, "
                "mi aspetto 1D oppure 2D (num_seeds x num_frazioni)."
            )

        num_seeds, num_frac = curves.shape
        if num_frac != len(good_fractions):
            raise ValueError(
                f"{method_name} – {piece_name}: colonne = {num_frac}, "
                f"ma len(good_fractions) = {len(good_fractions)}"
            )

        mean_y = curves.mean(axis=0)
        var_y  = curves.var(axis=0)
        std_y  = curves.std(axis=0)

        print(f"\n{method_name} – {piece_name}: {num_seeds} seed per ciascuna Good Fraction (PR)")
        for j, gf in enumerate(good_fractions):
            vals   = curves[:, j]
            mean_j = vals.mean()
            var_j  = vals.var()
            std_j  = vals.std()
            cv_pct = 100.0 * std_j / mean_j if mean_j != 0 else np.nan
            print(
                f"  GF = {gf:.2f} -> mean = {mean_j:.6f}, "
                f"var = {var_j:.6e}, std = {std_j:.6f}, CV = {cv_pct:.2f}%"
            )

        # tutti i punti (tutti i seed) per ogni GF
        for j, gf in enumerate(good_fractions):
            plt.scatter(
                np.full(num_seeds, gf),
                curves[:, j],
                alpha=0.4,
                s=20,
                color=color
            )

        # CURVA MEDIA
        plt.plot(
            good_fractions,
            mean_y,
            marker="o",
            linewidth=2,
            label=f"{piece_name} (media)",
            color=color
        )

        plt.title(f"{method_name} – {piece_name}: Pixel-level AUPRC (PR) vs Good Fraction")
        plt.xlabel("Good Fraction")
        plt.ylabel("Pixel AUPRC (PR)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# =================================================
# LANCIA PER FAPM
# =================================================

plot_method_pro("FAPM", results_pro_by_method["FAPM"])
plot_method_pr("FAPM",  results_pr_by_method["FAPM"])
plot_method_roc("FAPM", results_roc_by_method["FAPM"])


























