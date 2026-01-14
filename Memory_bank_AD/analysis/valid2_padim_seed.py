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
# PADIM – PZ1, PZ3
# ===========================

results_roc_by_method = {
    "PADIM": {

        "PZ1": [
            # ----- SEED 0 -----
            [   
                0.9674132264168334, 0.9704851260615408, 0.9711197786622975
                
            ],

            # ----- SEED 1 -----
           
            [   0.9677367345978742, 0.9694584287583834, 0.9706876426603802
            
            ],

            # ----- SEED 2 -----
            [
                0.9676314934806013, 0.9705305412361362, 0.9713544282293248
            
            ],

            # ----- SEED 3 -----
            [
                0.9678969836666941, 0.9710025168824411, 0.9714604137448778
            ],
            # ----- SEED 4 -----
            [
                0.9682409035756087, 0.9708535140379249, 0.9715899656730799
            ],
            # ----- SEED 5 -----
            [
                0.9686972705877266, 0.9706564583297904, 0.9709777464050428
            ],
            # ----- SEED 6 -----
            [
                0.9681047082483079, 0.9704166855020485, 0.9714139841660551
            ],
            # ----- SEED 7 -----
            [
                0.9668422207296388, 0.9697280677272011, 0.9713874212402603
            ],
            # ----- SEED 8 -----
            [
                0.9675567825953881, 0.9697816355889295, 0.9708893778597214
            ],
            # ----- SEED 9 -----
            [
                0.9675393235489497, 0.9702173312055618, 0.9709704529817419
            ]
        ],

        "PZ3": [
            # ----- SEED 0 -----
            [
                0.9780718538251473, 0.9800715836414127, 0.980644218604991
            ], 
            # ----- SEED 1 -----
            [
                0.9811196822535898, 0.9820355052616824, 0.9822922834158337
            ],
            # ----- SEED 2 -----
            [
                0.9794655424875806, 0.9813618041188822, 0.9816765781308714
            ],
            # ----- SEED 3 -----
            [
               0.9798546811054011, 0.9813854842380996, 0.9820497571977991
            ],
            # ----- SEED 4 -----
            [
                0.9755, 0.9746169804561842, 0.9748760211849484, 
                0.9753500864184387
            ],
            # ----- SEED 5 -----
            [
                0.9755, 0.974219172374373, 0.9746315940900036, 
                0.9747485754309835
            ],
            # ----- SEED 6 -----
            [
                0.9756, 0.9746604021077266, 0.9752692533103949, 
                0.975006471385377
            ],
            # ----- SEED 7 -----
            [
                0.9746214786004256, 0.974682640656932, 0.9747256696500755
            ],
            # ----- SEED 8 -----
            [
                0.974233146349221, 0.9748192971532689, 0.974900575534999
            ],
            # ----- SEED 9 -----
            [
                0.9746288204756802, 0.9742835857915194, 0.9755514543000826
            ]
        ],

    }
}


# ===========================
# PRO (Pixel AUC-PRO) PER SEED
# PADIM – PZ1, PZ3
# Ogni pezzo: lista di liste [seed0 ... seed9]
# ===========================

results_pro_by_method = {

    "PADIM": {

        "PZ1": [
            # ----- SEED 0 -----
            [
                0.1017706609250568, 0.10286799226702029, 0.10215865589244658
            ],
            # ----- SEED 1 -----
            [
                0.10214759152539822, 0.10201336825996588, 0.10121675395237723
            ],
            # ----- SEED 2 -----
            [
                0.10073509130660324, 0.10121909216478765, 0.10194950266587605
            ],
            # ----- SEED 3 -----
            [
                0.10297782872646027, 0.1034862161211686, 0.10337961020952996
            ],
            # ----- SEED 4 -----
            [
                0.10246019457349742, 0.10264697647836364, 0.10244079504633341
            ],
            # ----- SEED 5 -----
            [
                0.10166511230027395, 0.10075837280227642, 0.10050126880862037
            ],
            # ----- SEED 6 -----
            [
                0.10093469659773213, 0.10069739788851549, 0.1011340638352707
            ],
            # ----- SEED 7 -----
            [
                0.101824115010019, 0.1013980462043091, 0.1025536474894875
            ],
            # ----- SEED 8 -----
            [
                0.10193871791877383, 0.10087630012133562, 0.10113062046577811
            ],
            # ----- SEED 9 -----
            [
                0.1029284663110058, 0.10265688315335987, 0.10210014138959636
            ],
        ],

        "PZ3": [
            # ----- SEED 0 -----
            [
                0.2042, 0.20172735858703755, 0.20364842755245358, 
                0.20193868134924722
            ],
            # ----- SEED 1 -----
            [
                0.2030, 0.2103783777338128, 0.2054393775633519, 
                0.20671555986935242
            ],
            # ----- SEED 2 -----
            [
                0.2000, 0.2059415428377529, 0.2010878645149975, 
                0.20170118877891183
            ],
            # ----- SEED 3 -----
            [
                0.2003, 0.20420506344450673, 0.19758700401881235, 
                0.20531387350310368
            ],
            # ----- SEED 4 -----
            [
                0.2022, 0.200558674695798, 0.20237562509309814, 
                0.19961135582252007
            ],
            # ----- SEED 5 -----
            [
                0.2003, 0.200558674695798, 0.20237562509309814, 
                0.19961135582252007
            ],
            # ----- SEED 6 -----
            [
                0.2018, 0.204255197470482, 0.20385308802017083, 
                0.20178208694167635
            ],
            # ----- SEED 7 -----
            [
                0.21093367490241902, 0.208887129427938, 0.20764594403212228
            ],
            # ----- SEED 8 -----
            [
                0.20579762198890938, 0.20380851703412842, 0.20813433498048792
            ],
            # ----- SEED 9 -----
            [
                0.2046228176994478, 0.20493644722717924, 0.2028070677135662
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
                0.2464, 0.2456993318649111, 0.24687840522579282, 
                0.24599828903886226
            ],

            # ----- SEED 1 -----
            [   
                0.2464, 0.2422822082088868, 0.24155315354173476, 
                0.24130648782241998
            ],

            # ----- SEED 2 -----
            [
                0.2420555923285774, 0.24390349038501624, 0.24704372785998888
            ],

            # ----- SEED 3 -----
            [
                0.24367777614473762, 0.24762562978070413, 0.24521463488741152
            ],

            # ----- SEED 4 -----
            [
                0.24251409710286215, 0.24693961966629058, 0.245631958745761
            ],

            # ----- SEED 5 -----
            [
                0.24529054849817733, 0.24333937255399912, 0.2435723998385499
            ],

            # ----- SEED 6 -----
            [
                0.2405619233061814, 0.24273183053105674, 0.24523017114189455
            ],

            # ----- SEED 7 -----
            [
                0.24493153471255555, 0.2452476371035196, 0.24582156644578249
            ],

            # ----- SEED 8 -----
            [
                0.244412783568728, 0.24290844239240855, 0.24675356295950673
            ],

            # ----- SEED 9 -----
            [
                0.24773804371633282, 0.24815665337749876, 0.24826604753816459
            ],
        ],

            "PZ3": [
            # ----- SEED 0 -----
            [
                0.2504,0.2612442155731325, 0.2612014633839659, 
                0.2589240440014156
            ],

            # ----- SEED 1 -----
            [
                0.2499, 0.2539875041724812, 0.2470150946083372, 
                0.24998575252146485
            ],

            # ----- SEED 2 -----
            [
                0.2485, 0.2572796583455544, 0.2514521366110157, 
                0.2528644945287004
            ],

            # ----- SEED 3 -----
            [
                0.2483, 0.26128391023914754, 0.25010705521331045, 
                0.24711479698679295
            ],

            # ----- SEED 4 -----
            [
                0.2495,0.25248146067135163, 0.2502306376600505, 
                0.24934520873427762
            ],

            # ----- SEED 5 -----
            [
                0.2487, 0.24562108597354332, 0.2480481956801442, 
                0.24728480995290852
            ],

            # ----- SEED 6 -----
            [
                0.2499, 0.24669115262875702, 0.24733261105675197, 
                0.2453957798448853
            ],

            # ----- SEED 7 -----
            [
                0.25766488906950147, 0.25493728156901646, 0.2518633343423973
            ],

            # ----- SEED 8 -----
            [
                0.2595520976570394, 0.25897956102545944, 0.25353267110040967
            ],

            # ----- SEED 9 -----
            [
                0.2508224626319957, 0.2519351090300205, 0.24885281799295533
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


























