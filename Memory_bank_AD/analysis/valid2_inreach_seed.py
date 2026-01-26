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
    "INREACH": {

        "PZ1": [
            # ----- SEED 0 -----
            [   
                0.7861, 0.9597, 0.9613, 
                0.9619
                
            ],

            # ----- SEED 1 -----
           
            [   0.7835, 0.9606, 0.9618, 
             0.9619
            
            ],

            # ----- SEED 2 -----
            [
                0.7912, 0.9605, 0.9621, 
                0.9621
            
            ],

            # ----- SEED 3 -----
            [
                0.7880, 0.9610, 0.9625, 
                0.9623
            ],
            # ----- SEED 4 -----
            [
                0.7954, 0.9609, 0.9617 , 
                0.9622
            ],
            # ----- SEED 5 -----
            [
                0.7844, 0.9587, 0.9617, 
                0.9623
            ],
            # ----- SEED 6 -----
            [
                0.7839, 0.9608, 0.9618, 
                0.9624
            ],
            # ----- SEED 7 -----
            [
                0.7858, 0.9606, 0.9625, 
                0.9625
            ],
            # ----- SEED 8 -----
            [
                0.7891, 0.9606, 0.9611, 
                0.9622
            ],
            # ----- SEED 9 -----
            [
                0.7877, 0.9615, 0.9619, 
                0.9626
            ]
        ],

        "PZ3": [
            # ----- SEED 0 -----
            [
                0.9473, 0.9666, 0.9756, 
                0.9754
            ], 
            # ----- SEED 1 -----
            [
                0.9494, 0.9745, 0.9754, 
             0.9753
            ],
            # ----- SEED 2 -----
            [
                0.9494, 0.9740 , 0.9754, 
                0.9754
            ],
            # ----- SEED 3 -----
            [
               0.9444, 0.9747, 0.9755, 
               0.9751 
            ],
            # ----- SEED 4 -----
            [
                0.9488, 0.9750, 0.9757, 
                0.975
            ],
            # ----- SEED 5 -----
            [
                0.9464, 0.9711, 0.9754, 
                0.9748
            ],
            # ----- SEED 6 -----
            [
                0.9485, 0.9751, 0.9761, 
                0.9760
            ],
            # ----- SEED 7 -----
            [
                0.9486, 0.9760, 0.9756, 
                0.9757
            ],
            # ----- SEED 8 -----
            [
                0.9465, 0.9748, 0.9753, 
                0.9757
            ],
            # ----- SEED 9 -----
            [
                0.9464, 0.9755, 0.9752, 
                0.9755
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

    "INREACH": {

        "PZ1": [
            # ----- SEED 0 -----
            [
               0.0270, 0.0630, 0.0629, 0.0647
            ],
            # ----- SEED 1 -----
            [
                0.0262, 0.0634, 0.0631, 0.0631
            ],
            # ----- SEED 2 -----
            [
                0.0312, 0.0640, 0.0635, 0.0646
            ],
            # ----- SEED 3 -----
            [
                0.0275, 0.0641, 0.0653, 0.0627
            ],
            # ----- SEED 4 -----
            [
                0.0311, 0.0625, 0.0629, 0.0638
            ],
            # ----- SEED 5 -----
            [
               0.0262, 0.0636, 0.0611, 0.0619
            ],
            # ----- SEED 6 -----
            [
                0.0266, 0.0642, 0.0622, 0.0646
            ],
            # ----- SEED 7 -----
            [
                0.0271, 0.0624, 0.0646, 0.0641
            ],
            # ----- SEED 8 -----
            [
                0.0296, 0.0617, 0.0640, 0.0664
            ],
            # ----- SEED 9 -----
            [
                0.0283, 0.0654, 0.0645, 0.0633
            ],
        ],

        "PZ3": [
            # ----- SEED 0 -----
            [
                0.2260, 0.1972, 0.1990, 0.1924
            ],
            # ----- SEED 1 -----
            [
                0.2308, 0.2004, 0.2042, 0.1988
            ],
            # ----- SEED 2 -----
            [
                0.2241, 0.2037, 0.1968, 0.1955
            ],
            # ----- SEED 3 -----
            [
                0.2218, 0.1916, 0.1990, 0.1977
            ],
            # ----- SEED 4 -----
            [
                0.2246, 0.2014, 0.2041, 0.2025
            ],
            # ----- SEED 5 -----
            [
                0.2264, 0.2071, 0.1990, 0.1904
            ],
            # ----- SEED 6 -----
            [
                0.2263, 0.2074, 0.2031, 0.2040
            ],
            # ----- SEED 7 -----
            [
                0.2255, 0.1996, 0.1982, 0.1934
            ],
            # ----- SEED 8 -----
            [
                0.2279, 0.1992, 0.1974, 0.2029
            ],
            # ----- SEED 9 -----
            [
                0.2253, 0.2007, 0.1933, 0.2011
            ],
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
            # ----- SEED 0 -----
            [
                0.0804, 0.2395, 0.2416, 
                0.2444
            ],

            # ----- SEED 1 -----
            [   
                0.0821, 0.2369, 0.2373, 
                0.2366
            ],

            # ----- SEED 2 -----
            [
                0.0873, 0.2356, 0.2403, 
                0.2407
            ],

            # ----- SEED 3 -----
            [
                0.0842, 0.2375, 0.2511, 
                0.2383
            ],

            # ----- SEED 4 -----
            [
                0.0875, 0.2381, 0.2400, 
                0.2423
            ],

            # ----- SEED 5 -----
            [
                0.0818, 0.2338, 0.2376, 
                0.2414
            ],

            # ----- SEED 6 -----
            [
               0.0823, 0.2366, 0.2415, 
               0.2422 
            ],

            # ----- SEED 7 -----
            [
                0.0838, 0.2425, 0.2547, 
                0.2514 
            ],

            # ----- SEED 8 -----
            [
                0.0863, 0.2395, 0.2414, 
                0.2565
            ],

            # ----- SEED 9 -----
            [
                0.0846 , 0.2519, 0.2513, 
                0.2479
            ],
        ],

            "PZ3": [
            # ----- SEED 0 -----
            [
                0.2758, 0.2800, 0.2825, 
                0.2783
            ],

            # ----- SEED 1 -----
            [
                0.2826, 0.2800, 0.2736, 
                0.2778
            ],

            # ----- SEED 2 -----
            [
                0.2800, 0.2862, 0.2784, 
                0.2757
            ],

            # ----- SEED 3 -----
            [
                0.2671, 0.2811, 0.2822, 
                0.2763 
            ],

            # ----- SEED 4 -----
            [
                0.2738 , 0.2767, 0.2798, 
                0.2801
            ],

            # ----- SEED 5 -----
            [
                0.2707, 0.2833, 0.2801, 
                0.2719
            ],

            # ----- SEED 6 -----
            [
                0.2753, 0.2895, 0.2787, 
                0.2773
            ],

            # ----- SEED 7 -----
            [
                0.2722, 0.2803, 0.2780, 
                0.2736 
            ],

            # ----- SEED 8 -----
            [
                0.2747, 0.2806, 0.2821, 
                0.2790
            ],

            # ----- SEED 9 -----
            [
                0.2748, 0.2861, 0.2785, 
                0.2788 
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
# LANCIA PER SPADE
# =================================================

plot_method_pro("INREACH", results_pro_by_method["INREACH"])
plot_method_pr("INREACH",  results_pr_by_method["INREACH"])
plot_method_roc("INREACH", results_roc_by_method["INREACH"])





















