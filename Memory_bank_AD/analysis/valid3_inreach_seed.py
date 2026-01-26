import matplotlib.pyplot as plt
import numpy as np

# ===========================
# ASCISSE COMUNI (% GOOD)
# ===========================

good_fractions = np.array([
     0.20, 
])

# ===========================
# ROC (Pixel AUROC) PER SEED
# INREACH – PZ1..PZ5
# ===========================

results_roc_by_method = {
    "INREACH": {

        
        "PZ3": [
            # ----- SEED 0 -----
            [
                0.8834
            ],
            # ----- SEED 1 -----
            [
                0.8645
            ],
            # ----- SEED 2 -----
            [
                0.8692
            ],
            # ----- SEED 3 -----
            [
                0.8605
            ],
            # ----- SEED 4 -----
            [
                0.8606
            ],
            # ----- SEED 5 -----
            [
                0.8877
            ],
            # ----- SEED 6 -----
            [
                0.8993
            ],
            # ----- SEED 7 -----
            [
                0.8951
            ],
            # ----- SEED 8 -----
            [
                0.8648
            ],
            # ----- SEED 9 -----
            [
                0.8674
            ]
        ],

        "PZ4": [
            # ----- SEED 0 -----
            [
                0.9902
            ],
            # ----- SEED 1 -----
            [
                0.9914
            ],
            # ----- SEED 2 -----
            [
                0.9916
            ],
            # ----- SEED 3 -----
            [
                0.9887
            ],
            # ----- SEED 4 -----
            [
                0.9865
            ],
            # ----- SEED 5 -----
            [
                0.9885
            ],
            # ----- SEED 6 -----
            [
                0.9904
            ],
            # ----- SEED 7 -----
            [
                0.9895
            ],
            # ----- SEED 8 -----
            [
                0.9896
            ],
            # ----- SEED 9 -----
            [
                0.9871
            ]
        ],

        "PZ5": [
            # ----- SEED 0 -----
            [
                0.9951
            ],
            # ----- SEED 1 -----
            [
                0.9934
            ],
            # ----- SEED 2 -----
            [
                0.9952
            ],
            # ----- SEED 3 -----
            [
                0.9947
            ],
            # ----- SEED 4 -----
            [
                0.9938
            ],
            # ----- SEED 5 -----
            [
                0.9960
            ],
            # ----- SEED 6 -----
            [
                0.9956
            ],
            # ----- SEED 7 -----
            [
                0.9941
            ],
            # ----- SEED 8 -----
            [
                0.9939
            ],
            # ----- SEED 9 -----
            [
                0.9947
            ]
        ]
    }
}


# ===========================
# PRO (Pixel AUC-PRO) PER SEED
# FAPM – PZ1..PZ5
# Ogni pezzo: lista di liste [seed0 ... seed9]
# ===========================

results_pro_by_method = {

    "INREACH": {

        "PZ3": [
            # ----- SEED 0 -----
            [
                0.0296
            ],
            # ----- SEED 1 -----
            [
                0.0507
            ],
            # ----- SEED 2 -----
            [
                0.0241
            ],
            # ----- SEED 3 -----
            [
                0.0636
            ],
            # ----- SEED 4 -----
            [
                0.0323
            ],
            # ----- SEED 5 -----
            [
                0.0447
            ],
            # ----- SEED 6 -----
            [
                0.0909
            ],
            # ----- SEED 7 -----
            [
                0.0385
            ],
            # ----- SEED 8 -----
            [
                0.0478
            ],
            # ----- SEED 9 -----
            [
                0.0598
            ],
        ],

        "PZ4": [
            # ----- SEED 0 -----
            [
                0.5384
            ],
            # ----- SEED 1 -----
            [
                0.5311
            ],
            # ----- SEED 2 -----
            [
                0.5188
            ],
            # ----- SEED 3 -----
            [
                0.4790
            ],
            # ----- SEED 4 -----
            [
                0.4115
            ],
            # ----- SEED 5 -----
            [
                0.4159
            ],
            # ----- SEED 6 -----
            [
                0.5222
            ],
            # ----- SEED 7 -----
            [
                0.4970
            ],
            # ----- SEED 8 -----
            [
                0.4636
            ],
            # ----- SEED 9 -----
            [
                0.4417
            ],
        ],

        "PZ5": [
            # ----- SEED 0 -----
            [
                0.4921
            ],
            # ----- SEED 1 -----
            [
                0.4151
            ],

            # ----- SEED 2 -----
            [
                0.5094
            ],
            # ----- SEED 3 -----
            [
                0.4852
            ],
            # ----- SEED 4 -----
            [
                0.4306
            ],
            # ----- SEED 5 -----
            [
                0.5296
            ],
            # ----- SEED 6 -----
            [
                0.5097
            ],
            # ----- SEED 7 -----
            [
               0.4328
            ],
            # ----- SEED 8 -----
            [
                0.4703
            ],
            # ----- SEED 9 -----
            [
                0.4694
            ],
        ],
    }
}

# ===========================
# PR (Pixel AUPRC) PER SEED
# INREACH – PZ1..PZ5
# ===========================

results_pr_by_method = {
    
    "INREACH": {

        
            "PZ3": [
            # ----- SEED 0 -----
            [
                0.0339
            ],

            # ----- SEED 1 -----
            [
                0.0429 
            ],

            # ----- SEED 2 -----
            [
                0.0288
            ],

            # ----- SEED 3 -----
            [
                0.0397
            ],

            # ----- SEED 4 -----
            [
                0.0293
            ],

            # ----- SEED 5 -----
            [
                0.0409
            ],

            # ----- SEED 6 -----
            [
                0.0626
            ],

            # ----- SEED 7 -----
            [
                0.0388
            ],

            # ----- SEED 8 -----
            [
                0.0367
            ],

            # ----- SEED 9 -----
            [
                0.0387 
            ],
        ],

        "PZ4": [
            # ----- SEED 0 -----
            [
                0.2339
            ],

            # ----- SEED 1 -----
            [
                0.2701
            ],

            # ----- SEED 2 -----
            [
                0.2343
            ],

            # ----- SEED 3 -----
            [
                0.2029
            ],

            # ----- SEED 4 -----
            [
                0.1725
            ],

            # ----- SEED 5 -----
            [
                0.2437
            ],

            # ----- SEED 6 -----
            [
                0.2527
            ],
            # ----- SEED 7 -----
            [
                0.2742
            ],
            # ----- SEED 8 -----
            [
                0.2594
            ],
            # ----- SEED 9 -----
            [
                0.1974
            ],
        ],

        "PZ5": [

            # ----- SEED 0 -----
            [
                0.2251
            ],

            # ----- SEED 1 -----
            [
                0.1792
            ],

            # ----- SEED 2 -----
            [
                0.2340
            ],

            # ----- SEED 3 -----
            [
                0.2260 
            ],

            # ----- SEED 4 -----
            [
                0.1755
            ],

            # ----- SEED 5 -----
            [
                0.2268
            ],
            # ----- SEED 6 -----
            [
                0.1845
            ],
            # ----- SEED 7 -----
            [
                0.1938
            ],
            # ----- SEED 8 -----
            [
                0.2114 
            ],
            # ----- SEED 9 -----
            [
                0.2113
            ],
        ],
    }
}
# ============================================================
# NORMALIZZAZIONE "PER PEZZO" (MIN-MAX su seed x good_fraction)
# ============================================================

NORMALIZE_PIECE = True  

def _to_2d_array(seed_lists, method_name, piece_name, metric_name):
    if not isinstance(seed_lists, (list, tuple)) or len(seed_lists) == 0:
        raise ValueError(f"[{metric_name}] {method_name}/{piece_name}: lista seed vuota o non valida.")

    lengths = [len(s) for s in seed_lists]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"[{metric_name}] {method_name}/{piece_name}: seed con lunghezze diverse {lengths}. "
            f"Ogni seed deve avere la stessa lunghezza (= len(good_fractions))."
        )

    arr = np.array(seed_lists, dtype=float)  # shape: (num_seeds, num_gf)
    if arr.ndim != 2:
        raise ValueError(f"[{metric_name}] {method_name}/{piece_name}: atteso array 2D, ottenuto shape {arr.shape}.")
    return arr

def minmax_normalize_piece(seed_lists, method_name, piece_name, metric_name):
    arr = _to_2d_array(seed_lists, method_name, piece_name, metric_name)

    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))

    if not np.isfinite(mn) or not np.isfinite(mx):
        raise ValueError(f"[{metric_name}] {method_name}/{piece_name}: min/max non finiti (NaN/Inf).")

    if np.isclose(mx, mn):
        norm = np.zeros_like(arr, dtype=float)
    else:
        norm = (arr - mn) / (mx - mn)

    return norm.tolist(), mn, mx

def normalize_results_piecewise(results_by_method, metric_name):
    norm = {}
    stats = {}
    for method_name, pieces_dict in results_by_method.items():
        norm[method_name] = {}
        stats[method_name] = {}
        for piece_name, seed_lists in pieces_dict.items():
            norm_seeds, mn, mx = minmax_normalize_piece(seed_lists, method_name, piece_name, metric_name)
            norm[method_name][piece_name] = norm_seeds
            stats[method_name][piece_name] = {"min": mn, "max": mx}
    return norm, stats

# ---- Applica la normalizzazione ai tuoi tre dizionari (ROC / PRO / PR) ----
if NORMALIZE_PIECE:
    results_roc_by_method_norm, roc_piece_minmax = normalize_results_piecewise(results_roc_by_method, "ROC")
    results_pro_by_method_norm, pro_piece_minmax = normalize_results_piecewise(results_pro_by_method, "PRO")
    results_pr_by_method_norm,  pr_piece_minmax  = normalize_results_piecewise(results_pr_by_method,  "PR")
else:
    results_roc_by_method_norm, roc_piece_minmax = results_roc_by_method, {}
    results_pro_by_method_norm, pro_piece_minmax = results_pro_by_method, {}
    results_pr_by_method_norm,  pr_piece_minmax  = results_pr_by_method,  {}



# Colori per i pezzi
colors_pieces = {
    "PZ1": "blue",
    "PZ2": "orange",
    "PZ3": "green",
    "PZ4": "red",
    "PZ5": "purple",
}


def _as_curves(values, method_name, piece_name, good_fractions):
    arr = np.array(values, dtype=float)
    if arr.ndim == 1:
        curves = arr.reshape(1, -1)
    elif arr.ndim == 2:
        curves = arr
    else:
        raise ValueError(f"{method_name} – {piece_name}: values ndim={arr.ndim}, atteso 1D o 2D.")

    if curves.shape[1] != len(good_fractions):
        raise ValueError(
            f"{method_name} – {piece_name}: colonne={curves.shape[1]} ma len(good_fractions)={len(good_fractions)}"
        )
    return curves  # (num_seeds, num_gf)

def plot_metric_all_pieces(method_name, pieces_dict, good_fractions, ylabel, title):
    
    plt.figure(figsize=(8, 5))

    for piece_name, values in pieces_dict.items():
        color = colors_pieces.get(piece_name, None)
        curves = _as_curves(values, method_name, piece_name, good_fractions)
        num_seeds, _ = curves.shape

        # scatter di tutti i seed
        for j, gf in enumerate(good_fractions):
            plt.scatter(
                np.full(num_seeds, gf),
                curves[:, j],
                alpha=0.4,
                s=20,
                color=color
            )

        # curva media
        mean_y = curves.mean(axis=0)
        plt.plot(
            good_fractions,
            mean_y,
            marker="o",
            linewidth=2,
            label=f"{piece_name} (media)",
            color=color
        )

    plt.title(title)
    plt.xlabel("Good Fraction")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # LEGENDA FUORI A DESTRA (così non copre i grafici)
    #plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    #plt.tight_layout(rect=[0, 0, 0.82, 1])  # lascia spazio alla legenda

    plt.show()

plot_metric_all_pieces(
    "INREACH",
    results_pro_by_method_norm["INREACH"],
    good_fractions,
    ylabel="Pixel AUC-PRO",
    title="INREACH: Pixel-level AUC-PRO vs Good Fraction"
)

plot_metric_all_pieces(
    "INREACH",
    results_pr_by_method_norm["INREACH"],
    good_fractions,
    ylabel="Pixel AUPRC (PR)",
    title="INREACH: Pixel-level AUPRC (PR) vs Good Fraction"
)

plot_metric_all_pieces(
    "INREACH",
    results_roc_by_method_norm["INREACH"],
    good_fractions,
    ylabel="Pixel AUROC (ROC)",
    title="INREACH: Pixel-level AUROC (ROC) vs Good Fraction"
)



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
# LANCIA PER INREACH
# =================================================

plot_method_pro("INREACH", results_pro_by_method_norm["INREACH"])
plot_method_pr("INREACH",  results_pr_by_method_norm["INREACH"])
plot_method_roc("INREACH", results_roc_by_method_norm["INREACH"])























