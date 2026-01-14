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
    "SPADE": {

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
               0.97988020398774, 0.9814236466874218, 0.9820999932757786
            ],
            # ----- SEED 4 -----
            [
                0.9800392017645385, 0.9812596663305307, 0.9816812609963655
            ],
            # ----- SEED 5 -----
            [
                0.9808090358263576, 0.981592586389998, 0.981800727678396
            ],
            # ----- SEED 6 -----
            [
                0.9804437901154305, 0.9812420651036783, 0.9819570768129783
            ],
            # ----- SEED 7 -----
            [
                0.980605773485807, 0.9811656015946688, 0.9814194022271161
            ],
            # ----- SEED 8 -----
            [
                0.980316617046395, 0.9811461580613604, 0.9817660661574876
            ],
            # ----- SEED 9 -----
            [
                0.9807041059102107, 0.9813561720553355, 0.981816040621594
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
                00.31092392167792815, 0.2989773959532974, 0.2981812257804848
            ],
            # ----- SEED 1 -----
            [
                0.30298542869678424, 0.30437126407899173, 0.3018676283403777
            ],
            # ----- SEED 2 -----
            [
                0.30535339943612916, 0.29766751277488024, 0.2962367777176224
            ],
            # ----- SEED 3 -----
            [
                0.31177649633253635, 0.3066265772878437, 0.30138164883282015
            ],
            # ----- SEED 4 -----
            [
                0.30726231245761065, 0.2965099831183801, 0.29314295666739437
            ],
            # ----- SEED 5 -----
            [
                0.30111812550593053, 0.298933752535023, 0.29735690940233833
            ],
            # ----- SEED 6 -----
            [
                0.32091250189961845, 0.3178332997903123, 0.3110348088264494
            ],
            # ----- SEED 7 -----
            [
                0.30224218330755986, 0.2954100922384716, 0.29231519122966165
            ],
            # ----- SEED 8 -----
            [
                0.30544725666818223, 0.3005351910741462, 0.2958490932234812
            ],
            # ----- SEED 9 -----
            [
                0.2934255432677223, 0.28989020424405465, 0.2914917841543684
            ],
        ],

    }
}

# ===========================
# PR (Pixel AUPRC) PER SEED
# SPADE – PZ1, PZ3
# ===========================

results_pr_by_method = {
    
    "PADIM": {

        "PZ1": [
            # ----- SEED 0 -----
            [
                0.2963191310250656, 0.30221583513514577, 0.3024191990494133
            ],

            # ----- SEED 1 -----
            [   
                0.2975485924546018, 0.29897046467521876, 0.2994156844024647
            ],

            # ----- SEED 2 -----
            [
                0.2967166597414679, 0.3019284217284208, 0.3048965314097624
            ],

            # ----- SEED 3 -----
            [
                0.2979239444856626, 0.3032579798593711, 0.30248353776671466
            ],

            # ----- SEED 4 -----
            [
                0.2994356347973901, 0.3042899085685058, 0.3037324418071304
            ],

            # ----- SEED 5 -----
            [
                0.29833510375648753, 0.29896063059541894, 0.29917046533360425
            ],

            # ----- SEED 6 -----
            [
               0.29587266343392876, 0.29772870055761746, 0.301228634402572
            ],

            # ----- SEED 7 -----
            [
                0.2976860797012864, 0.30111132974145777, 0.3041621327024994
            ],

            # ----- SEED 8 -----
            [
                0.29768368952683777, 0.2983405067849142, 0.30188137024908623
            ],

            # ----- SEED 9 -----
            [
                0.30091536468990226, 0.3040623352928969, 0.30441742878119693
            ],
        ],

            "PZ3": [
            # ----- SEED 0 -----
            [
                0.34112544972755643, 0.342186579995613, 0.3436827692760518
            ],

            # ----- SEED 1 -----
            [
                0.34859258185827474, 0.34540145259576227, 0.34391871325780643
            ],

            # ----- SEED 2 -----
            [
                0.3378208137047476, 0.3394253935422407, 0.3395756194889191
            ],

            # ----- SEED 3 -----
            [
                0.34790073550989015, 0.3487022290790994, 0.3425613281368791
            ],

            # ----- SEED 4 -----
            [
                0.3435191275237631, 0.3385777851087358, 0.3366331509681729
            ],

            # ----- SEED 5 -----
            [
                0.34579116320999587, 0.3451095027255717, 0.344913563666184
            ],

            # ----- SEED 6 -----
            [
                0.34793574751305034, 0.3466836211891252, 0.3456325389845454
            ],

            # ----- SEED 7 -----
            [
                0.34670798582555173, 0.344520758865943, 0.3426650671306211
            ],

            # ----- SEED 8 -----
            [
                0.3459559849713917, 0.3457233843606626, 0.34435487656651886
            ],

            # ----- SEED 9 -----
            [
                0.3426754383782351, 0.34257204434953004, 0.3384321655892755
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


























