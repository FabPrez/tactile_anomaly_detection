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
                0.9562, 0.9764379293698949, 0.9746092223775805, 
                0.9743986183162644
                
            ],

            # ----- SEED 1 -----
           
            [   0.9562, 0.9753453643256644, 0.9744819732019336, 
             0.9744898599710651
            
            ],

            # ----- SEED 2 -----
            [
                0.9562, 0.9757466312780938, 0.9758655609125192, 
                0.9751176943115992
            
            ],

            # ----- SEED 3 -----
            [
                0.9562, 0.9751986261485392, 0.9763286592949088, 
                0.9762875114371196
            ],
            # ----- SEED 4 -----
            [
                0.9562, 0.9768257128234056, 0.9751861306469718, 
                0.9733946374980187
            ],
            # ----- SEED 5 -----
            [
                0.9562, 0.9749912893008035, 0.975323103315784, 
                0.9754003891736901
            ],
            # ----- SEED 6 -----
            [
                0.9562, 0.9755392527393044, 0.9746995474229246, 
                0.974297344181162
            ],
            # ----- SEED 7 -----
            [
                0.9562, 0.9747667091907849, 0.9758814260006098, 
                0.9744441826504899
            ],
            # ----- SEED 8 -----
            [
                0.9562, 0.9745433675321916, 0.9748916793163351, 
                0.9768827943519941
            ],
            # ----- SEED 9 -----
            [
                0.9562, 0.9767531298623053, 0.976714813213234, 
                0.9744467466693404
            ]
        ],

        "PZ3": [
            # ----- SEED 0 -----
            [
                0.9631, 0.9801378526453456, 0.9802495833373293, 
                0.9799945683877391
            ], 
            # ----- SEED 1 -----
            [
                0.9631, 0.9814109831454922, 0.9804301033894431, 
             0.9806290822679737
            ],
            # ----- SEED 2 -----
            [
                0.9631, 0.9821174704500262, 0.981184419206002, 
                0.9811407844952523
            ],
            # ----- SEED 3 -----
            [
               0.9631, 0.9819432779447198, 0.9808725099810146, 
                0.9809285464470283
            ],
            # ----- SEED 4 -----
            [
                0.9631, 0.9829532814657571, 0.9812033977981454, 
                0.9808413727031524
            ],
            # ----- SEED 5 -----
            [
                0.9631, 0.9813648441358966, 0.9810185646624561, 
                0.9810033399853216
            ],
            # ----- SEED 6 -----
            [
                0.9631, 0.9806646500788777, 0.9810652248633158, 
                0.9807883833509842
            ],
            # ----- SEED 7 -----
            [
                0.9631, 0.9826724577734777, 0.9812386416664871, 
                0.9813414654592019
            ],
            # ----- SEED 8 -----
            [
                0.9631, 0.9829000730930437, 0.9809355231378762, 
                0.9810066981008122
            ],
            # ----- SEED 9 -----
            [
                0.9631, 0.9828158512667403, 0.9807654802360162, 
                0.9809456115731005
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

    "SPADE": {

        "PZ1": [
            # ----- SEED 0 -----
            [
                0.1286, 0.13314642028882734, 0.13264384537341165, 0.13209054304661616
            ],
            # ----- SEED 1 -----
            [
                0.1286, 0.1281987722635128, 0.13328647950324687, 0.13066387941883362
            ],
            # ----- SEED 2 -----
            [
                0.1286, 0.13032460854967584, 0.12854105716833755, 0.1320437858270342
            ],
            # ----- SEED 3 -----
            [
                0.1286, 0.12845775319585886, 0.12679940295305137, 0.12634418936229003
            ],
            # ----- SEED 4 -----
            [
                0.1286, 0.13281695540093208, 0.13272346507130456, 0.13218288014327412
            ],
            # ----- SEED 5 -----
            [
               0.1286, 0.12550749497025182, 0.12638167075920953, 0.13275522286780644
            ],
            # ----- SEED 6 -----
            [
                0.1286, 0.12823335137820024, 0.1302738342052202, 0.13102248605653255
            ],
            # ----- SEED 7 -----
            [
                0.1286, 0.13145526905713428, 0.12663063331787014, 0.1316616451209686
            ],
            # ----- SEED 8 -----
            [
                0.1286, 0.13210237195002272, 0.13098004928827633, 0.13143506249015297
            ],
            # ----- SEED 9 -----
            [
                0.1286, 0.13151975552215622, 0.13164714979046496, 0.13007779885768547
            ],
        ],

        "PZ3": [
            # ----- SEED 0 -----
            [
                0.3397, 0.3139419784484173, 0.3131288581489242, 0.3224110595986001
            ],
            # ----- SEED 1 -----
            [
                0.3397, 0.33566074307873583, 0.3263835772597728, 0.33007675879640114
            ],
            # ----- SEED 2 -----
            [
                0.3397, 0.3334164249946853, 0.3250366250401734, 0.3303185501000884
            ],
            # ----- SEED 3 -----
            [
                0.3397, 0.3355496672750007, 0.3345791162297158, 0.33601234280559267
            ],
            # ----- SEED 4 -----
            [
                0.3397, 0.3285070242140458, 0.3338182370388329, 0.3352449970217749
            ],
            # ----- SEED 5 -----
            [
                0.3397, 0.3172872403808572, 0.3280383729895433, 0.33365379319197824
            ],
            # ----- SEED 6 -----
            [
                0.3397, 0.3244201912487703, 0.3365375928214726, 0.33273825100197724
            ],
            # ----- SEED 7 -----
            [
                0.3397, 0.32727544539170267, 0.3327102406177497, 0.3284097076048122
            ],
            # ----- SEED 8 -----
            [
                0.3397, 0.34474242680925415, 0.3329374588277883, 0.33503966523334033
            ],
            # ----- SEED 9 -----
            [
                0.3397, 0.33476380829662444, 0.3294543181417679, 0.33045253889444826
            ],
        ],

    }
}

# ===========================
# PR (Pixel AUPRC) PER SEED
# SPADE – PZ1, PZ3
# ===========================

results_pr_by_method = {
    
    "SPADE": {

        "PZ1": [
            # ----- SEED 0 -----
            [
                0.3529, 0.39057668995263345, 0.38671060630891274, 
                0.38584603640372184
            ],

            # ----- SEED 1 -----
            [   
                0.3529, 0.37509290189101985, 0.3825817685744042, 
                0.3830281427587109
            ],

            # ----- SEED 2 -----
            [
                0.3529, 0.37632566211723834, 0.3757430421889734, 
                0.3845480324482221
            ],

            # ----- SEED 3 -----
            [
                0.3529, 0.3698889046720422, 0.375778353888348, 
                0.37571295984133085
            ],

            # ----- SEED 4 -----
            [
                0.3529, 0.3900596136767318, 0.38740953760416624, 
                0.38217934525584357
            ],

            # ----- SEED 5 -----
            [
                0.3529, 0.37023631360214193, 0.36981876736142566, 
                0.3847722333184367
            ],

            # ----- SEED 6 -----
            [
               0.3529, 0.3747109392424359, 0.381669442083192, 
               0.3824209112111736
            ],

            # ----- SEED 7 -----
            [
                0.3529, 0.36863828410092, 0.374475118254481, 
                0.38422662290355064
            ],

            # ----- SEED 8 -----
            [
                0.3529, 0.3706501905001246, 0.3705999581315806, 
                0.3834611262494277
            ],

            # ----- SEED 9 -----
            [
                0.3529, 0.3896136939449119, 0.38860196964337296, 
                0.3828718368841508
            ],
        ],

            "PZ3": [
            # ----- SEED 0 -----
            [
                0.3712, 0.3838132857518978, 0.3866627339690747, 
                0.3902311567496692
            ],

            # ----- SEED 1 -----
            [
                0.3712, 0.4016398700374556, 0.3966352797597319, 
                0.3958229986048871
            ],

            # ----- SEED 2 -----
            [
                0.3712, 00.40128650428851714, 0.3916782244377324, 
                0.3941192370083071
            ],

            # ----- SEED 3 -----
            [
                0.3712, 0.3965621400090789, 0.3966855753834512, 
                0.3970953809187594
            ],

            # ----- SEED 4 -----
            [
                0.3712, 0.40462482592479565, 0.3977729863742517, 
                0.39589658509734604
            ],

            # ----- SEED 5 -----
            [
                0.3712, 0.3937213829458964, 0.3951198748925067, 
                0.3978964878708776
            ],

            # ----- SEED 6 -----
            [
                0.3712, 0.39240152914675697, 0.39928207773128743, 
                0.3998557238228939
            ],

            # ----- SEED 7 -----
            [
                0.3712, 0.39695609609153126, 0.39759720610002336, 
                0.3958287416621399
            ],

            # ----- SEED 8 -----
            [
                0.3712, 0.4110305799402664, 0.3982279940342357, 
                0.4026082126400949
            ],

            # ----- SEED 9 -----
            [
                0.3712, 0.40809088107715347, 0.39679907955199534, 
                0.3962428462605258
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

plot_method_pro("SPADE", results_pro_by_method["SPADE"])
plot_method_pr("SPADE",  results_pr_by_method["SPADE"])
plot_method_roc("SPADE", results_roc_by_method["SPADE"])
























