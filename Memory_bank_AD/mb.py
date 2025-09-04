import os
from pathlib import Path
from typing import List
import random

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import models
from sklearn.neighbors import NearestNeighbors

from dataio import list_images_by_part
from transforms_util import build_resnet_preprocess

# --- Feature extractor semplice (ResNet18, layer3) ---
class SimpleResNetExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = torch.nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        feats = self.features(x)  # (B, 256, 14, 14)
        B, C, H, W = feats.shape
        return feats.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

# --- Funzione per estrarre feature da una lista di immagini ---
def extract_features(paths: List[Path], extractor, preprocess, device) -> np.ndarray:
    if len(paths) == 0:
        print("Nessuna immagine trovata per estrarre le feature.")
        return np.empty((0, 256))
    all_feats = []
    for p in tqdm(paths, desc="Extracting features"):
        img = Image.open(p).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        feats = extractor(x)[0].cpu().numpy()  # (H*W, C)
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)  # (Npatch_tot, C)

# --- Funzione per calcolare anomaly score su immagini ---
def anomaly_scores_with_map(paths: List[Path], extractor, preprocess, knn, device) -> tuple[np.ndarray, list[np.ndarray]]:
    scores = []
    maps = []
    for p in tqdm(paths, desc="Scoring"):
        img = Image.open(p).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
        feats = extractor(x)[0].cpu().numpy()  # (H*W, C)
        dists, _ = knn.kneighbors(feats)
        patch_scores = dists[:, 0]  # (H*W,)
        scores.append(np.max(patch_scores))  # score globale = max patch
        map_ = patch_scores.reshape(14, 14)
        maps.append(map_)
    return np.array(scores), maps

def main():
    # --- Configurazione ---
    DATASET_ROOT = os.path.join(os.path.dirname(__file__))  # Memory_bank_AD come root
    PART = "PZ1"  # Nome del pezzo da analizzare
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PREPROCESS = build_resnet_preprocess(target_size=224, mode="letterbox")

    # --- Assicurati che la cartella esista e abbia la struttura corretta ---
    part_dir = os.path.join(DATASET_ROOT, PART)
    pos_dir = os.path.join(part_dir, "pos1")
    rgb_dir = os.path.join(pos_dir, "rgb")
    fault_rgb_dir = os.path.join(pos_dir, "fault", "rgb")

    if not os.path.exists(rgb_dir):
        print(f"Creo la struttura {rgb_dir}")
        os.makedirs(rgb_dir, exist_ok=True)
    if not os.path.exists(fault_rgb_dir):
        print(f"Creo la struttura {fault_rgb_dir}")
        os.makedirs(fault_rgb_dir, exist_ok=True)

    # --- Carica immagini buone e difettose ---
    good_paths, defect_paths = list_images_by_part(DATASET_ROOT, PART)
    print(f"Trovate {len(good_paths)} immagini buone e {len(defect_paths)} difettose per {PART}")

    if len(good_paths) == 0:
        print("Errore: nessuna immagine buona trovata. Inserisci immagini in data/PZ1/good.")
        return
    if len(defect_paths) == 0:
        print("Attenzione: nessuna immagine difettosa trovata. Inserisci immagini in data/PZ1/defect.")

    # --- Suddividi immagini buone in train/test ---
    random.seed(42)
    n_train = int(0.7 * len(good_paths))  # 70% train, 30% test
    good_paths_shuffled = good_paths.copy()
    random.shuffle(good_paths_shuffled)
    train_good = good_paths_shuffled[:n_train]
    test_good = good_paths_shuffled[n_train:]

    print(f"Suddivisione: {len(train_good)} train, {len(test_good)} test buone")

    # --- Costruisci il memory bank dalle immagini buone di train ---
    extractor = SimpleResNetExtractor().to(DEVICE)
    memory_bank = extract_features(train_good, extractor, PREPROCESS, DEVICE)

    # --- Costruisci kNN ---
    knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    knn.fit(memory_bank)

    # --- Calcola anomaly scores e mappe ---
    test_good_scores, test_good_maps = anomaly_scores_with_map(test_good, extractor, PREPROCESS, knn, DEVICE)
    defect_scores, defect_maps = anomaly_scores_with_map(defect_paths, extractor, PREPROCESS, knn, DEVICE)

    # --- Stampa statistiche ---
    print(f"Score immagini buone (test): min={test_good_scores.min():.3f}  med={np.median(test_good_scores):.3f}  max={test_good_scores.max():.3f}")
    print(f"Score immagini difettose: min={defect_scores.min():.3f}  med={np.median(defect_scores):.3f}  max={defect_scores.max():.3f}")

    # --- Threshold per segnalare anomalie ---
    threshold = np.percentile(test_good_scores, 99)
    print(f"\nRisultati classificazione immagini test (buone e difettose):")
    for p, s, amap in zip(test_good, test_good_scores, test_good_maps):
        if s > threshold:
            print(f"  {p.name} (score={s:.3f}) --> DIFETTOSA")
        else:
            print(f"  {p.name} (score={s:.3f}) --> BUONA")
    for p, s, amap in zip(defect_paths, defect_scores, defect_maps):
        if s > threshold:
            print(f"  {p.name} (score={s:.3f}) --> DIFETTOSA")
            plt.figure()
            plt.title(f"Anomaly map: {p.name}")
            plt.imshow(amap, cmap="jet")
            plt.colorbar()
            plt.show()
        else:
            print(f"  {p.name} (score={s:.3f}) --> BUONA")

if __name__ == "__main__":
    main()