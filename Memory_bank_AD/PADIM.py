import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, DataLoader
from torchvision.models import wide_resnet50_2, resnet18
# import datasets.mvtec as mvtec
import torchvision.transforms.v2 as transforms

from view_utils import show_dataset_images, show_validation_grid_from_loader, show_heatmaps_from_loader
from spade import MyDataset, eval_pixel_metrics
from data_loader import get_items, load_split_pickle, save_split_pickle

# ----------------- CONFIG -----------------
METHOD = "PADIM"
CODICE_PEZZO = "PZ1"
POSITION    = "pos1"    # oppure "all"
TOP_K       = 7
IMG_SIZE    = 224       # input ResNet
SEED        = 42

VIS_VALID_DATASET = False
VIS_PREDICTION_ON_VALID_DATASET = False
GAUSSIAN_SIGMA = 2      # sigma per filtro gaussiano
# ------------------------------------------



# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')



def main():

    # args = parse_args()

    # load model
    # if args.arch == 'resnet18':
    #     model = resnet18(pretrained=True, progress=True)
    #     t_d = 448
    #     d = 100
    # elif args.arch == 'wide_resnet50_2':
    
    # padim selezione wide_resnet50 o resnet_18. Per il momento teniamo la wide come spade
    model = wide_resnet50_2(pretrained=True, progress=True)
    t_d = 1792
    d = 550
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    # os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    # for class_name in mvtec.CLASS_NAMES:

    # load our data
    pre_processing = transforms.Compose([transforms.CenterCrop(IMG_SIZE)])

    # ---- carico immagini + (per fault) maschere ----
    good_imgs_pil = get_items(CODICE_PEZZO, "rgb", label="good", positions=POSITION, return_type="pil")
    
    fault_imgs_pil, fault_masks_np = get_items(
        CODICE_PEZZO, "rgb", label="fault", positions=POSITION,
        return_type="pil",
        with_masks=True, mask_return_type="numpy", mask_binarize=True,  # binarie 0/1
        mask_align="order" 
    )
    
    # ---- dataset ----
    good_ds  = MyDataset(good_imgs_pil,  label_type='good',  transform=pre_processing, masks=None)
    fault_ds = MyDataset(fault_imgs_pil, label_type='fault', transform=pre_processing, masks=fault_masks_np)
    
    # split: train (solo good), val (k good + tutte fault)
    n_good, n_fault = len(good_ds), len(fault_ds)
    k = min(n_fault, n_good)

    g = torch.Generator().manual_seed(SEED)
    perm_good  = torch.randperm(n_good,  generator=g) # ordine casuale degli indici delle immagini good
    perm_fault = torch.randperm(n_fault, generator=g) # ordine casuale degli indici delle immagini fault

    val_good_idx   = perm_good[:k].tolist()
    train_good_idx = perm_good[k:].tolist()
    val_fault_idx  = perm_fault[:k].tolist()  # tutte le fault

    train_set     = Subset(good_ds,  train_good_idx)
    val_good_set  = Subset(good_ds,  val_good_idx)
    val_fault_set = Subset(fault_ds, val_fault_idx)
    val_set       = ConcatDataset([val_good_set, val_fault_set])
    
    if VIS_VALID_DATASET:
        show_dataset_images(val_set, batch_size=5, show_mask=True)
    
    # show_dataset_images(val_set, batch_size=5, show_mask=True)
    print(f"Train: {len(train_set)} good")
    print(f"Val:   {len(val_good_set)} good + {len(val_fault_set)} fault = {len(val_set)}")

    # DataLoader: su Windows evita multiprocess per stabilità (num_workers=0)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin)
    
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # extract train set features
    try:
        train_outputs = load_split_pickle(CODICE_PEZZO, POSITION, split="train", method=METHOD)
        print(f"[cache] Train features caricate da pickle ({METHOD}).")
    except FileNotFoundError:
        print(f"[cache] Nessun pickle train ({METHOD}): estraggo feature...")

        for (x, _, _) in tqdm(train_loader, '| feature extraction | train | %s |' % CODICE_PEZZO):
            with torch.no_grad():
                _ = model(x.to(device))
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)
        print("i'm here!")
        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            print("i'm hereeeeeeeee!")
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
        print("i'm here!")
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        for i in range(H * W):
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I

        # salva lo stesso payload che usavi (lista [mean, cov])
        train_outputs = [mean, cov]
        save_split_pickle(train_outputs, CODICE_PEZZO, POSITION, split="train", method=METHOD)


    gt_list = []
    gt_mask_list = []
    test_imgs = []

    # extract test set features ------------
    try:
        pack = load_split_pickle(CODICE_PEZZO, POSITION, split="validation", method=METHOD)
        test_outputs = pack['features']
        gt_list      = list(pack.get('labels', []))
        gt_mask_list = list(pack.get('masks', []))
        test_imgs    = list(pack.get('images', []))
        print(f"[cache] Validation features caricate da pickle ({METHOD}).")
    except FileNotFoundError:
        print(f"[cache] Nessun pickle validation ({METHOD}): estraggo feature...")

        for (x, y, mask) in tqdm(val_loader, '| feature extraction | test | %s |' % CODICE_PEZZO):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            with torch.no_grad():
                _ = model(x.to(device))
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        # salva anche validation così puoi ricaricarla dopo
        pack = {
            'features': test_outputs,
            'labels':  np.array(gt_list, dtype=np.int64),
            'masks':   np.array(gt_mask_list),
            'images':  np.array(test_imgs),
        }
        save_split_pickle(pack, CODICE_PEZZO, POSITION, split="validation", method=METHOD)
    
    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    
    # calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE), mode='bilinear',
                            align_corners=False).squeeze().numpy()
    
    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)
    
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    
    # calculate image-level ROC AUC score
    img_scores = score_map.reshape(score_map.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    total_roc_auc.append(img_roc_auc)
    print('image ROCAUC: %.3f' % (img_roc_auc))
    fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (CODICE_PEZZO, img_roc_auc))
    
    # get optimal threshold
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score_map.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    preds = (img_scores >= threshold).astype(np.int32) 
    
    if VIS_PREDICTION_ON_VALID_DATASET:
        show_validation_grid_from_loader(
            val_loader, img_scores, preds,
            per_page=4, samples_per_row=2,
            show_mask=True, show_mask_product=True,
            overlay=True, overlay_alpha=0.45
        )

    # calculate per-pixel level ROCAUC
    fpr, tpr, thr_roc = roc_curve(gt_mask.flatten(), score_map.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), score_map.flatten())
    total_pixel_roc_auc.append(per_pixel_rocauc)
    print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))
    

    fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (CODICE_PEZZO, per_pixel_rocauc))
    # save_dir = args.save_path + '/' + f'pictures_{args.arch}'
    # os.makedirs(save_dir, exist_ok=True)
    # plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)
        
    # end old for

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    # fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)
     
     # (N_tot_pixel,)
    plt.plot(fpr, tpr, label=f"AUC={per_pixel_rocauc:.3f}")
    plt.plot([0,1],[0,1],'k--',linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("Pixel-level ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    J = tpr - fpr                         #TODO: vedere cosa è Youden's J statistic
    best_idx_roc = int(np.argmax(J))
    best_thr_roc = float(thr_roc[best_idx_roc])
    print(f"[pixel-level] Best threshold (ROC/Youden): {best_thr_roc:.6f}  | TPR={tpr[best_idx_roc]:.3f}  FPR={fpr[best_idx_roc]:.3f}")
    
    pred_pix = np.concatenate([sm.reshape(-1) for sm in score_map], axis=0) 
    
    gt_pix = []
    loader_masks = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
    for _, _, m in loader_masks:                 # m: (B,H,W) uint8 {0,1}
        gt_pix.append(m.numpy().reshape(m.size(0), -1))  # (B, H*W)
    gt_pix = np.concatenate(gt_pix, axis=0).ravel().astype(np.uint8)  

    # ----------------------------
    # 2) THRESHOLD da PR (F1 max)
    # ----------------------------
    prec, rec, thr_pr = precision_recall_curve(gt_pix, pred_pix)
    # NB: thr_pr ha len = len(prec)-1 = len(rec)-1
    f1_vals = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx_pr = int(np.argmax(f1_vals))
    best_thr_pr = float(thr_pr[best_idx_pr])
    print(f"[pixel-level] Best threshold (PR/F1):    {best_thr_pr:.6f}  | P={prec[best_idx_pr]:.3f}  R={rec[best_idx_pr]:.3f}  F1={f1_vals[best_idx_pr]:.3f}")

    # ------------------------------------------
    # 3) Applica le due soglie alle score map 2D
    #    (score_map_list è la lista delle mappe (H,W))
    # ------------------------------------------
    masks_roc = [(sm >= best_thr_roc).astype(np.uint8) for sm in score_map]
    masks_pr  = [(sm >= best_thr_pr ).astype(np.uint8) for sm in score_map]
    acc_r, p_r, r_r, f1_r = eval_pixel_metrics(masks_roc, val_set)
    acc_p, p_p, r_p, f1_p = eval_pixel_metrics(masks_pr,  val_set)

    print(f"[pixel-level] ROC thr -> Acc={acc_r:.3f}  P={p_r:.3f}  R={r_r:.3f}  F1={f1_r:.3f}")
    print(f"[pixel-level]  PR thr -> Acc={acc_p:.3f}  P={p_p:.3f}  R={r_p:.3f}  F1={f1_p:.3f}")
    
    
    show_heatmaps_from_loader(
        ds_or_loader=val_loader.dataset,   # o val_set
        score_maps=masks_roc,       
        scores=img_scores,                    # i tuoi image-level scores (N,)
        per_page=6,
        cols=3,
        normalize_each=False,
        overlay_alpha=0.45,
        cmap="jet",
        title_fmt="idx {i} | label {g} | score {s:.3f}",
    )


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        # fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z



if __name__ == '__main__':
    main()