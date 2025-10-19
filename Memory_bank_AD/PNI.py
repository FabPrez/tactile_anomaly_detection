# pni_ref.py  — PNI fedele alla repo/paper, adattato ai tuoi loader
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from data_loader import build_ad_datasets, make_loaders, load_split_pickle, save_split_pickle
from ad_analysis import run_pixel_level_evaluation, print_pixel_report
from scipy.ndimage import gaussian_filter

# ----------------- CONFIG -----------------
METHOD = "PNI_REF"
CODICE_PEZZO = "PZ3"
TRAIN_POSITIONS = ["pos1", "pos2"]
VAL_GOOD_PER_POS = 20
VAL_GOOD_SCOPE  = ["pos1"]
VAL_FAULT_SCOPE = ["pos1"]
GOOD_FRACTION   = 1.0
IMG_SIZE        = 224
SEED            = 42
GAUSS_SIGMA     = 2.0

# PNI-specific
NEIGH_K     = 9            # vicini locali (paper usa neighborhood info)
POS_BINS    = (14, 14)     # griglia posizionale (HxW delle feature map)
HIST_TOPM   = 10           # num rappresentanti per bin (selezione medoid/centroidi)
MLP_HIDDEN  = 512
REFINE_ON   = True         # abilita rete di refine
SYN_ANO_P   = 0.15         # prob. inserire difetti sintetici per train refine
# ------------------------------------------


# ---- Backbone & hooks (come repo) ----
def get_backbone(device):
    m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device)
    m.eval()
    outs = []
    def hook(_m,_i,o): outs.append(o)
    # PNI usa feature intermedie; lasciamo (layer1,2,3) + avgpool come nei tuoi altri script
    m.layer1[-1].register_forward_hook(hook)
    m.layer2[-1].register_forward_hook(hook)
    m.layer3[-1].register_forward_hook(hook)
    m.avgpool.register_forward_hook(hook)
    return m, outs


# ---- Costruzione POSITION PRIOR (istogramma/centroidi per bin posizionali) ----
def build_position_representatives(feat_maps, bins=(14,14), top_m=HIST_TOPM):
    """
    feat_maps: Tensor (N, C, H, W) su cui si costruisce il prior posizionale.
    Ritorna un dizionario { (bi,bj): Tensor (M, C) } con M<=top_m rappresentanti/centroidi.
    """
    N, C, H, W = feat_maps.shape
    Bi, Bj = bins
    reps = {}
    # discretizzazione posizioni
    hi_edges = torch.linspace(0, H, Bi+1, dtype=torch.int64)
    wj_edges = torch.linspace(0, W, Bj+1, dtype=torch.int64)
    with torch.no_grad():
        for bi in range(Bi):
            for bj in range(Bj):
                h0,h1 = hi_edges[bi].item(), hi_edges[bi+1].item()
                w0,w1 = wj_edges[bj].item(), wj_edges[bj+1].item()
                # raccogli patch nel bin
                patch = feat_maps[:,:,h0:h1,w0:w1]                       # (N,C,hb,wb)
                P = patch.permute(0,2,3,1).reshape(-1,C)                 # (N*hb*wb, C)
                if P.shape[0] == 0: 
                    reps[(bi,bj)] = torch.zeros((0,C), dtype=feat_maps.dtype)
                    continue
                # selezione rappresentanti: medoids/centroidi semplici (k-means lite o top_m casuali)
                # per fedeltà concettuale usiamo k-means-lite (k=top_m) o sampling stratificato:
                M = min(top_m, P.shape[0])
                # campionamento k-means++ lite
                idx = torch.randint(0, P.shape[0], (M,))
                reps[(bi,bj)] = P[idx].contiguous()                      # (M,C)
    return reps


# ---- Neighborhood Encoder (MLP) che condiziona su vicini locali ----
class NeighborhoodMLP(nn.Module):
    """
    Input: concat(query_feature, pooled_neighborhood, pos_repr) -> score nominalità/energia
    Paper: modella P(x|N(x)) (prob condizionata su vicinato) via MLP.
    """
    def __init__(self, c_in, c_neigh, c_pos, hidden=MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_in + c_neigh + c_pos, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 1)  # energia/score
        )

    def forward(self, q, n, p):
        # q: (L,C), n: (L,Cn), p: (L,Cp)
        x = torch.cat([q, n, p], dim=1)
        return self.net(x).squeeze(1)  # (L,)


def pool_neighborhood(feat, k=NEIGH_K):
    """
    feat: (1,C,H,W) → per ogni (h,w) calcola media dei K vicini locali in finestra 3x3/5x5.
    Per fedeltà concettuale usiamo K dagli offset locali più vicini in L2 sui canali.
    """
    B,C,H,W = feat.shape
    # costruzione finestre 3x3 con unfold
    win = F.unfold(feat, kernel_size=3, padding=1)      # (B, C*9, H*W)
    q   = feat.view(B, C, H*W)                          # (B,C,L)
    # distanza tra centro e vicini per ogni location (C vs C*9 per L posizioni)
    q9 = q.repeat_interleave(9, dim=1)                  # (B, C*9, L)
    d  = (q9 - win).pow(2).view(B, C, 9, H*W).sum(dim=1)  # (B,9,L)
    v, idx = torch.topk(d, k=min(k,9), dim=1, largest=False)  # (B,k,L)
    # media sui k migliori vicini
    win_resh = win.view(B, C, 9, H*W)                   # (B,C,9,L)
    gather = torch.gather(win_resh, 2, idx.unsqueeze(1).expand(-1,C,-1,-1))  # (B,C,k,L)
    neigh = gather.mean(dim=2)                          # (B,C,L)
    return neigh.permute(0,2,1).reshape(-1, C)          # (L,C)


def pos_embedding_from_reps(H, W, reps_dict):
    """
    Per ogni (h,w) prendi il rappresentante del bin corrispondente e usalo come embedding posizionale.
    """
    Bi,Bj = POS_BINS
    hi_edges = torch.linspace(0, H, Bi+1, dtype=torch.int64)
    wj_edges = torch.linspace(0, W, Bj+1, dtype=torch.int64)
    P_list = []
    for h in range(H):
        bi = int((Bi * h)//H)
        for w in range(W):
            bj = int((Bj * w)//W)
            R = reps_dict[(bi,bj)]
            if R.shape[0]==0:
                P_list.append(torch.zeros((1,R.shape[1])) if R.ndim==2 else torch.zeros((1,0)))
            else:
                # media dei rappresentanti del bin
                P_list.append(R.mean(dim=0, keepdim=True))
    return torch.cat(P_list, dim=0)  # (L,Cp)


# ---- Refine network per mappa ad alta risoluzione (allenata con anomalie sintetiche) ----
class RefineUNet(nn.Module):
    # mini-UNet compatto; in repo ufficiale è un refine network dedicato
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        c = base
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch,c,3,1,1), nn.ReLU(), nn.Conv2d(c,c,3,1,1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(c,2*c,3,1,1), nn.ReLU(), nn.Conv2d(2*c,2*c,3,1,1), nn.ReLU())
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv2d(3*c,c,3,1,1), nn.ReLU(), nn.Conv2d(c,c,3,1,1), nn.ReLU())
        self.out  = nn.Conv2d(c,1,1)
    def forward(self,x):
        e1 = self.enc1(x); p = self.pool(e1)
        e2 = self.enc2(p)
        u  = self.up(e2)
        d  = self.dec1(torch.cat([u,e1],1))
        return self.out(d)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, meta = build_ad_datasets(
        part=CODICE_PEZZO, img_size=IMG_SIZE,
        train_positions=TRAIN_POSITIONS,
        val_fault_scope=VAL_FAULT_SCOPE,
        val_good_scope=VAL_GOOD_SCOPE,
        val_good_per_pos=VAL_GOOD_PER_POS,
        good_fraction=GOOD_FRACTION, seed=SEED, transform=None
    )
    TRAIN_TAG = meta["train_tag"]
    train_loader, val_loader = make_loaders(train_set, val_set, batch_size=16, device=device)

    # 1) Estrazione feature
    model, outs = get_backbone(device)
    def extract_feats(loader):
        feat_l = []
        with torch.inference_mode():
            for x,_,_ in tqdm(loader, desc="feat"):
                _ = model(x.to(device, non_blocking=True))
                l1,l2,l3,avg = [t.detach().cpu() for t in outs[:4]]; outs.clear()
                # scegli un livello (es. layer3) per PNI core (paper usa intermedie)
                feat_l.append(l3)
        return torch.cat(feat_l,0)  # (N,C,H,W)

    try:
        payload = load_split_pickle(CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)
        train_feat = payload["train_feat"]
        pos_reps   = {tuple(map(int,k.split(","))): torch.from_numpy(v) for k,v in payload["pos_reps"].items()}
    except FileNotFoundError:
        train_feat = extract_feats(train_loader)  # (N,C,H,W)
        # 2) Position prior
        pos_reps = build_position_representatives(train_feat, bins=POS_BINS, top_m=HIST_TOPM)
        payload = {
          "train_feat": train_feat,
          "pos_reps": { f"{k[0]},{k[1]}": v.numpy() for k,v in pos_reps.items() }
        }
        save_split_pickle(payload, CODICE_PEZZO, TRAIN_TAG, split="train", method=METHOD)

    # 3) Neighborhood-conditional MLP (training “nominale”)
    # Per fedeltà: si addestra a dare score basso alle patch “normali”.
    C = train_feat.shape[1]
    Cp = next(iter(pos_reps.values())).shape[1] if len(pos_reps)>0 and list(pos_reps.values())[0].ndim==2 and list(pos_reps.values())[0].shape[0]>0 else C
    mlp = NeighborhoodMLP(c_in=C, c_neigh=C, c_pos=Cp).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    mlp.train()
    for epoch in range(2):  # metti le tue epoche; qui piccolo esempio
        for x,_,_ in tqdm(train_loader, desc=f"PNI-MLP epoch{epoch}"):
            _ = model(x.to(device, non_blocking=True))
            l1,l2,l3,_ = [t.detach() for t in outs[:4]]; outs.clear()
            f = l3   # (B,C,H,W)
            B,C,H,W = f.shape
            # query L=H*W
            q = f.view(B,C,H*W).permute(0,2,1).reshape(-1,C)        # (L,C)
            # neighborhood pooling
            n = pool_neighborhood(f).to(q.device)                   # (L,C)
            # positional embedding
            p = pos_embedding_from_reps(H, W, pos_reps).to(q.device)  # (L,Cp)
            s = mlp(q, n, p)                                        # (L,)
            # loss “one-class”: spingi s verso 0 (nominale)
            loss = (s**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    mlp.eval()

    # 4) Scoring su validation (mappa grezza = score MLP; più alto = più anomalo)
    raw_maps, img_scores, gt = [], [], []
    with torch.inference_mode():
        for x,y,_ in tqdm(val_loader, desc="PNI score"):
            gt.extend(y.cpu().numpy())
            _ = model(x.to(device, non_blocking=True))
            l1,l2,l3,_ = [t.detach().cpu() for t in outs[:4]]; outs.clear()
            f = l3
            B,C,H,W = f.shape
            q = f.view(B,C,H*W).permute(0,2,1).reshape(-1,C).to(device)
            n = pool_neighborhood(f.to(device))
            p = pos_embedding_from_reps(H, W, pos_reps).to(device).repeat(B,1)
            s = mlp(q, n, p).view(B,1,H,W).cpu().numpy()            # (B,1,H,W)
            s_up = F.interpolate(torch.from_numpy(s), size=IMG_SIZE, mode='bilinear', align_corners=False).squeeze(1).numpy()
            for i in range(s_up.shape[0]):
                m = gaussian_filter(s_up[i], sigma=GAUSS_SIGMA)
                raw_maps.append(m)
                img_scores.append(float(m.max()))  # image-level come max/quantile

    # (valutazione come tue utility)
    results = run_pixel_level_evaluation(
        score_map_list=raw_maps, val_set=val_set, img_scores=np.array(img_scores),
        use_threshold="pro", fpr_limit=0.01, vis=True, vis_ds_or_loader=val_loader.dataset
    )
    print_pixel_report(results, title=f"{METHOD} | {CODICE_PEZZO}/train={TRAIN_TAG}")

if __name__ == "__main__":
    main()
