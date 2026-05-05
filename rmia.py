import time
import numpy as np
import torch
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
# ─── Configuration ─────────────────────────────────────────────────────────────
PATH = Path(__file__).parent
PUB_PATH   = PATH / "pub.pt"
PRIV_PATH  = PATH / "priv.pt"
MODEL_PATH = PATH / "model.pt"
OUTPUT_CSV = PATH / "submission.csv"
REF_DIR    = PATH / "ref_models"


# Hyperparameters ───────────────────────────────────────────────────────────
NUM_CLASSES    = 9
NUM_REF_MODELS = 4    # OUT reference models to train (more → better, slower)
REF_EPOCHS     = 60     # epochs per reference model
GAMMA          = 2.0    # pairwise LR threshold γ (paper §3, Fig.7 → γ=2 best)
A_PARAM        = 0.3    # offline Pr(x) correction 'a' (paper App.B.2.2; 0.3 CIFAR)
BATCH_SIZE     = 256
USE_AUG        = False   # 6-view augmentation for confidence estimates

# SM-Taylor-Softmax hyperparams (empirically chosen)
SMT_N, SMT_M, SMT_T = 4, 0.6, 2.0

MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]
NORM = transforms.Compose([transforms.Resize(32),
                            transforms.Normalize(MEAN, STD)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[mia_solution] Device: {device}")


# ─── Dataset Classes (identical to task_template.py) ──────────────────────────
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids, self.imgs, self.labels = [], [], []
        self.transform = transform

    def __getitem__(self, i):
        img = self.transform(self.imgs[i]) if self.transform else self.imgs[i]
        return self.ids[i], img, self.labels[i]

    def __len__(self): return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, i):
        id_, img, lbl = super().__getitem__(i)
        return id_, img, lbl, self.membership[i]


class Strip(Dataset):
    """Strips the membership field so all loaders see (id, img, label)."""
    def __init__(self, ds): self.ds = ds
    def __len__(self):      return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        return item[:3]   # works for both TaskDataset (3) and MembershipDataset (4)


# ─── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    m = resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, NUM_CLASSES)
    return m


# ─── SM-Taylor-Softmax Confidence ─────────────────────────────────────────────
def smt_conf(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    
    logits = logits / SMT_T
    B, dev = logits.shape[0], logits.device
    idx    = torch.arange(B, device=dev)

    def apx(x):
        """nth-order Taylor approx of exp(x)."""
        r = t = torch.ones_like(x)
        for i in range(1, SMT_N + 1):
            t = t * x / i
            r = r + t
        return r.clamp(min=1e-10)

    cy  = logits[idx, labels]               # logit of true class
    num = apx(cy - SMT_M)                   # numerator
    aa  = apx(logits)                       # [B, C]
    den = aa.sum(1) - aa[idx, labels] + num # denominator
    return (num / den.clamp(min=1e-10)).clamp(0, 1)


# ─── Confidence Extraction ────────────────────────────────────────────────────
@torch.no_grad()
def get_conf(model: nn.Module, ds: Dataset, augment: bool = False) -> dict:
    """
    Run `model` over `ds` and return {id: scalar_confidence}.

    When `augment=True`, confidence is the mean of 6 views:
      original, h-flip, roll-right-2, roll-left-2, roll-down-2, roll-up-2.
    This mirrors the multi-query boost described in paper §5.2 / App. B.5.
    """
    model.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=(device.type == "cuda"))
    all_ids, all_c = [], []

    for ids, imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if augment:
            views = [
                imgs,
                imgs.flip(-1),              # horizontal flip
                imgs.roll( 2, -1),          # shift right by 2px (cyclic)
                imgs.roll(-2, -1),          # shift left
                imgs.roll( 2, -2),          # shift down
                imgs.roll(-2, -2),          # shift up
            ]
            # Mean confidence over all views
            c = torch.stack([smt_conf(model(v), labels) for v in views]).mean(0)
        else:
            c = smt_conf(model(imgs), labels)

        if torch.is_tensor(ids):
            ids = ids.tolist()
        all_ids += list(ids)
        all_c   += c.cpu().numpy().tolist()

    return dict(zip(all_ids, all_c))


# ─── Reference Model Training ─────────────────────────────────────────────────
def train_ref_model(subset: Dataset, epochs: int = REF_EPOCHS) -> nn.Module:
    """
    Train one OUT reference model on `subset` (50% of pub.pt chosen at random).
    Same architecture and training recipe as the target model setup (SGD +
    cosine-annealing LR, weight-decay=5e-4, batch=256).
    """
    model = build_model().to(device)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, drop_last=False,
                        pin_memory=(device.type == "cuda"))
    opt   = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        for _, imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            crit(model(imgs), labels).backward()
            opt.step()
        sch.step()
        if ep % 20 == 0:
            print(f"      [ref] epoch {ep}/{epochs}")

    model.eval()
    return model


# ─── RMIA (Offline) ───────────────────────────────────────────────────────────
def compute_rmia(
    target:  nn.Module,
    refs:    list,
    q_ds:    Dataset,   # samples to score
    z_ds:    Dataset,   # population Z (non-members of pub.pt)
    augment: bool = False,
) -> dict:
    """
    Offline RMIA (Algorithm 1, paper §3 + App. B.2.2).
    Returns {id: score ∈ [0,1]} for every sample in q_ds.

    Key formulas
    ------------
    Ratio_z  = Pr(z|θ) / Pr(z)_OUT          # z uses raw OUT mean (Alg.1 L17)
    Pr(x)    = ½·[(1+a)·Pr(x)_OUT + (1-a)]  # offline correction, eq.10
    Ratio_x  = Pr(x|θ) / Pr(x)
    Score(x) = fraction of z ∈ Z where Ratio_x / Ratio_z > γ  (eq.5)
    """
    print("    → target model confidences ...")
    q_tc = get_conf(target, q_ds, augment)   # Pr(x|θ)
    z_tc = get_conf(target, z_ds, augment)     # Pr(z|θ)  — no aug for Z

    print("    → reference model confidences ...")
    q_rc_list, z_rc_list = [], []
    for k, ref in enumerate(refs):
        print(f"      ref {k+1}/{len(refs)}")
        q_rc_list.append(get_conf(ref, q_ds, augment))
        z_rc_list.append(get_conf(ref, z_ds, augment))

    q_ids = list(q_tc.keys())
    z_ids = list(z_tc.keys())
    K     = len(refs)

    # ── Pre-compute Z ratios ──────────────────────────────────────────────────
    # Pr(z)_OUT = raw OUT mean (Alg.1 line 17)
    z_out = np.array([
        sum(rc[z] for rc in z_rc_list) / K
        for z in z_ids
    ])
    z_tgt = np.array([z_tc[z] for z in z_ids])
    z_rat = z_tgt / np.maximum(z_out, 1e-10)   # Ratio_z  [|Z|]

    # ── Score each query ──────────────────────────────────────────────────────
    scores = {}
    for xid in q_ids:
        # Pr(x)_OUT
        px_out = sum(rc[xid] for rc in q_rc_list) / K
        # Offline Pr(x) approximation  (eq.10)
        px     = 0.5 * ((1.0 + A_PARAM) * px_out + (1.0 - A_PARAM))
        # Ratio_x = Pr(x|θ) / Pr(x)
        rx     = q_tc[xid] / max(px, 1e-10)
        # RMIA score: fraction of z where rx / rz > γ
        scores[xid] = float(np.mean(rx / np.maximum(z_rat, 1e-10) > GAMMA))

    return scores


# ─── Evaluation Helper ────────────────────────────────────────────────────────
def tpr_at_5fpr(score_dict: dict, mem_ds) -> float:
    """
    Compute TPR @ 5% FPR on a MembershipDataset (which has ground-truth labels).
    We sweep the score threshold from high to low (decreasing order) and
    track the maximum TPR achieved while FPR ≤ 0.05.
    """
    labels, preds = [], []
    for i in range(len(mem_ds)):
        id_, _, _, m = mem_ds[i]
        labels.append(int(m) if m is not None else 0)
        preds.append(score_dict.get(id_, 0.5))

    labels, preds = np.array(labels), np.array(preds)
    order  = np.argsort(-preds)        # descending score order
    sl     = labels[order]
    n_pos  = sl.sum()
    n_neg  = len(sl) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    fp = tp = best = 0
    for lv in sl:
        if lv: tp += 1
        else:  fp += 1
        fpr = fp / n_neg
        if fpr <= 0.05:
            best = tp / n_pos
        else:
            break   # FPR only increases from here; break early

    return best

REF_DIR.mkdir(exist_ok=True)





# ── Load datasets ──────────────────────────────────────────────────────────
print("\n[1/4] Loading datasets ...")
pub_ds  = torch.load(PUB_PATH,  weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)
pub_ds.transform  = NORM
priv_ds.transform = NORM

pub_plain  = Strip(pub_ds)
priv_plain = Strip(priv_ds)

m_idx  = [i for i in range(len(pub_ds)) if pub_ds.membership[i] == 1]
nm_idx = [i for i in range(len(pub_ds)) if pub_ds.membership[i] == 0]
split_point = len(nm_idx) // 2
z_idx       = nm_idx[:split_point]
ref_pool_nm = nm_idx[split_point:]
z_ds = Subset(pub_plain, z_idx)
ref_train_pool = m_idx + ref_pool_nm

print(f"  pub.pt    : {len(pub_ds):,} samples")
print(f"  priv.pt   : {len(priv_ds):,} samples")
print(f"  Z samples (clean): {len(z_idx):,} (Strictly isolated)")
print(f"  Ref train pool   : {len(ref_train_pool):,} samples")

# ── Load target model ──────────────────────────────────────────────────────
print("\n[2/4] Loading target model ...")
target = build_model().to(device)
target.load_state_dict(torch.load(MODEL_PATH, map_location=device))
target.eval()



# ── Baseline: confidence attack  ──────
print("\n[3/4] Baseline confidence scores ...")
t0 = time.time()
pub_conf  = get_conf(target, pub_plain,  augment=USE_AUG)
priv_conf = get_conf(target, priv_plain, augment=USE_AUG)
bl_score  = tpr_at_5fpr(pub_conf, pub_ds)
print(f"  SM-Taylor confidence  TPR@5%%FPR = {bl_score:.4f}  "
      f"({time.time()-t0:.1f}s)")

# ── Train / load reference models ──────────────────────────────────────────
print(f"\n[4/4] Reference models ({NUM_REF_MODELS} OUT models × {REF_EPOCHS} epochs) ...")
refs = []
train_masks = []
np.random.seed(42)      

for k in range(NUM_REF_MODELS):
    ckpt = REF_DIR / f"ref_{k}_ep{REF_EPOCHS}.pt"
    mask_path = REF_DIR / f"ref_{k}_ep{REF_EPOCHS}_mask.pt"
    if ckpt.exists() and mask_path.exists():
        print(f"  Loading cached ref {k+1} and its train mask ← {ckpt.name}")
        m = build_model().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        mask = torch.load(mask_path, weights_only=False)
    else:
        print(f"  Training ref {k+1}/{NUM_REF_MODELS} ...")
        t0  = time.time()
        # Random 50% of pub.pt → OUT model  (x ∉ this subset ∀x ∈ priv.pt)
        np.random.shuffle(ref_train_pool)
        subset_idx = ref_train_pool[: len(ref_train_pool) // 2]
        sub = Subset(pub_plain, subset_idx)
        m   = train_ref_model(sub, REF_EPOCHS)
        mask = set([pub_ds.ids[i] for i in subset_idx])

        torch.save(m.state_dict(), ckpt)
        torch.save(mask, mask_path)
        print(f"  Done ({time.time()-t0:.0f}s) → saved {ckpt.name} + mask")

    refs.append(m)
    train_masks.append(mask)


# ── Define True Validation RMIA ────────────────────────────────────────────
def compute_true_validation_rmia(q_tc, z_tc, q_rc_list, z_rc_list, train_masks, gamma, a_param):
    z_ids = list(z_tc.keys())
    K_total = len(q_rc_list)

    # Pr(z)_OUT: Since Z is strictly isolated, all ref models are "OUT" for all Z
    z_out = np.array([sum(rc[z] for rc in z_rc_list) / K_total for z in z_ids])
    z_tgt = np.array([z_tc[z] for z in z_ids])
    z_rat = z_tgt / np.maximum(z_out, 1e-10)

    scores = {}
    for xid in q_tc.keys():
        # FIND THE "OUT" MODELS: only use models where xid was NOT in the training mask
        out_model_confs = [q_rc_list[k][xid] for k in range(K_total) if xid not in train_masks[k]]

        if len(out_model_confs) == 0:
            px_out = sum(q_rc_list[k][xid] for k in range(K_total)) / K_total
        else:
            px_out = sum(out_model_confs) / len(out_model_confs)

        px = 0.5 * ((1.0 + a_param) * px_out + (1.0 - a_param))
        rx = q_tc[xid] / max(px, 1e-10)
        scores[xid] = float(np.mean(rx / np.maximum(z_rat, 1e-10) > gamma))

    return scores

# ── Extract Confidences (Runs only once!) ──────────────────────────────────
print("\n[4/4] Extracting Confidences (Augmentation:", USE_AUG, ") ...")
t0 = time.time()
pub_tc  = get_conf(target, pub_plain,  augment=USE_AUG)
priv_tc = get_conf(target, priv_plain, augment=USE_AUG)
z_tc    = get_conf(target, z_ds,       augment=USE_AUG)

pub_rc_list, priv_rc_list, z_rc_list = [], [], []
for k, ref in enumerate(refs):
    print(f"  Extracting ref {k+1}/{len(refs)} ...")
    pub_rc_list.append(get_conf(ref, pub_plain, augment=USE_AUG))
    priv_rc_list.append(get_conf(ref, priv_plain, augment=USE_AUG))
    z_rc_list.append(get_conf(ref, z_ds, augment=USE_AUG))
print(f"  Done in {time.time()-t0:.1f}s")

# ── RMIA Hyperparameter Sweep on pub.pt ────────────────────────────────────
print("\n── Sweeping Hyperparameters on pub.pt (Validation) ──")
gammas_to_test   = [1.5, 2.0, 2.5]
a_params_to_test = [0.1, 0.3, 0.5]

best_rm_score = 0.0
best_params   = (GAMMA, A_PARAM)
best_pub_rmia = None

for g in gammas_to_test:
    for a in a_params_to_test:
        pub_rmia = compute_true_validation_rmia(pub_tc, z_tc, pub_rc_list, z_rc_list, train_masks, g, a)
        score    = tpr_at_5fpr(pub_rmia, pub_ds)
        print(f"  Gamma: {g:<3} | A_param: {a:<3} --> TPR@5%FPR: {score:.4f}")

        if score > best_rm_score:
            best_rm_score = score
            best_params   = (g, a)
            best_pub_rmia = pub_rmia

opt_gamma, opt_a = best_params
print(f"\n  ⭐ BEST RMIA VALIDATION: {best_rm_score:.4f} (Gamma={opt_gamma}, A_param={opt_a})")

# ── Score priv.pt using best parameters ────────────────────────────────────
print(f"\n── Scoring priv.pt with best parameters ...")
# For priv.pt, the test samples were NEVER seen by any reference model,
# so we can safely pass an empty mask [set() for _ in range(NUM_REF_MODELS)]
empty_masks = [set() for _ in range(NUM_REF_MODELS)]
priv_rmia = compute_true_validation_rmia(priv_tc, z_tc, priv_rc_list, z_rc_list, empty_masks, opt_gamma, opt_a)

# ── Baseline Confidence (Attack-P) ─────────────────────────────────────────
bl_score = tpr_at_5fpr(pub_tc, pub_ds)

print("\n" + "─"*55)
print(f"  Baseline Confidence TPR@5%%FPR : {bl_score:.4f}")
print(f"  RMIA (Offline)      TPR@5%%FPR : {best_rm_score:.4f}")

if best_rm_score >= bl_score:
    final  = priv_rmia
    method = f"RMIA (Gamma={opt_gamma}, a={opt_a})"
else:
    final  = priv_tc
    method = "Confidence (SM-Taylor)"
print(f"  → Submitting with : {method}")
print("─"*55)

# ── Build submission CSV ───────────────────────────────────────────────────
ids    = [str(x) for x in priv_ds.ids]
scores = [float(final.get(x, 0.5)) for x in priv_ds.ids]

assert len(ids) == len(scores), "ID/score length mismatch"
assert all(0.0 <= s <= 1.0 for s in scores), "Score(s) outside [0,1]"
assert len(set(ids)) == len(ids), "Duplicate IDs found"

df = pd.DataFrame({"id": ids, "score": scores})
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")
print(f"  {len(df):,} rows | mean score = {np.mean(scores):.4f}")
