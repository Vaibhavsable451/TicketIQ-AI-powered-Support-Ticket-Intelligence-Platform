"""Deep learning in PyTorch: MLP over engineered features with early stopping."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from src.config import MODELS_DIR, SEED


class MLP(nn.Module):
    def __init__(self, d_in, n_cls, hidden=(256, 64), p=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def train_mlp(Xtr, ytr, Xte, yte, classes, epochs=60, lr=1e-3, patience=6, batch=64):
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cidx = {c: i for i, c in enumerate(classes)}
    Xtr, Xte = _dense(Xtr).astype("float32"), _dense(Xte).astype("float32")
    ytr_i, yte_i = np.array([cidx[c] for c in ytr]), np.array([cidx[c] for c in yte])
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr_i, test_size=0.15, stratify=ytr_i, random_state=SEED)
    T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    Xa_t, ya_t, Xv_t, yv_t = T(Xa), T(ya, torch.long), T(Xv), T(yv, torch.long)

    model = MLP(Xtr.shape[1], len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    loss_fn = nn.CrossEntropyLoss()
    hist, best, best_state, bad = {"train_loss": [], "val_loss": [], "val_acc": []}, 1e9, None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xa_t), device=dev)
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xa_t[idx]), ya_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            out = model(Xv_t)
            vl = loss_fn(out, yv_t).item()
            va = (out.argmax(1) == yv_t).float().mean().item()
        sched.step(vl)
        hist["train_loss"].append(tot / len(Xa_t)); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[mlp] early stop at epoch {ep + 1}")
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(T(Xte)).argmax(1).cpu().numpy()
    metrics = dict(model="PyTorch MLP", accuracy=accuracy_score(yte_i, pred),
                   f1_macro=f1_score(yte_i, pred, average="macro"))
    torch.save({"state": best_state, "d_in": Xtr.shape[1], "classes": list(classes)}, MODELS_DIR / "mlp.pt")
    print(f"[mlp] acc={metrics['accuracy']:.3f} f1={metrics['f1_macro']:.3f}")
    return metrics, hist
