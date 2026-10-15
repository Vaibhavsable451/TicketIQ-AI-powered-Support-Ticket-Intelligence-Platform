"""Fine-tune DistilBERT for ticket classification (plain PyTorch loop; GPU recommended, CPU works on small data)."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from src.config import MODELS_DIR, SEED, TARGET_CLASS


def finetune_distilbert(train, test, epochs=2, batch=16, lr=5e-5, max_len=96, model_name="distilbert-base-uncased", sample=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if sample:
        train = train.sample(min(sample, len(train)), random_state=SEED)
    classes = sorted(train[TARGET_CLASS].unique())
    cidx = {c: i for i, c in enumerate(classes)}
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(classes), id2label=dict(enumerate(classes)), label2id=cidx).to(dev)

    def enc(texts):
        return tok(list(texts), truncation=True, padding=True, max_length=max_len, return_tensors="pt")

    ytr = torch.tensor([cidx[c] for c in train[TARGET_CLASS]])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    texts = train["text"].tolist()
    for ep in range(epochs):
        model.train()
        perm = np.random.RandomState(SEED + ep).permutation(len(texts))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            b = {k: v.to(dev) for k, v in enc([texts[j] for j in idx]).items()}
            loss = model(**b, labels=ytr[idx].to(dev)).loss
            loss.backward(); opt.step(); opt.zero_grad()
        print(f"[finetune] epoch {ep + 1}/{epochs} last loss {loss.item():.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        tt = test["text"].tolist()
        for i in range(0, len(tt), 64):
            b = {k: v.to(dev) for k, v in enc(tt[i:i + 64]).items()}
            preds += model(**b).logits.argmax(1).cpu().tolist()
    y = [cidx[c] for c in test[TARGET_CLASS]]
    out = MODELS_DIR / "distilbert-ticket"
    model.save_pretrained(out); tok.save_pretrained(out)
    return dict(model="DistilBERT (fine-tuned)", accuracy=accuracy_score(y, preds), f1_macro=f1_score(y, preds, average="macro"))
