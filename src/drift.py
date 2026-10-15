"""Feature 10: simple drift monitor - compares live category mix with training data + escalation rate."""
import pandas as pd
from src.config import DATA_PROCESSED
from src.observability import read_events


def drift_report(min_requests: int = 20, tvd_alert: float = 0.3, esc_alert: float = 0.4):
    ev, tp = read_events(), DATA_PROCESSED / "train.csv"
    if ev.empty or "category" not in ev or not tp.exists():
        return None
    req = ev[ev.node == "request"].dropna(subset=["category"])
    if req.empty:
        return None
    live = req.category.value_counts(normalize=True)
    ref = pd.read_csv(tp).category.value_counts(normalize=True)
    cats = sorted(set(live.index) | set(ref.index))
    table = pd.DataFrame({"training": [ref.get(c, 0) for c in cats], "live": [live.get(c, 0) for c in cats]}, index=cats)
    tvd = float(0.5 * (table.training - table.live).abs().sum())  # total variation distance, 0..1
    esc = float((req.status == "escalated").mean())
    return dict(n=len(req), tvd=tvd, escalation_rate=esc, avg_conf=float(req.confidence.mean()), table=table,
                alert=len(req) >= min_requests and (tvd > tvd_alert or esc > esc_alert))