"""Data collection: synthetic ticket generator, optional HuggingFace loader, and knowledge base writer."""
import random
import numpy as np
import pandas as pd
from src.config import DATA_RAW, KB_DIR, SEED

TEMPLATES = {
    "billing": ["I was charged twice for my {plan} subscription this month",
                "Please refund the {amount} charge on my invoice",
                "My invoice shows the wrong amount and my card was billed again",
                "Why did my payment fail even though my card is valid",
                "I want to cancel auto renewal and get a refund for {plan}"],
    "technical": ["The app crashes every time I open the {feature} page",
                  "I keep getting error {code} when I try to sync my data",
                  "The dashboard is loading very slowly and sometimes times out",
                  "API requests return 500 errors since this morning",
                  "Export to {feature} is broken after the latest update"],
    "account": ["I cannot log in and the password reset email never arrives",
                "Please change the email address on my account",
                "My account was locked after too many attempts",
                "How do I enable two factor authentication",
                "I need to delete my account and all my personal data"],
    "shipping": ["My order {order} has not arrived and tracking is stuck",
                 "The package was delivered to the wrong address",
                 "I received a damaged item in order {order}",
                 "Can I change the delivery address for order {order}",
                 "The courier marked my parcel delivered but I never got it"],
    "general": ["Do you offer discounts for students or nonprofits",
                "What are your support hours during holidays",
                "Can you tell me more about the {plan} features",
                "I would like to give feedback about your service",
                "Where can I find your documentation and tutorials"],
}
PLANS = ["Pro", "Basic", "Enterprise", "Team", "Starter"]
FEATURES = ["reports", "billing", "settings", "analytics", "profile", "checkout"]
FILLERS = ["I have been a customer for a while", "I already tried restarting everything",
           "Please let me know what to do next", "I contacted support before without luck",
           "Thanks in advance for your help", "This started a few days ago"]
URGENT = ["urgent", "asap", "immediately", "critical", "emergency", "right now"]
BASE_HOURS = {"billing": 12, "technical": 30, "account": 6, "shipping": 48, "general": 4}
PRI_MULT = {"low": 1.3, "medium": 1.0, "high": 0.6}


def _fill(t, rng):
    return t.format(plan=rng.choice(PLANS), amount=f"${rng.randint(5, 500)}",
                    feature=rng.choice(FEATURES), code=rng.randint(100, 599),
                    order=f"#{rng.randint(10000, 99999)}")


def generate_synthetic(n: int = 4000, seed: int = SEED) -> pd.DataFrame:
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    cats = list(TEMPLATES)
    rows = []
    for _ in range(n):
        cat = rng.choices(cats, weights=[.22, .28, .20, .18, .12])[0]
        pri = rng.choices(["low", "medium", "high"], weights=[.35, .45, .20])[0]
        text = _fill(rng.choice(TEMPLATES[cat]), rng)
        if rng.random() < .6:
            text += ". " + rng.choice(FILLERS)
        if rng.random() < .08:  # mixed-topic tickets add realistic noise
            text += ". Also, " + _fill(rng.choice(TEMPLATES[rng.choice(cats)]), rng).lower()
        if pri == "high":
            text = rng.choice(URGENT).capitalize() + ": " + text + rng.choice(["!", "!!", ". This is blocking my work."])
        elif pri == "low" and rng.random() < .5:
            text += ". No rush"
        tenure, prior = int(nrng.integers(0, 72)), int(nrng.poisson(2))
        hours = BASE_HOURS[cat] * PRI_MULT[pri] * (1 + .1 * (prior > 3)) * float(nrng.lognormal(0, .25))
        rows.append(dict(text=text, category=cat, priority=pri, channel=rng.choice(["email", "chat", "phone", "web"]),
                         tenure_months=tenure, prior_tickets=prior, resolution_hours=round(hours, 2)))
    return pd.DataFrame(rows)


def load_hf_dataset(limit: int = 5000) -> pd.DataFrame:
    """Optional real-world data (Bitext customer support). Numeric columns are filled synthetically."""
    from datasets import load_dataset
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train").to_pandas()
    ds = ds.sample(min(limit, len(ds)), random_state=SEED)
    out = generate_synthetic(len(ds))
    out["text"], out["category"] = ds["instruction"].values, ds["category"].str.lower().values
    return out


KB = {
 "billing": "# Billing and refunds\nDuplicate charges are refunded automatically within 5-7 business days once verified. Customers can request a refund from Settings > Billing > Invoices within 30 days of the charge.\nTo cancel auto renewal go to Settings > Billing > Plan and choose Cancel renewal. Access continues until the end of the paid period.\nFailed payments are retried three times over 7 days. Update the card under Settings > Billing > Payment methods. Enterprise invoices can be paid by bank transfer.",
 "technical": "# Technical troubleshooting\nFor crashes, clear the app cache, update to the latest version and retry. If the crash persists, collect logs from Help > Diagnostics and attach them to the ticket.\nSync errors (codes 400-599) usually resolve after signing out and in again. Platform incidents are published on the status page.\nAPI 500 errors: check rate limits, retry with exponential backoff, and include the request ID from the response header when contacting support. Slow dashboards are often caused by large date ranges; narrow the filter.",
 "account": "# Account and security\nPassword reset emails arrive within 5 minutes; check spam and verify the address. After 5 failed logins an account locks for 30 minutes, or can be unlocked via the reset link.\nTwo factor authentication can be enabled in Settings > Security using any authenticator app.\nEmail changes require confirmation from both the old and new address. Account deletion removes personal data within 30 days in line with GDPR; request it from Settings > Privacy.",
 "shipping": "# Shipping and delivery\nStandard delivery takes 3-7 business days. Tracking can stay unchanged for up to 48 hours between hubs. If a parcel is marked delivered but missing, wait 24 hours and check with neighbours, then open a claim.\nDamaged items are replaced or refunded; attach photos of the package and item. The delivery address can be changed before dispatch from the Orders page.",
 "general": "# General information\nSupport hours are 8am-8pm on weekdays and 10am-4pm on weekends, with reduced hours on public holidays. Students and nonprofits receive a 30 percent discount on request.\nPlan comparisons are on the pricing page. Documentation and video tutorials are available in the Help Center. Feedback can be submitted through the in-app feedback form.",
}


def write_knowledge_base():
    for k, v in KB.items():
        (KB_DIR / f"{k}.md").write_text(v)
    return list(KB)


def load_or_generate(n=4000, source="synthetic", force=False) -> pd.DataFrame:
    path = DATA_RAW / "tickets.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    df = load_hf_dataset(n) if source == "hf" else generate_synthetic(n)
    df.to_csv(path, index=False)
    return df
