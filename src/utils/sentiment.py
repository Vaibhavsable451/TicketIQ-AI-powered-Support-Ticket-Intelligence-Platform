"""Feature 2: lightweight lexicon sentiment (no model download). Score in [-1, 1]."""
import math, re

NEG = {"angry", "terrible", "awful", "worst", "horrible", "furious", "unacceptable", "ridiculous", "useless",
       "scam", "fraud", "disappointed", "frustrated", "annoyed", "hate", "waste", "pathetic", "disgusting", "unhappy"}
POS = {"thanks", "thank", "great", "love", "good", "happy", "appreciate", "excellent", "perfect", "helpful", "awesome"}


def score_text(text: str):
    toks = re.findall(r"[a-z']+", text.lower())
    pos, neg = sum(t in POS for t in toks), sum(t in NEG for t in toks)
    raw = pos - neg - (0.3 * min(text.count("!"), 3) if neg else 0)
    s = math.tanh(raw / 2)
    label = "very negative" if s <= -0.5 else "negative" if s < -0.1 else "positive" if s > 0.1 else "neutral"
    return round(s, 2), label