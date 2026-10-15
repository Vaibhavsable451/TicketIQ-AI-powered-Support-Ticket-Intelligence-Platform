"""LangGraph agent:  enrich -> classify -> (retrieve -> draft -> verify) | escalate -> respond"""
import hashlib, json, re
from collections import OrderedDict
from functools import lru_cache
from typing import TypedDict
import joblib
import pandas as pd
from langgraph.graph import END, START, StateGraph
from src.config import GROQ_API_KEY, GROQ_MODEL, MODELS_DIR
from src.data.preprocess import prepare_frame
from src.duplicates import similar_tickets
from src.explain import top_terms
from src.observability import log, log_request, setup_langsmith, traced
from src.utils.pii import redact
from src.utils.sentiment import score_text

CONF_THRESHOLD = 0.55
SLOW_HOURS = 40
ANGRY = -0.5


class TicketState(TypedDict, total=False):
    ticket: str
    ticket_safe: str
    pii_found: list
    sentiment: float
    sentiment_label: str
    channel: str
    tenure_months: int
    prior_tickets: int
    category: str
    confidence: float
    priority: str
    priority_conf: float
    eta_hours: float
    top_terms: list
    duplicates: list
    context: str
    sources: list
    answer: str
    draft_answer: str
    verdict: str
    grounded: bool
    escalate: bool
    reason: str


@lru_cache(maxsize=1)
def _models():
    return joblib.load(MODELS_DIR / "classifier.joblib"), joblib.load(MODELS_DIR / "regressor.joblib")


@lru_cache(maxsize=1)
def _priority_model():
    p = MODELS_DIR / "priority.joblib"
    return joblib.load(p) if p.exists() else None


@lru_cache(maxsize=1)
def _llm():
    if not GROQ_API_KEY:
        return None
    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, temperature=0.2)


def _esc(s: str) -> str:  # braces would be read as template variables
    return s.replace("{", "(").replace("}", ")")


def _ask(system: str, human: str) -> str:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    chain = ChatPromptTemplate.from_messages([("system", system), ("human", human)]) | _llm() | StrOutputParser()
    return chain.invoke({})


@traced("enrich")
def enrich(state: TicketState):
    safe, found = redact(state["ticket"])          # Feature 3: PII redaction
    s, label = score_text(state["ticket"])          # Feature 2: sentiment
    return {"ticket_safe": safe, "pii_found": found, "sentiment": s, "sentiment_label": label}


@traced("classify")
def classify(state: TicketState):
    clf, reg = _models()
    row = prepare_frame(pd.DataFrame([{"text": state["ticket"], "channel": state.get("channel", "email"),
                                       "tenure_months": state.get("tenure_months", 12),
                                       "prior_tickets": state.get("prior_tickets", 0)}]))
    proba = clf.predict_proba(row)[0]
    i = int(proba.argmax())
    cat = str(clf.classes_[i])
    out = {"category": cat, "confidence": float(proba[i]), "eta_hours": float(reg.predict(row)[0]),
           "priority": "n/a", "priority_conf": 0.0,
           "top_terms": top_terms(state["ticket"], cat),               # Feature 5
           "duplicates": similar_tickets(state["ticket"])}             # Feature 8
    pri = _priority_model()                                            # Feature 1
    if pri is not None:
        pp = pri.predict_proba(row)[0]
        j = int(pp.argmax())
        out["priority"], out["priority_conf"] = str(pri.classes_[j]), float(pp[j])
    return out


def route_after_classify(state: TicketState):
    if state["confidence"] < CONF_THRESHOLD:
        return "escalate"
    if state.get("priority") == "high" and state.get("sentiment", 0) <= ANGRY:
        return "escalate"
    return "retrieve"


@traced("retrieve")
def retrieve(state: TicketState):
    from src.rag.vector_store import get_vector_store
    docs = get_vector_store().similarity_search(f"{state['category']}: {state['ticket_safe']}", k=3)
    return {"context": "\n\n".join(d.page_content for d in docs),
            "sources": sorted({d.metadata.get("source", "?") for d in docs})}


@traced("draft")
def draft(state: TicketState):
    if _llm() is None:  # offline fallback: extractive answer from top chunk
        return {"answer": "(No GROQ_API_KEY - showing knowledge-base excerpt)\n\n" + state["context"].split("\n\n")[0]}
    tone = " The customer is upset: open with a short, sincere apology." if state.get("sentiment", 0) < -0.1 else ""
    ans = _ask("You are a concise, friendly support agent. Answer ONLY using the context. "
               "If the context is insufficient say so." + tone + " Context:\n" + _esc(state["context"]),
               "Customer ticket: " + _esc(state["ticket_safe"]))  # redacted text only goes to the LLM
    return {"answer": ans}


@traced("verify")
def verify(state: TicketState):
    if _llm() is None:
        return {"grounded": True}
    verdict = _ask(
        "You are a fair fact-checker. Reply with ONE word. YES if the answer is consistent with the context "
        "(paraphrasing or omitting details is fine). NO only if it contradicts the context or invents facts "
        "that are not in it.\nContext:\n" + _esc(state["context"]),
        "Answer:\n" + _esc(state["answer"]))
    words = re.sub(r"[^A-Za-z ]", " ", verdict).split()
    return {"grounded": not (words and words[0].upper() == "NO"), "verdict": verdict.strip()[:200]}


def route_after_verify(state: TicketState):
    return "respond" if state.get("grounded", True) else "escalate"


@traced("escalate")
def escalate(state: TicketState):
    if state["confidence"] < CONF_THRESHOLD:
        why = "low classifier confidence"
    elif "context" not in state:
        why = "angry customer with high-priority ticket"
    else:
        why = "answer not grounded in knowledge base"
    return {"escalate": True, "reason": why, "draft_answer": state.get("answer", ""),
            "answer": f"Escalated to a human agent ({why})."}


@traced("respond")
def respond(state: TicketState):
    slow = state["eta_hours"] > SLOW_HOURS
    return {"escalate": False, "reason": "predicted slow resolution - flag for monitoring" if slow else "auto-resolved"}


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(TicketState)
    for name, fn in (("enrich", enrich), ("classify", classify), ("retrieve", retrieve), ("draft", draft),
                     ("verify", verify), ("escalate", escalate), ("respond", respond)):
        g.add_node(name, fn)
    g.add_edge(START, "enrich")
    g.add_edge("enrich", "classify")
    g.add_conditional_edges("classify", route_after_classify, {"retrieve": "retrieve", "escalate": "escalate"})
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"respond": "respond", "escalate": "escalate"})
    g.add_edge("respond", END)
    g.add_edge("escalate", END)
    return g.compile()


_CACHE: "OrderedDict[str, dict]" = OrderedDict()  # Feature 7: response cache (LRU, 256 entries)


def handle_ticket(ticket: str, use_cache: bool = True, **meta) -> dict:
    setup_langsmith()
    key = hashlib.sha256(json.dumps([ticket, meta], sort_keys=True, default=str).encode()).hexdigest()
    if use_cache and key in _CACHE:
        _CACHE.move_to_end(key)
        return {**_CACHE[key], "cached": True}
    result = build_graph().invoke({"ticket": ticket, **meta},
                                  config={"run_name": "ticket_triage", "tags": ["ticketiq"], "metadata": meta})
    log_request(ticket, result)
    log.info("category=%s conf=%.2f priority=%s escalate=%s", result.get("category"),
             result.get("confidence", 0), result.get("priority"), result.get("escalate"))
    _CACHE[key] = result
    if len(_CACHE) > 256:
        _CACHE.popitem(last=False)
    return {**result, "cached": False}