"""Observability: LangSmith tracing, structured logging, and per-node latency events (JSONL)."""
import functools, json, logging, os, time
from datetime import datetime, timezone
from src.config import LANGSMITH_API_KEY, LANGSMITH_PROJECT, LOG_DIR

EVENTS = LOG_DIR / "events.jsonl"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ticketiq")


def setup_langsmith() -> bool:
    """Enable LangSmith tracing for all LangChain/LangGraph runs when a key is present."""
    if not LANGSMITH_API_KEY:
        return False
    os.environ.update(LANGSMITH_TRACING="true", LANGSMITH_API_KEY=LANGSMITH_API_KEY, LANGSMITH_PROJECT=LANGSMITH_PROJECT)
    return True


def _emit(event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(EVENTS, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def traced(node: str):
    """Decorator for LangGraph nodes: records latency, status, and errors."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state):
            t = time.perf_counter()
            try:
                out = fn(state)
                _emit(dict(node=node, status="ok", latency_ms=round((time.perf_counter() - t) * 1000, 1)))
                return out
            except Exception as e:  # noqa: BLE001
                log.exception("node %s failed", node)
                _emit(dict(node=node, status="error", error=str(e), latency_ms=round((time.perf_counter() - t) * 1000, 1)))
                raise
        return wrapper
    return deco


def log_request(ticket: str, result: dict):
    _emit(dict(node="request", status="escalated" if result.get("escalate") else "answered",
               category=result.get("category"), confidence=result.get("confidence"), ticket_len=len(ticket)))


def read_events():
    import pandas as pd
    if not EVENTS.exists():
        return pd.DataFrame()
    return pd.read_json(EVENTS, lines=True)
