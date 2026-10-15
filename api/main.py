"""Feature 9: REST API.  Run:  uvicorn api.main:app --reload   (docs at http://localhost:8000/docs)"""
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.batch import predict_batch
from src.rag.graph import handle_ticket

app = FastAPI(title="TicketIQ API", version="1.0")


class TicketIn(BaseModel):
    text: str
    channel: str = "email"
    tenure_months: int = 12
    prior_tickets: int = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage")
def triage(t: TicketIn):
    if not t.text.strip():
        raise HTTPException(400, "text is empty")
    r = handle_ticket(t.text, channel=t.channel, tenure_months=t.tenure_months, prior_tickets=t.prior_tickets)
    r.pop("context", None)
    return r


@app.post("/batch")
def batch(tickets: List[TicketIn]):
    return predict_batch(pd.DataFrame([t.model_dump() for t in tickets])).to_dict("records")