# 🎫 TicketIQ - Support Ticket Intelligence Platform

One end-to-end ML project that covers the full lifecycle: **data collection -> preprocessing -> NLP feature engineering -> classical ML -> deep learning -> fine-tuning -> RAG agent -> Streamlit deployment -> observability**.

**Business problem:** For every incoming support ticket, (1) classify its category, (2) predict resolution time, (3) draft a grounded reply from the knowledge base, and (4) escalate to a human when the system is unsure.

## Stack
| Layer | Tools |
|---|---|
| Data & plots | pandas, NumPy, Matplotlib, Seaborn |
| NLP / features | regex cleaning, TF-IDF (1-2 grams), engineered numeric features, one-hot |
| Classification | Logistic Regression, KNN, SVM, Decision Tree, Random Forest (+ tuning), Gradient Boosting |
| Regression | Linear, Ridge, Random Forest, Gradient Boosting (log-target) |
| Deep learning | PyTorch MLP (BatchNorm, dropout, early stopping, LR scheduler) |
| Fine-tuning | DistilBERT via Hugging Face Transformers |
| LLM / agent | LangChain + LangGraph, Groq (Llama 3.3 70B) |
| Vector DB | Pinecone serverless + MiniLM embeddings (in-memory fallback) |
| Frontend | Streamlit |
| Observability | LangSmith tracing, JSONL node-latency events, logging |

## Architecture
```
tickets -> clean -> features -> models (clf + reg) -> models/*.joblib
                                         |
ticket -> [classify] --low conf--> [escalate]
              | ok                      ^
          [retrieve: Pinecone] -> [draft: Groq] -> [verify groundedness] -> [respond]
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add GROQ / PINECONE / LANGSMITH keys

python -m src.pipeline --skip-deep      # fast: data + classical models (~30 s)
python -m src.pipeline --tune           # + RF hyperparameter search + PyTorch MLP
python -m src.pipeline --finetune       # + DistilBERT fine-tuning (GPU recommended)
python -m src.rag.vector_store          # (optional) push KB to Pinecone
streamlit run app/streamlit_app.py      # launch UI
jupyter notebook notebooks/01_end_to_end.ipynb
```

## API keys (all optional - the app degrades gracefully)
- **Groq** - https://console.groq.com -> `GROQ_API_KEY`. Without it the agent returns a knowledge-base excerpt.
- **Pinecone** - https://app.pinecone.io -> `PINECONE_API_KEY`. Index is auto-created (384-dim, cosine). Without it an in-memory store is used.
- **LangSmith** - https://smith.langchain.com -> `LANGSMITH_API_KEY` for full traces of every graph run.
- Never commit `.env` (add it to `.gitignore`).

## Results (synthetic data, 4,000 tickets, test split 20%)
GradientBoosting ≈ 0.99 macro-F1; Logistic Regression ≈ 0.985; KNN weakest (~0.69) because cosine distance mixes sparse TF-IDF and scaled numeric features. Resolution-time R² ≈ 0.82.
**Caveat:** synthetic data is template-based so scores are optimistic. Use `--source hf` (Bitext dataset) or your own CSV (`text, category, channel, tenure_months, prior_tickets, resolution_hours`) for realistic numbers.

## Project layout
```
src/config.py            paths, seeds, keys
src/data/                collect.py (synthetic/HF/KB), preprocess.py
src/features/            engineering.py (ColumnTransformer)
src/models/              classical.py, deep.py, finetune.py, evaluate.py
src/rag/                 vector_store.py (Pinecone), graph.py (LangGraph)
src/observability.py     LangSmith + JSONL metrics
src/pipeline.py          training orchestrator
app/streamlit_app.py     frontend
notebooks/               Jupyter walkthrough
tests/                   pytest unit tests
```

## Production "huge advice" checklist
1. **Prevent train/serve skew** - `prepare_frame()` is shared by training and inference; keep it that way.
2. **Evaluate on a time-based split** with real data, and track per-class recall, not just accuracy.
3. **Calibrate probabilities** (`CalibratedClassifierCV`) before trusting the 0.55 escalation threshold.
4. **Version everything** - add MLflow or DVC for data/model versions; pin dependency versions.
5. **Guardrails** - the verify node blocks ungrounded answers; add PII redaction before sending text to any LLM.
6. **Evaluate RAG separately** - build a labelled Q&A set; measure retrieval hit-rate and answer faithfulness (LangSmith datasets/evals).
7. **Monitor drift** - compare live category distribution and confidence histograms with training data; alert on rising escalation rate.
8. **Deploy** - containerise (Docker), serve the graph behind FastAPI, keep Streamlit as thin UI, store secrets in a secret manager, cache embeddings, add rate limits and retries around Groq/Pinecone.
9. **Cost/latency** - use the small model for verify, batch embeddings, and set `max_tokens`.
