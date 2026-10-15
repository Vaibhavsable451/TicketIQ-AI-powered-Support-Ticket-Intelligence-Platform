import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from src.config import DATA_PROCESSED, FIG_DIR, GROQ_API_KEY, LANGSMITH_API_KEY, MODELS_DIR, PINECONE_API_KEY, REPORTS_DIR

st.set_page_config(page_title="TicketIQ", page_icon="🎫", layout="wide")
st.title("🎫 TicketIQ - Support Ticket Intelligence")

ready = (MODELS_DIR / "classifier.joblib").exists()
with st.sidebar:
    st.header("System status")
    st.write("Models trained:", "✅" if ready else "❌ run `python -m src.pipeline`")
    st.write("Priority model:", "✅" if (MODELS_DIR / "priority.joblib").exists() else "❌ retrain")
    st.write("Groq LLM:", "✅" if GROQ_API_KEY else "⚠️ offline fallback")
    st.write("Pinecone:", "✅" if PINECONE_API_KEY else "⚠️ in-memory store")
    st.write("LangSmith tracing:", "✅" if LANGSMITH_API_KEY else "⚠️ off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Triage assistant", "📦 Batch CSV", "🏆 Model leaderboard",
                                        "📊 Data explorer", "🔭 Observability"])

# ---------------------------------------------------------------- single ticket
with tab1:
    if not ready:
        st.warning("Train the models first: `python -m src.pipeline --skip-deep`")
    else:
        c1, c2 = st.columns([2, 1])
        ticket = c1.text_area("Customer ticket", "URGENT: I was charged twice for my Pro subscription!", height=140)
        channel = c2.selectbox("Channel", ["email", "chat", "phone", "web"])
        tenure = c2.slider("Tenure (months)", 0, 72, 12)
        prior = c2.slider("Prior tickets", 0, 10, 1)
        if st.button("Analyze ticket", type="primary") and ticket.strip():
            from src.rag.graph import handle_ticket
            with st.spinner("Running LangGraph agent..."):
                st.session_state.result = handle_ticket(ticket, channel=channel, tenure_months=tenure, prior_tickets=prior)

        r = st.session_state.get("result")
        if r:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Category", r["category"]); m2.metric("Confidence", f"{r['confidence']:.0%}")
            m3.metric("Priority", r.get("priority", "n/a")); m4.metric("Sentiment", r.get("sentiment_label", "n/a"))
            m5.metric("Predicted resolution", f"{r['eta_hours']:.1f} h")
            if r.get("cached"):
                st.caption("⚡ Served from cache")
            if r.get("pii_found"):
                st.info("🔒 Personal data redacted before LLM call: " + ", ".join(r["pii_found"]))
            (st.error if r["escalate"] else st.success)(f"{'Escalated' if r['escalate'] else 'Auto-resolved'}: {r['reason']}")

            st.subheader("Suggested reply"); st.write(r["answer"])
            if r.get("draft_answer"):
                with st.expander("Draft the verifier rejected"):
                    st.write(r["draft_answer"]); st.caption(f"Verifier said: {r.get('verdict', 'n/a')}")
            if r.get("sources"):
                st.caption("Sources: " + ", ".join(r["sources"]))

            if r.get("top_terms"):
                with st.expander("🔍 Why this category? (top words)"):
                    st.bar_chart(pd.DataFrame(r["top_terms"], columns=["word", "weight"]).set_index("word"))
            if r.get("duplicates"):
                with st.expander(f"♻️ {len(r['duplicates'])} similar past ticket(s)"):
                    st.dataframe(pd.DataFrame(r["duplicates"]), use_container_width=True)

            from src.feedback import save_feedback
            st.write("Was this helpful?")
            f1, f2, _ = st.columns([1, 1, 8])
            if f1.button("👍"):
                save_feedback(r.get("ticket_safe", ""), r["category"], r["answer"], True); st.toast("Thanks for the feedback!")
            if f2.button("👎"):
                save_feedback(r.get("ticket_safe", ""), r["category"], r["answer"], False); st.toast("Feedback saved.")

# ---------------------------------------------------------------- batch
with tab2:
    st.write("Upload a CSV with a **text** column (optional: channel, tenure_months, prior_tickets).")
    up = st.file_uploader("CSV file", type="csv")
    if up and ready:
        from src.batch import predict_batch
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("CSV needs a 'text' column.")
        else:
            out = predict_batch(df)
            st.dataframe(out, use_container_width=True)
            st.bar_chart(out["category"].value_counts())
            st.metric("Needs human review", int(out["needs_review"].sum()))
            st.download_button("⬇️ Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------------------------------------------------------- leaderboard
with tab3:
    mp = REPORTS_DIR / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        st.subheader("Classification"); st.dataframe(pd.DataFrame(m["classification"]), use_container_width=True)
        st.subheader("Regression (resolution hours)"); st.dataframe(pd.DataFrame(m["regression"]), use_container_width=True)
        if "priority" in m:
            st.subheader("Priority model"); st.json(m["priority"])
        cols = st.columns(2)
        for i, f in enumerate(["clf_leaderboard.png", "confusion_matrix.png", "top_terms.png", "regression_fit.png", "mlp_training.png"]):
            if (FIG_DIR / f).exists():
                cols[i % 2].image(str(FIG_DIR / f), caption=f)
    else:
        st.info("No metrics yet.")

# ---------------------------------------------------------------- data
with tab4:
    tp = DATA_PROCESSED / "train.csv"
    if tp.exists():
        df = pd.read_csv(tp)
        st.dataframe(df.head(50), use_container_width=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        sns.countplot(data=df, x="category", ax=ax[0]); sns.boxplot(data=df, x="category", y="resolution_hours", ax=ax[1])
        st.pyplot(fig)
    else:
        st.info("No processed data yet.")

# ---------------------------------------------------------------- observability
with tab5:
    from src.drift import drift_report
    from src.feedback import load_feedback
    from src.observability import read_events

    d = drift_report()
    st.subheader("Drift & health")
    if d is None:
        st.info("Analyze some tickets to populate the monitor.")
    else:
        (st.error if d["alert"] else st.success)("⚠️ Drift / escalation alert" if d["alert"] else "✅ Traffic looks normal")
        a, b, c, e = st.columns(4)
        a.metric("Requests", d["n"]); b.metric("Category drift (TVD)", f"{d['tvd']:.2f}")
        c.metric("Escalation rate", f"{d['escalation_rate']:.0%}"); e.metric("Avg confidence", f"{d['avg_conf']:.0%}")
        st.bar_chart(d["table"])
        st.caption("Alert needs 20+ requests and TVD > 0.3 or escalation rate > 40%.")

    fb = load_feedback()
    if not fb.empty:
        st.subheader("User feedback")
        st.metric("Helpful rate", f"{fb.helpful.mean():.0%}", f"{len(fb)} votes")

    ev = read_events()
    if not ev.empty:
        st.subheader("Pipeline latency")
        st.bar_chart(ev[ev.node != "request"].groupby("node").latency_ms.mean())
        st.metric("Errors", int((ev.status == "error").sum()))
        st.dataframe(ev.tail(50), use_container_width=True)