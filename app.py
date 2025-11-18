import streamlit as st
import pickle as pk
import nltk
from nltk.corpus import stopwords
import numpy as np
import sqlite3
from datetime import datetime
import pandas as pd

# ---------------- NLTK ----------------
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')

download_nltk_data()
STOP_WORDS = set(stopwords.words('english'))

# ---------------- Text Cleaning ----------------
def clean_review(review: str) -> str:
    words = review.split()
    cleaned = [w for w in words if w.lower() not in STOP_WORDS]
    return " ".join(cleaned)

# ---------------- Load Model & Vectorizer ----------------
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pk.load(f)

with open('model.pkl', 'rb') as f:
    model = pk.load(f)

# Label mapping based on your training code: 'pos' -> 1, 'neg' -> 0
LABEL_MAP = {0: "Negative", 1: "Positive"}
EMOJI_MAP = {"Positive": "👍", "Negative": "👎"}
COLOR_MAP = {"Positive": "green", "Negative": "red"}

# ---------------- Database (SQLite) ----------------
@st.cache_resource
def get_connection():
    # Creates reviews.db in the same folder as app.py
    conn = sqlite3.connect("reviews.db", check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS review_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            original_text TEXT,
            cleaned_text TEXT,
            sentiment_label TEXT,
            confidence REAL
        )
        """
    )
    conn.commit()
    return conn

conn = get_connection()

def save_review_to_db(original_text, cleaned_text, sentiment_label, confidence):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO review_history (created_at, original_text, cleaned_text, sentiment_label, confidence)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(timespec="seconds"),
            original_text,
            cleaned_text,
            sentiment_label,
            float(confidence) if confidence is not None else None,
        ),
    )
    conn.commit()

def load_history_from_db(limit=200):
    query = """
        SELECT id, created_at, sentiment_label, confidence, original_text, cleaned_text
        FROM review_history
        ORDER BY id DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    return df

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="🎬",
    layout="wide"
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚙️ Options")
    show_debug = st.toggle("Show debug info", value=False)

    st.write("---")
    st.subheader("Example reviews")
    examples = {
        "Very Positive":
            "This movie was absolutely fantastic! Great acting and an amazing story.",
        "Mixed":
            "The movie was okay. Some scenes were good, but overall it was average.",
        "Very Negative":
            "Terrible movie. The plot was stupid and the acting was awful."
    }
    choice = st.selectbox("Load an example:", ["None"] + list(examples.keys()))

# ---------------- Layout: Tabs ----------------
tab_analyze, tab_dashboard = st.tabs(["🔍 Analyze Review", "📊 User Dashboard"])

# ======================= TAB 1: ANALYZE =======================
with tab_analyze:
    st.markdown(
        """
        <h1 style="text-align:center;">🎬 Movie Review Sentiment Analyzer</h1>
        <p style="text-align:center;">
            Paste any movie review below and I'll classify it as 
            <b>Positive</b> or <b>Negative</b>.
        </p>
        """,
        unsafe_allow_html=True
    )

    default_text = "Write your review here..."
    if choice != "None":
        default_text = examples[choice]

    user_input = st.text_area(
        "📝 Your Review",
        value=default_text,
        height=160
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        classify = st.button("🔍 Classify Sentiment", use_container_width=True)
    with col2:
        clear = st.button("🧹 Clear", use_container_width=True)

    if clear:
        st.experimental_rerun()

    if classify:
        raw_text = user_input.strip()

        if not raw_text or raw_text == "Write your review here...":
            st.warning("Please enter a review to classify.")
        else:
            # ----- Preprocess -----
            cleaned_input = clean_review(raw_text)
            processed_input = vectorizer.transform([cleaned_input])

            # ----- Predict -----
            prediction = model.predict(processed_input)
            pred_label = int(prediction[0])      # 0 or 1
            sentiment = LABEL_MAP.get(pred_label, "Unknown")
            emoji = EMOJI_MAP.get(sentiment, "❓")
            color = COLOR_MAP.get(sentiment, "gray")

            # Try to get confidence
            try:
                proba = model.predict_proba(processed_input)[0]
                confidence = float(np.max(proba))
            except Exception:
                proba = None
                confidence = None

            # ----- Save to DB -----
            save_review_to_db(
                original_text=raw_text,
                cleaned_text=cleaned_input,
                sentiment_label=sentiment,
                confidence=confidence,
            )

            # ----- Show Result -----
            st.markdown("---")
            st.markdown(
                f"""
                <div style="
                    border-radius: 12px;
                    padding: 18px 20px;
                    border: 1px solid #e0e0e0;
                    background-color: #f9f9f9;
                ">
                    <h3 style="margin-bottom:6px;">
                        Prediction: <span style="color:{color};">{sentiment}</span> {emoji}
                    </h3>
                """,
                unsafe_allow_html=True
            )

            if confidence is not None:
                st.markdown(
                    f"""
                    <p style="margin-top:0;">
                        Model confidence: <b>{confidence*100:.1f}%</b>
                    </p>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(min(max(confidence, 0.0), 1.0))

            st.markdown("</div>", unsafe_allow_html=True)

            # ----- Show Details -----
            st.markdown("### 🔎 Input Details")
            st.markdown("**Original Review:**")
            st.write(raw_text)

            st.markdown("**Cleaned Review:**")
            st.code(cleaned_input, language="text")

            if show_debug:
                st.markdown("### 🧪 Debug Info")
                st.write("Raw prediction:", prediction)
                st.write("Model classes_:", getattr(model, "classes_", "N/A"))
                if proba is not None:
                    prob_dict = {str(cls): float(p) for cls, p in zip(model.classes_, proba)}
                    st.write("Class probabilities:", prob_dict)

# ======================= TAB 2: DASHBOARD =======================
with tab_dashboard:
    st.markdown("## 📊 Review History Dashboard")

    df_history = load_history_from_db(limit=500)

    if df_history.empty:
        st.info("No reviews have been classified yet. Go to the *Analyze Review* tab and run a few predictions first.")
    else:
        # --------- Filters ----------
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            sentiments_selected = st.multiselect(
                "Filter by sentiment:",
                options=["Positive", "Negative"],
                default=["Positive", "Negative"]
            )

        if sentiments_selected:
            df_filtered = df_history[df_history["sentiment_label"].isin(sentiments_selected)]
        else:
            df_filtered = df_history.copy()

        # --------- Summary cards ----------
        total_reviews = len(df_history)
        total_filtered = len(df_filtered)
        pos_count = (df_filtered["sentiment_label"] == "Positive").sum()
        neg_count = (df_filtered["sentiment_label"] == "Negative").sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Reviews (All Time)", total_reviews)
        with c2:
            st.metric("Positive (filtered)", pos_count)
        with c3:
            st.metric("Negative (filtered)", neg_count)

        # --------- Table ----------
        st.markdown("### 🧾 Recent Reviews")
        st.dataframe(
            df_filtered[["id", "created_at", "sentiment_label", "confidence", "original_text"]],
            use_container_width=True,
            hide_index=True
        )

        # Optional: show details of a selected row
        st.markdown("### 🔍 Inspect a Single Review")
        selected_id = st.number_input(
            "Enter review ID to inspect:",
            min_value=int(df_filtered["id"].min()),
            max_value=int(df_filtered["id"].max()),
            value=int(df_filtered["id"].max()),
            step=1
        )

        selected_row = df_filtered[df_filtered["id"] == selected_id]
        if not selected_row.empty:
            row = selected_row.iloc[0]
            st.markdown(f"**ID:** {row['id']}")
            st.markdown(f"**Time (UTC):** {row['created_at']}")
            st.markdown(f"**Sentiment:** {row['sentiment_label']}")
            st.markdown(f"**Confidence:** {row['confidence']:.3f}" if not pd.isna(row['confidence']) else "**Confidence:** N/A")

            st.markdown("**Original Review:**")
            st.write(row["original_text"])

            st.markdown("**Cleaned Review:**")
            st.code(row["cleaned_text"], language="text")
