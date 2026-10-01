import streamlit as st
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="KalkiAtlas Console",
    layout="wide"
)

st.title("⚡ KALKIATLAS | Enterprise AI Console")

st.caption(
    "Sovereign Multilingual Embeddings & Context Precision Reranking Engine"
)


# -----------------------------
# MODEL LOADER
# -----------------------------
# IMPORTANT:
# The model is NOT loaded when the website opens.
# It loads only after the user clicks a button.
@st.cache_resource(show_spinner=False)
def load_engine():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    return model


def get_model():
    try:
        with st.spinner("Loading AI engine..."):
            model = load_engine()

        return model

    except Exception as e:
        st.error("AI model failed to load.")
        st.exception(e)
        st.stop()


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("KalkiAtlas Engine Status")

st.sidebar.success("Status: LIVE")
st.sidebar.info("Compute: CPU")
st.sidebar.info("Model: all-MiniLM-L6-v2")


# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(
    [
        "🎯 KalkiRerank (Context Accuracy)",
        "🔍 KalkiEmbed (Vector Generation)"
    ]
)


# ============================================================
# TAB 1 - RERANK
# ============================================================

with tab1:

    st.subheader(
        "Multilingual Context Search Reranking Engine"
    )

    query = st.text_input(
        "User Search Query:",
        "Mera transaction fail hua refund kab aayega?"
    )

    docs = st.text_area(
        "Retrieved Enterprise Docs:",
        """1. Personal loan interest rates start at 10.5% per annum.
2. UPI refund process takes 24 to 48 working hours to credit back.
3. Account KYC must be updated periodically."""
    )

    if st.button("Execute KalkiRerank"):

        doc_list = [
            d.strip()
            for d in docs.split("\n")
            if d.strip()
        ]

        if not query.strip():
            st.warning("Please enter a search query.")

        elif not doc_list:
            st.warning("Please enter at least one document.")

        else:

            # MODEL LOADS HERE
            # Not when the website opens
            model = get_model()

            with st.spinner("Calculating relevance scores..."):

                q_emb = model.encode(
                    query,
                    normalize_embeddings=True
                )

                d_embs = model.encode(
                    doc_list,
                    normalize_embeddings=True
                )

                scores = np.dot(
                    d_embs,
                    q_emb
                ).tolist()

                results = []

                for idx, (doc, score) in enumerate(
                    zip(doc_list, scores)
                ):

                    results.append(
                        {
                            "index": idx,
                            "document": doc,
                            "relevance_score": float(
                                round(score, 4)
                            )
                        }
                    )

                results.sort(
                    key=lambda x: x["relevance_score"],
                    reverse=True
                )

            st.success("Re-ranking Complete!")

            st.json(results)


# ============================================================
# TAB 2 - EMBEDDINGS
# ============================================================

with tab2:

    st.subheader(
        "Multilingual Vector Embedding Generator"
    )

    text = st.text_area(
        "Input Text Sequence:",
        "Aadhaar authentication and PAN card linking status"
    )

    if st.button("Generate Kalki Embeddings"):

        if not text.strip():

            st.warning(
                "Please enter some text."
            )

        else:

            # MODEL LOADS HERE
            model = get_model()

            with st.spinner(
                "Generating embedding..."
            ):

                vector = model.encode(
                    text,
                    normalize_embeddings=True
                ).tolist()

            st.success(
                "Embeddings Generated!"
            )

            st.write(
                "Vector Dimension:",
                len(vector)
            )

            st.write(
                "First 10 Dimensions:",
                vector[:10]
            )
