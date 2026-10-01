import streamlit as st
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KalkiAtlas Console",
    layout="wide"
)

st.title("⚡ KALKIATLAS | Enterprise AI Console")

st.caption(
    "Sovereign Multilingual Embeddings & Context Precision Reranking Engine"
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_engine():

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    return model


def get_model():

    with st.spinner("Loading AI engine..."):
        model = load_engine()

    return model


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("KalkiAtlas Engine Status")

st.sidebar.success(
    "Status: LIVE"
)

st.sidebar.info(
    "Compute: CPU"
)

st.sidebar.info(
    "Model: all-MiniLM-L6-v2"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🎯 KalkiRerank (Context Accuracy)",
        "🔍 KalkiEmbed (Vector Generation)"
    ]
)


# ============================================================
# TAB 1
# KALKI RERANK
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
3. Account KYC must be updated periodically.""",
        height=130
    )

    if st.button(
        "Execute KalkiRerank",
        key="rerank_button"
    ):

        doc_list = [
            document.strip()
            for document in docs.split("\n")
            if document.strip()
        ]

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        elif not doc_list:

            st.warning(
                "Please enter at least one document."
            )

        else:

            try:

                # -----------------------------------------
                # STEP 1
                # -----------------------------------------

                st.info(
                    "Step 1: Starting model load..."
                )

                model = get_model()


                # -----------------------------------------
                # STEP 2
                # -----------------------------------------

                st.success(
                    "Step 2: Model loaded successfully!"
                )


                # -----------------------------------------
                # STEP 3
                # -----------------------------------------

                st.info(
                    "Step 3: Encoding query..."
                )

                q_emb = model.encode(
                    query,
                    normalize_embeddings=True
                )


                # -----------------------------------------
                # STEP 4
                # -----------------------------------------

                st.success(
                    "Step 4: Query encoded!"
                )


                # -----------------------------------------
                # STEP 5
                # -----------------------------------------

                st.info(
                    "Step 5: Encoding documents..."
                )

                d_embs = model.encode(
                    doc_list,
                    normalize_embeddings=True
                )


                # -----------------------------------------
                # STEP 6
                # -----------------------------------------

                st.success(
                    "Step 6: Documents encoded!"
                )


                # -----------------------------------------
                # CALCULATE SIMILARITY
                # -----------------------------------------

                scores = np.dot(
                    d_embs,
                    q_emb
                ).tolist()


                # -----------------------------------------
                # BUILD RESULTS
                # -----------------------------------------

                results = []

                for idx, (document, score) in enumerate(
                    zip(doc_list, scores)
                ):

                    results.append(
                        {
                            "rank_source_index": idx,
                            "document": document,
                            "relevance_score": float(
                                round(score, 4)
                            )
                        }
                    )


                # -----------------------------------------
                # SORT RESULTS
                # -----------------------------------------

                results.sort(
                    key=lambda item:
                    item["relevance_score"],
                    reverse=True
                )


                # -----------------------------------------
                # DISPLAY RESULTS
                # -----------------------------------------

                st.success(
                    "✅ Re-ranking Complete!"
                )

                st.json(
                    results
                )


            except Exception as error:

                st.error(
                    "❌ KalkiRerank failed."
                )

                st.exception(
                    error
                )


# ============================================================
# TAB 2
# KALKI EMBEDDINGS
# ============================================================

with tab2:

    st.subheader(
        "Multilingual Vector Embedding Generator"
    )

    text = st.text_area(
        "Input Text Sequence:",
        "Aadhaar authentication and PAN card linking status",
        key="embedding_text"
    )

    if st.button(
        "Generate Kalki Embeddings",
        key="embedding_button"
    ):

        if not text.strip():

            st.warning(
                "Please enter some text."
            )

        else:

            try:

                # -----------------------------------------
                # STEP 1
                # -----------------------------------------

                st.info(
                    "Step 1: Starting model load..."
                )

                model = get_model()


                # -----------------------------------------
                # STEP 2
                # -----------------------------------------

                st.success(
                    "Step 2: Model loaded successfully!"
                )


                # -----------------------------------------
                # STEP 3
                # -----------------------------------------

                st.info(
                    "Step 3: Generating embedding..."
                )

                vector = model.encode(
                    text,
                    normalize_embeddings=True
                ).tolist()


                # -----------------------------------------
                # RESULTS
                # -----------------------------------------

                st.success(
                    "✅ Embedding Generated!"
                )

                st.write(
                    "Vector Dimension:",
                    len(vector)
                )

                st.write(
                    "First 10 Dimensions:",
                    vector[:10]
                )


            except Exception as error:

                st.error(
                    "❌ Embedding generation failed."
                )

                st.exception(
                    error
                )
