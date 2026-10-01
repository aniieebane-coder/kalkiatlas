import streamlit as st
import numpy as np
import onnxruntime as ort

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KalkiAtlas Console",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ KALKIATLAS | Enterprise AI Console")

st.caption(
    "Sovereign Multilingual Embeddings & Context Precision Reranking Engine"
)


# ============================================================
# MODEL
# ============================================================

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_engine():

    tokenizer_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="tokenizer.json"
    )

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="onnx/model.onnx"
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)

    tokenizer.enable_truncation(
        max_length=256
    )

    tokenizer.enable_padding(
        length=256,
        pad_id=0,
        pad_token="[PAD]"
    )

    options = ort.SessionOptions()

    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    return tokenizer, session


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def encode_texts(texts):

    tokenizer, session = load_engine()

    if isinstance(texts, str):
        texts = [texts]

    encoded = tokenizer.encode_batch(texts)

    input_ids = np.array(
        [item.ids for item in encoded],
        dtype=np.int64
    )

    attention_mask = np.array(
        [item.attention_mask for item in encoded],
        dtype=np.int64
    )

    token_type_ids = np.zeros_like(
        input_ids,
        dtype=np.int64
    )

    required_inputs = {
        item.name
        for item in session.get_inputs()
    }

    inputs = {}

    if "input_ids" in required_inputs:
        inputs["input_ids"] = input_ids

    if "attention_mask" in required_inputs:
        inputs["attention_mask"] = attention_mask

    if "token_type_ids" in required_inputs:
        inputs["token_type_ids"] = token_type_ids

    outputs = session.run(
        None,
        inputs
    )

    token_embeddings = outputs[0]

    # Mean pooling
    mask = attention_mask[..., None].astype(
        np.float32
    )

    summed = np.sum(
        token_embeddings * mask,
        axis=1
    )

    counts = np.clip(
        mask.sum(axis=1),
        1e-9,
        None
    )

    embeddings = summed / counts

    # Normalize embeddings
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    embeddings = embeddings / np.clip(
        norms,
        1e-12,
        None
    )

    return embeddings.astype(np.float32)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚡ KalkiAtlas Engine")

st.sidebar.success("🟢 Status: LIVE")

st.sidebar.info("💻 Compute: CPU")

st.sidebar.info("⚙️ Runtime: ONNX")

st.sidebar.info("🧠 Model: all-MiniLM-L6-v2")

st.sidebar.divider()

st.sidebar.caption(
    "KalkiAtlas Semantic Intelligence Engine"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🎯 KalkiRerank",
        "🔍 KalkiEmbed"
    ]
)


# ============================================================
# KALKI RERANK
# ============================================================

with tab1:

    st.subheader(
        "🎯 Multilingual Context Search Reranking"
    )

    st.write(
        "Enter a query and a list of documents. "
        "KalkiAtlas will rank the most relevant documents."
    )

    query = st.text_input(
        "User Search Query",
        "Mera transaction fail hua refund kab aayega?"
    )

    docs = st.text_area(
        "Retrieved Enterprise Documents",
        """Personal loan interest rates start at 10.5% per annum.
UPI refund process takes 24 to 48 working hours to credit back.
Account KYC must be updated periodically.""",
        height=160
    )

    if st.button(
        "🚀 Execute KalkiRerank",
        type="primary"
    ):

        documents = [
            line.strip()
            for line in docs.split("\n")
            if line.strip()
        ]

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        elif not documents:

            st.warning(
                "Please enter at least one document."
            )

        else:

            try:

                with st.spinner(
                    "KalkiAtlas is analyzing semantic relevance..."
                ):

                    all_texts = [query] + documents

                    embeddings = encode_texts(
                        all_texts
                    )

                    query_embedding = embeddings[0]

                    document_embeddings = embeddings[1:]

                    scores = np.dot(
                        document_embeddings,
                        query_embedding
                    )

                    results = []

                    for index, (
                        document,
                        score
                    ) in enumerate(
                        zip(
                            documents,
                            scores
                        )
                    ):

                        results.append(
                            {
                                "document": document,
                                "relevance_score": round(
                                    float(score),
                                    4
                                ),
                                "source_index": index
                            }
                        )

                    results.sort(
                        key=lambda item:
                        item["relevance_score"],
                        reverse=True
                    )


                # ==================================================
                # PROFESSIONAL RESULT DISPLAY
                # ==================================================

                st.success(
                    "✅ Re-ranking Complete!"
                )

                st.subheader(
                    "🏆 Ranked Results"
                )

                for rank, result in enumerate(
                    results,
                    start=1
                ):

                    score = result[
                        "relevance_score"
                    ]

                    document = result[
                        "document"
                    ]


                    # Relevance classification
                    if score >= 0.60:

                        status = "🟢 High Relevance"

                    elif score >= 0.35:

                        status = "🟡 Medium Relevance"

                    else:

                        status = "🔴 Low Relevance"


                    # Result container
                    with st.container(
                        border=True
                    ):

                        col1, col2 = st.columns(
                            [1, 5]
                        )

                        with col1:

                            st.metric(
                                "Rank",
                                f"#{rank}"
                            )

                        with col2:

                            st.markdown(
                                f"### {status}"
                            )

                            st.write(
                                document
                            )

                            st.progress(
                                min(
                                    max(score, 0.0),
                                    1.0
                                )
                            )

                            st.caption(
                                f"Semantic relevance score: "
                                f"{score:.4f}"
                            )


                # ==================================================
                # RAW JSON
                # ==================================================

                with st.expander(
                    "🔧 View Developer JSON Response"
                ):

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
# KALKI EMBED
# ============================================================

with tab2:

    st.subheader(
        "🔍 Vector Embedding Generator"
    )

    st.write(
        "Convert text into a semantic vector representation."
    )

    text = st.text_area(
        "Input Text",
        "Aadhaar authentication and PAN card linking status",
        height=120
    )

    if st.button(
        "⚡ Generate Embedding",
        type="primary"
    ):

        if not text.strip():

            st.warning(
                "Please enter some text."
            )

        else:

            try:

                with st.spinner(
                    "Generating semantic embedding..."
                ):

                    vector = encode_texts(
                        text
                    )[0]


                st.success(
                    "✅ Embedding Generated!"
                )


                # ==================================================
                # EMBEDDING INFORMATION
                # ==================================================

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Vector Dimensions",
                        len(vector)
                    )

                with col2:

                    st.metric(
                        "Runtime",
                        "ONNX CPU"
                    )


                st.subheader(
                    "Vector Preview"
                )

                st.code(
                    str(
                        vector[:10].tolist()
                    ),
                    language="text"
                )


                # Full vector hidden by default
                with st.expander(
                    "View Full Embedding Vector"
                ):

                    st.json(
                        vector.tolist()
                    )


            except Exception as error:

                st.error(
                    "❌ Embedding generation failed."
                )

                st.exception(
                    error
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚡ KalkiAtlas • Semantic Search, Reranking & Vector Intelligence"
)
