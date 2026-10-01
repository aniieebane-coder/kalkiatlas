import streamlit as st
import numpy as np
import onnxruntime as ort

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


st.set_page_config(
    page_title="KalkiAtlas Console",
    layout="wide"
)

st.title("⚡ KALKIATLAS | Enterprise AI Console")

st.caption(
    "Sovereign Multilingual Embeddings & Context Precision Reranking Engine"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD ONNX ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_engine():

    # Download tokenizer
    tokenizer_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="tokenizer.json"
    )

    # Download ONNX model
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="onnx/model.onnx"
    )

    tokenizer = Tokenizer.from_file(
        tokenizer_path
    )

    tokenizer.enable_truncation(
        max_length=256
    )

    tokenizer.enable_padding(
        length=256,
        pad_id=0,
        pad_token="[PAD]"
    )

    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
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

    # Determine what inputs the ONNX model expects
    model_inputs = {
        item.name
        for item in session.get_inputs()
    }

    inputs = {}

    if "input_ids" in model_inputs:
        inputs["input_ids"] = input_ids

    if "attention_mask" in model_inputs:
        inputs["attention_mask"] = attention_mask

    if "token_type_ids" in model_inputs:
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

    # L2 normalization
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

    return embeddings.astype(
        np.float32
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "KalkiAtlas Engine Status"
)

st.sidebar.success(
    "Status: LIVE"
)

st.sidebar.info(
    "Compute: CPU"
)

st.sidebar.info(
    "Runtime: ONNX"
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
3. Account KYC must be updated periodically.""",
        height=130
    )

    if st.button(
        "Execute KalkiRerank"
    ):

        documents = [
            line.strip()
            for line in docs.split("\n")
            if line.strip()
        ]

        if not query.strip():

            st.warning(
                "Please enter a query."
            )

        elif not documents:

            st.warning(
                "Please enter at least one document."
            )

        else:

            try:

                with st.spinner(
                    "Running KalkiRerank..."
                ):

                    # Encode query and docs together
                    all_text = [
                        query
                    ] + documents

                    embeddings = encode_texts(
                        all_text
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

                st.success(
                    "✅ Re-ranking Complete!"
                )

                st.json(
                    results
                )

            except Exception as error:

                st.error(
                    "KalkiRerank failed."
                )

                st.exception(
                    error
                )


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

    if st.button(
        "Generate Kalki Embeddings"
    ):

        if not text.strip():

            st.warning(
                "Please enter text."
            )

        else:

            try:

                with st.spinner(
                    "Generating embedding..."
                ):

                    vector = encode_texts(
                        text
                    )[0]

                st.success(
                    "✅ Embedding Generated!"
                )

                st.write(
                    "Vector Dimension:",
                    len(vector)
                )

                st.write(
                    "First 10 Dimensions:"
                )

                st.write(
                    vector[:10].tolist()
                )

            except Exception as error:

                st.error(
                    "Embedding generation failed."
                )

                st.exception(
                    error
                )
