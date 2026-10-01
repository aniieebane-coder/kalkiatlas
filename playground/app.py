import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="KalkiAtlas Console",
    layout="wide"
)

st.title("⚡ KALKIATLAS | Enterprise AI Console")
st.caption(
    "Sovereign Multilingual Embeddings & Context Precision Reranking Engine"
)

@st.cache_resource
def load_engine():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

try:
    with st.spinner("Loading AI model... First startup may take a few minutes."):
        model = load_engine()

    st.success("AI engine loaded successfully.")

except Exception as e:
    st.error("AI engine failed to load.")
    st.exception(e)
    st.stop()
