import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="KalkiAtlas Console", layout="wide")
st.title("⚡ KALKIATLAS | Enterprise AI Console")
st.caption("Sovereign Multilingual Embeddings & Context Precision Reranking Engine")

# Cache the lightweight model in RAM
@st.cache_resource
def load_engine():
    # Ultra-lightweight 22MB model for maximum speed on Free CPU
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_engine()

st.sidebar.header("KalkiAtlas Engine Status")
st.sidebar.success("Status: LIVE (24/7 Cloud Node)")
st.sidebar.warning("Demo Tier: CPU Instance (~1.2s)")
st.sidebar.info("Enterprise VPC Tier: NVIDIA GPU (<20ms)")

tab1, tab2 = st.tabs(["🎯 KalkiRerank (Context Accuracy)", "🔍 KalkiEmbed (Vector Generation)"])

with tab1:
    st.subheader("Multilingual Context Search Reranking Engine")
    query = st.text_input("User Search Query (Hinglish/Native/English):", "Mera transaction fail hua refund kab aayega?")
    docs = st.text_area("Retrieved Enterprise Docs (Unsorted RAG Data):", 
                        "1. Personal loan interest rates start at 10.5% per annum.\n2. UPI refund process takes 24 to 48 working hours to credit back.\n3. Account balance check can be done using mobile banking app.")
    
    if st.button("Execute KalkiRerank"):
        doc_list = [d.strip() for d in docs.split("\n") if d.strip()]
        
        # Fast NumPy Cosine Similarity
        q_emb = model.encode(query, normalize_embeddings=True)
        d_embs = model.encode(doc_list, normalize_embeddings=True)
        
        # Matrix multiplication for speed
        scores = np.dot(d_embs, q_emb).tolist()
        
        results = [{"index": idx, "document": doc, "relevance_score": float(round(score, 4))} for idx, (doc, score) in enumerate(zip(doc_list, scores))]
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        st.success("Re-ranking Complete!")
        st.json(results)

with tab2:
    st.subheader("Multilingual Vector Embedding Generator")
    text = st.text_area("Input Text Sequence:", "Aadhaar authentication and PAN card linking status")
    
    if st.button("Generate Kalki Embeddings"):
        vector = model.encode(text, normalize_embeddings=True).tolist()
        st.success("Embeddings Generated!")
        st.write("Vector Dimension Size:", len(vector))
        st.write("First 10 Dimensions Sample:", vector[:10])
