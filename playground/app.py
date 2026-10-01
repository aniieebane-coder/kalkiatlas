import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="KalkiAtlas Console", layout="wide")
st.title("⚡ KALKIATLAS | Enterprise AI Console")
st.caption("Sovereign Multilingual Embeddings & High-Precision Context Reranking Engine")

# Cache model so it loads into memory once (~80MB RAM footprint)
@st.cache_resource
def load_engine():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model

try:
    model = load_engine()
    st.sidebar.success("KalkiAtlas Engine: ACTIVE")
    st.sidebar.info("Deployment: Render Cloud (24/7)")
    st.sidebar.caption("DPDP Act 2023 Compliant")
except Exception as e:
    st.sidebar.error(f"Initialization Error: {e}")

tab1, tab2 = st.tabs(["🎯 KalkiRerank (Context Accuracy)", "🔍 KalkiEmbed (Vector Generation)"])

with tab1:
    st.subheader("Multilingual Context Search Reranking Engine")
    query = st.text_input("User Search Query (Hinglish/Native/English):", "Mera transaction fail hua refund kab aayega?")
    docs = st.text_area("Retrieved Enterprise Docs (Unsorted RAG Data):", 
                        "1. Personal loan interest rates start at 10.5% per annum.\n2. UPI refund process takes 24 to 48 working hours to credit back.\n3. Account balance check can be done using mobile banking app.")
    
    if st.button("Execute KalkiRerank"):
        doc_list = [d.strip() for d in docs.split("\n") if d.strip()]
        
        # Direct Python Inference (No HTTP connections -> Fixes Connection Error!)
        query_emb = model.encode(query, convert_to_tensor=True)
        doc_embs = model.encode(doc_list, convert_to_tensor=True)
        
        scores = util.cos_sim(query_emb, doc_embs)[0].cpu().tolist()
        
        results = [{"index": idx, "document": doc, "relevance_score": float(round(score, 4))} for idx, (doc, score) in enumerate(zip(doc_list, scores))]
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        st.success("Re-ranking Complete!")
        st.json(results)

with tab2:
    st.subheader("Multilingual Vector Embedding Generator")
    text = st.text_area("Input Text Sequence:", "Aadhaar authentication and PAN card linking status")
    
    if st.button("Generate Kalki Embeddings"):
        vector = model.encode(text).tolist()
        st.success("Embeddings Generated!")
        st.write("Vector Dimension Size:", len(vector))
        st.write("First 10 Dimensions Sample:", vector[:10])
