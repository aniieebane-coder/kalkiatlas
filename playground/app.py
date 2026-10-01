import streamlit as st, requests

st.set_page_config(page_title="KalkiAtlas Console", layout="wide")
st.title("⚡ KALKIATLAS | Enterprise AI Console")

API_URL = "http://localhost:8000/v1"
HEADERS = {"X-Kalki-Key": "sk-kalki-live-secret-key"}

tab1, tab2 = st.tabs(["🎯 KalkiRerank", "🔍 KalkiEmbed"])

with tab1:
    q = st.text_input("Search Query:", "Mera transaction fail hua refund kab aayega?")
    docs = st.text_area("Docs:", "1. Loan rates start at 10.5%.\n2. UPI refund takes 24-48 working hours.\n3. Check balance via app.")
    if st.button("Execute KalkiRerank"):
        d_list = [d.strip() for d in docs.split("\n") if d.strip()]
        try:
            res = requests.post(f"{API_URL}/rerank", headers=HEADERS, json={"query": q, "documents": d_list, "top_n": 2}, timeout=120)
            if res.status_code == 200: 
                st.json(res.json())
            else:
                st.error(f"Error: {res.text}")
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")

with tab2:
    txt = st.text_area("Text:", "Aadhaar authentication and PAN linking")
    if st.button("Generate Embeddings"):
        try:
            res = requests.post(f"{API_URL}/embed", headers=HEADERS, json={"texts": [txt]}, timeout=120)
            if res.status_code == 200: 
                st.json(res.json())
            else:
                 st.error(f"Error: {res.text}")
        except Exception as e:
             st.error(f"Connection Error: {str(e)}")
