# app.py
import streamlit as st
from retrieval_temp import retrieve_documents

st.title("SwiftVisa Demo (Temporary)")

query = st.text_input("Ask about any visa:")

if query:
    results = retrieve_documents(query)
    if not results:
        st.warning("No documents found. Try another query.")
    else:
        for i, doc in enumerate(results, 1):
            st.subheader(f"{i}. {doc.country} — {doc.visa_type}")
            st.write(doc.content)
