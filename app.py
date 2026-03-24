# app.py
import streamlit as st
from retrieval import retrieve_documents

st.set_page_config(page_title="SwiftVisa", layout="centered")
st.title("🌏 SwiftVisa AI - Visa Eligibility Helper")

query = st.text_input("Ask about a visa (e.g., 'Student visa for Germany'):")

if query:
    results = retrieve_documents(query)

    if results:
        st.subheader("Relevant Information:")
        for i, doc in enumerate(results, 1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.page_content)
            st.write("---")
    else:
        st.warning("No relevant visa information found for your query.")
