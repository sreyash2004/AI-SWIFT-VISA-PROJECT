# retrieval.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load embeddings
embedding_model = HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

# Load prebuilt FAISS index
db = FAISS.load_local("faiss_index", embedding_model)

# Synonym mapping
VISA_SYNONYMS = {
    "student visa": ["study permit", "student permit", "study visa"],
    "work visa": ["work permit", "job visa"],
    "family visa": ["dependent visa", "spouse visa", "partner visa"]
}

COUNTRIES = [
    "USA", "Canada", "UK", "Germany", "France", "Australia", "New Zealand",
    "Japan", "Singapore", "Netherlands", "Sweden", "Switzerland", "Italy", "Ireland"
]

VISA_TYPES = ["Student Visa", "Work Visa", "Family Visa"]

# Helper functions
def detect_country(query_lower):
    for country in COUNTRIES:
        if country.lower() in query_lower:
            return country
    return None

def detect_visa(query_lower):
    for visa in VISA_TYPES:
        visa_lower = visa.lower()
        if visa_lower in query_lower:
            return visa
        if visa_lower in VISA_SYNONYMS:
            for synonym in VISA_SYNONYMS[visa_lower]:
                if synonym in query_lower:
                    return visa
    return None

def retrieve_documents(query, k=3):
    query_lower = query.lower()
    detected_country = detect_country(query_lower)
    detected_visa = detect_visa(query_lower)

    if not detected_country:
        return []

    search_filter = {"country": detected_country}
    if detected_visa:
        search_filter["visa_type"] = detected_visa

    results = db.similarity_search(query, k=k, filter=search_filter)
    return results
