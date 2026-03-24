# retrievalout.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # keep if still needed for FAISS wrapper

# --------------------------
# LOAD EMBEDDING MODEL
# --------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------
# LOAD FAISS INDEX
# --------------------------
db = FAISS.load_local(
    "index.faiss",              # path to your FAISS index folder
    embedding_model,
    allow_dangerous_deserialization=True
)

# --------------------------
# SYNONYM MAP
# --------------------------
VISA_SYNONYMS = {
    "student visa": ["study permit", "student permit", "study visa"],
    "work visa": ["work permit", "job visa"],
    "family visa": ["dependent visa", "spouse visa", "partner visa"]
}

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def detect_country(query_lower, all_countries):
    for country in all_countries:
        if country.lower() in query_lower:
            return country
    return None

def detect_visa(query_lower, all_visa_types):
    for visa in all_visa_types:
        visa_lower = visa.lower()

        if visa_lower in query_lower:
            return visa

        if visa_lower in VISA_SYNONYMS:
            for synonym in VISA_SYNONYMS[visa_lower]:
                if synonym in query_lower:
                    return visa
    return None

# --------------------------
# DOCUMENT RETRIEVAL FUNCTION
# --------------------------
def retrieve_documents(query, k=3):
    query_lower = query.lower()

    all_countries = set(
        doc.metadata.get("country")
        for doc in db.docstore._dict.values()
        if doc.metadata.get("country")
    )

    all_visa_types = set(
        doc.metadata.get("visa_type")
        for doc in db.docstore._dict.values()
        if doc.metadata.get("visa_type")
    )

    detected_country = detect_country(query_lower, all_countries)
    detected_visa = detect_visa(query_lower, all_visa_types)

    if not detected_country:
        return []

    search_filter = {"country": detected_country}
    if detected_visa:
        search_filter["visa_type"] = detected_visa

    results_with_scores = db.similarity_search_with_score(
        query, k=k, filter=search_filter
    )

    if not results_with_scores:
        return []

    # Return best chunks (top 1 or top 2 if scores are very close)
    best_doc, best_score = results_with_scores[0]
    if len(results_with_scores) > 1:
        second_score = results_with_scores[1][1]
        if abs(best_score - second_score) < 0.05:
            return [results_with_scores[0][0], results_with_scores[1][0]]

    return [best_doc]
