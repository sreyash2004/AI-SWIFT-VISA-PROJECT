# retrievalout.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# --------------------------
# EMBEDDING MODEL
# --------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------
# EXAMPLE VISA DOCUMENTS
# Replace/add your visa rules here
# --------------------------
visa_docs = [
    Document(
        page_content="US Student Visa eligibility: Must have admission letter from a US university. Financial proof required.",
        metadata={"country": "US", "visa_type": "Student Visa"}
    ),
    Document(
        page_content="US Work Visa: Requires job offer and labor certification.",
        metadata={"country": "US", "visa_type": "Work Visa"}
    ),
    Document(
        page_content="Canada Work Visa: Requires job offer from Canadian employer and language test results.",
        metadata={"country": "Canada", "visa_type": "Work Visa"}
    ),
    Document(
        page_content="Canada Student Visa: Must have admission letter from Canadian university. Financial proof required.",
        metadata={"country": "Canada", "visa_type": "Student Visa"}
    ),
]

# --------------------------
# BUILD FAISS INDEX IN MEMORY
# --------------------------
db = FAISS.from_documents(visa_docs, embedding_model)

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

    results_with_scores = db.similarity_search_with_score(query, k=k, filter=search_filter)
    if not results_with_scores:
        return []

    best_doc, best_score = results_with_scores[0]
    if len(results_with_scores) > 1:
        second_score = results_with_scores[1][1]
        if abs(best_score - second_score) < 0.05:
            return [results_with_scores[0][0], results_with_scores[1][0]]

    return [best_doc]
