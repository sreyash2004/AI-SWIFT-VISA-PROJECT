# retrieval_temp.py

from dataclasses import dataclass

@dataclass
class VisaDoc:
    country: str
    visa_type: str
    content: str

# 14 countries × 3 visa types
countries = ["USA", "Canada", "UK", "Germany", "France", "Australia", "New Zealand",
             "Japan", "Singapore", "Netherlands", "Sweden", "Switzerland", "Italy", "Ireland"]
visa_types = ["Student Visa", "Work Visa", "Family Visa"]

visa_docs = []
for country in countries:
    for visa in visa_types:
        content = f"{country} {visa}: Description and requirements for {visa.lower()}."
        visa_docs.append(VisaDoc(country, visa, content))

# Simple retrieval function
def retrieve_documents(query, k=3):
    query_lower = query.lower()
    results = []

    for doc in visa_docs:
        if doc.country.lower() in query_lower or doc.visa_type.lower() in query_lower:
            results.append(doc)
        if len(results) >= k:
            break
    return results
