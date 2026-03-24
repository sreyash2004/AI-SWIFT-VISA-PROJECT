# prompt_builder.py

def build_prompt(user_data, retrieved_docs):
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are an expert Visa Eligibility Assistant.

Using the visa rules below, analyze the user and give a detailed decision.

-------------------------
VISA RULES:
{context}
-------------------------

USER DETAILS:
{user_data}

-------------------------
INSTRUCTIONS (STRICT):

1. Give FINAL DECISION:
   - ELIGIBLE or NOT ELIGIBLE

2. Give CLEAR REASON:
   - Explain WHY the user is eligible or not

3. Give CONFIDENCE SCORE:
   - Percentage (0% to 100%)

4. If NOT eligible:
   - Suggest WHAT TO IMPROVE

5. If eligible:
   - List REQUIRED DOCUMENTS

-------------------------

OUTPUT FORMAT:

📊 Final Decision:
(ELIGIBLE / NOT ELIGIBLE)

📌 Reason:
(Explain clearly)

📈 Confidence Score:
(Example: 85%)

📄 Required Documents / Improvements:
(List properly)

-------------------------
"""

    return prompt