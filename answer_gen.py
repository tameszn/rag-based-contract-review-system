import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# Imports your retrieval function from data_ret.py
from data_ret import retrieve_contract_context

def generate_strategic_contract_analysis(query: str):
    """
    Retrieves context from ChromaDB and forces Gemini to output a structured risk analysis.
    """
    # 1. Retrieve the top context chunks (increased to 8 to ensure broad coverage of all 4 categories)
    retrieved_chunks = retrieve_contract_context(query, top_k=8)
    
    if not retrieved_chunks:
        return "No relevant clauses found in the database. Ensure the contract is ingested correctly."

    context_text = "\n\n---\n\n".join(
        [f"[{item['section']} | Clause {item['clause']}]\n{item['content']}" for item in retrieved_chunks]
    )

    # 2. Advanced System Prompt Engineering
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI Legal Counsel. Your objective is to deeply analyze the provided contract clauses and generate a highly structured risk assessment. 

You MUST format your response exactly using the following 3 sections. If a specific issue is not found in the context, explicitly state "Not covered in the provided context."

### 1. TARGETED CLAUSE ANALYSIS
Evaluate the provided context strictly across these four categories:
*   **Compliance & Legal Issues:** (e.g., Governing law, regulatory adherence, confidentiality/NDA)
*   **Liability Issues:** (e.g., Liability caps, indemnification obligations, risk allocation)
*   **Financial Issues:** (e.g., Payment terms, financial penalties, costs)
*   **Termination & IP Issues:** (e.g., Exit conditions, cause for termination, intellectual property ownership)

### 2. RISK PRIORITIZATION MATRIX
Identify the specific risks found in the analysis above. Rank them strictly as:
*   [CRITICAL RISK]: (Deal-breakers or severe financial/legal exposure)
*   [MEDIUM RISK]: (Standard operational risks that need monitoring)
*   [LOW RISK]: (Standard boilerplate protections)
Briefly explain *why* each risk received its rank.

### 3. EXECUTIVE SUMMARY & FINAL DECISION
*   **Summary:** Provide a well-elaborated summary of the contract's overall safety and posture.
*   **Decision:** Conclude with a definitive, actionable decision under safe conditions (e.g., "PROCEED TO SIGN", "RENEGOTIATE SPECIFIC CLAUSES", or "DO NOT SIGN"). Detail the exact conditions or amendments the user must secure before moving forward.
"""),
        ("user", "Context:\n{context}\n\nTargeting Query: {query}")
    ])

    # 3. Initialize Gemini (Using Gemini 2.5 Flash for high-speed reasoning)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash", 
        temperature=0.1 # Low temperature for highly analytical, deterministic legal outputs
    )
    
    chain = prompt | llm

    # 4. Generate the final response
    response = chain.invoke({"context": context_text, "query": query})
    return response.content

# Replace the bottom execution block in answer_gen.py with this:

if __name__ == "__main__":
    strategic_query = "Analyze the contract for compliance, liability, financial, and IP/termination risks."
    
    print("Running strategic contract analysis...")
    raw_response = generate_strategic_contract_analysis(strategic_query)
    
    # Safely unpack text if Gemini returns a list of content dicts
    if isinstance(raw_response, list):
        clean_text = "".join([block.get("text", "") for block in raw_response if isinstance(block, dict)])
    else:
        clean_text = str(raw_response)

    # Save to Markdown for clean previewing in Codespaces
    output_file = "contract_report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"\nAnalysis complete! Open '{output_file}' in your sidebar to read the clean report.")