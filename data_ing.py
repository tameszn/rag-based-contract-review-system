import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
PERSIST_DIRECTORY = "./chroma_contract_db"

def run_ingestion_pipeline(pdf_path: str):
    """
    Parses a PDF contract into hierarchical clause chunks and ingests them into ChromaDB.
    """
    print(f"Loading contract: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Combine full text and clean page number artifacts
    full_text = "\n".join([page.page_content for page in pages])
    full_text = re.sub(r'\nPage\s+\d+\n?', '\n', full_text, flags=re.IGNORECASE)

    # Split by major section headers (e.g., "1 Supply of Ingredients", "5 Confidential Information.")[cite: 1, 4]
    section_pattern = re.compile(r'\n(\d{1,2}\s+[A-Z][^\n]+)')
    sections = section_pattern.split(full_text)

    documents = []

    for i in range(1, len(sections), 2):
        parent_header = sections[i].strip()
        section_body = sections[i+1].strip()

        # Split into sub-clauses (e.g., "1.1", "5.3.1")[cite: 1, 5]
        clause_pattern = re.compile(r'\n(\d{1,2}\.\d+(?:\.\d+)?)(?=\s|\n)')
        clauses = clause_pattern.split("\n" + section_body)

        # Handle introductory text preceding the first sub-clause
        intro_text = clauses[0].strip()
        if intro_text:
            documents.append(Document(
                page_content=f"Parent Section Header: {parent_header}\n\n{intro_text}",
                metadata={"source": pdf_path, "section": parent_header}
            ))

        # Build chunks with injected parent headers
        for j in range(1, len(clauses), 2):
            clause_number = clauses[j].strip()
            clause_text = clauses[j+1].strip()
            clause_text = re.sub(r'\|\s*', '', clause_text)  # Strip layout formatting lines[cite: 3]

            if clause_text:
                documents.append(Document(
                    page_content=f"Parent Section Header: {parent_header}\n\nClause {clause_number}: {clause_text}",
                    metadata={
                        "source": pdf_path,
                        "section": parent_header,
                        "clause": clause_number
                    }
                ))

    print(f"Created {len(documents)} structural clause chunks.")

    # Initialize shared BAAI embedding model
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Persist vector collection
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"Successfully stored vector index in '{PERSIST_DIRECTORY}'.")
    return vectorstore

if __name__ == "__main__":
    run_ingestion_pipeline("sample_contract.PDF")