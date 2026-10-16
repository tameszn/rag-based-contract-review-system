from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
PERSIST_DIRECTORY = "./chroma_contract_db"

def load_vectorstore():
    """
    Loads the persisted Chroma database using the identical embedding model.
    """
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

def retrieve_contract_context(query: str, top_k: int = 4):
    """
    Retrieves the most relevant contract clause chunks for a given query.
    """
    vectorstore = load_vectorstore()
    
    # Perform similarity search with distance scores
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "section": doc.metadata.get("section", "N/A"),
            "clause": doc.metadata.get("clause", "N/A"),
            "score": score
        })
        
    return formatted_results

if __name__ == "__main__":
    sample_query = "What are the rules regarding liability caps and indemnification obligations?"
    print(f"Querying Vector Database: '{sample_query}'\n")

    matched_chunks = retrieve_contract_context(sample_query, top_k=3)

    for idx, match in enumerate(matched_chunks, 1):
        print(f"--- MATCH {idx} (Distance Score: {match['score']:.4f}) ---")
        print(f"Parent Section: {match['section']}")
        print(f"Clause Ref    : {match['clause']}")
        print(f"Content:\n{match['content']}")
        print("=" * 60)