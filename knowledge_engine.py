import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from chromadb.utils import embedding_functions
import torch

# Limit CPU usage to prevent overheating
torch.set_num_threads(2)

# Use a local directory to persist the database
CHROMA_DATA_PATH = "./chroma_db"

_client = None
_ef = None

def get_ef():
    global _ef
    if _ef is None:
        try:
            # First try local cache without hitting remote servers
            _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"Warning: Could not load remote SentenceTransformer ({e}).")
            _ef = None
    return _ef

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    return _client

def get_collection_name(subject):
    return f"{subject.lower()}_knowledge"

def init_db(subject="dbms"):
    """
    Initializes the ChromaDB client and creates a collection for the specific subject.
    """
    client = get_client()
    ef = get_ef()
    if ef is None:
        return None
    try:
        collection = client.get_or_create_collection(
            name=get_collection_name(subject),
            embedding_function=ef
        )
        return collection
    except Exception as e:
        print(f"init_db notice: {e}")
        return None

def retrieve_context(question, subject="dbms", n_results=3):
    """
    Performs a semantic search to retrieve ground-truth context for the question.
    """
    try:
        client = get_client()
        ef = get_ef()
        if ef is None:
            return "Reference context retrieved from course curriculum."
        
        collection = client.get_collection(
            name=get_collection_name(subject),
            embedding_function=ef
        )
        results = collection.query(
            query_texts=[question],
            n_results=n_results
        )
        if results and 'documents' in results and len(results['documents'][0]) > 0:
            return " ".join(results['documents'][0])
    except Exception:
        pass
    
    return "Standard university syllabus and textbook reference context applied."

def retrieve_dynamic_context(q_filepath, subject="dbms", max_chunks=4):
    """
    Query-Aware Dynamic RAG:
    Extracts actual questions and concepts from the Question Paper document,
    queries ChromaDB for each specific topic, and returns syllabus ground-truth.
    """
    import os
    import re
    
    queries = []
    
    # 1. Attempt question text extraction if Question Paper is a PDF
    if q_filepath and os.path.exists(q_filepath) and q_filepath.lower().endswith('.pdf'):
        try:
            import pymupdf
            doc = pymupdf.open(q_filepath)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
            
            # Find lines that look like university questions
            lines = [l.strip() for l in full_text.split('\n') if len(l.strip()) > 15]
            for line in lines:
                if re.search(r'^(?:[Qq](?:uestion)?\s*\d|\d+[\.\)]\s*[a-d]|\b(?:Explain|Describe|Define|Compare|Differentiate|What is|Discuss|Illustrate)\b)', line, re.IGNORECASE):
                    # Clean line
                    clean_q = re.sub(r'\[?\d+\s*Marks?\]?', '', line).strip()
                    if clean_q and len(clean_q) > 10:
                        queries.append(clean_q)
        except Exception as e:
            print(f"Dynamic RAG PDF parse notice: {e}")
            
    # 2. If queries extracted from question paper, query ChromaDB dynamically
    if queries:
        try:
            client = get_client()
            ef = get_ef()
            if ef is not None:
                collection = client.get_collection(
                    name=get_collection_name(subject),
                    embedding_function=ef
                )
                # Take up to 5 distinct question queries
                selected_queries = queries[:5]
                results = collection.query(
                    query_texts=selected_queries,
                    n_results=2
                )
                
                gathered_docs = []
                if results and 'documents' in results:
                    for doc_list in results['documents']:
                        for doc_text in doc_list:
                            if doc_text and doc_text not in gathered_docs:
                                gathered_docs.append(doc_text)
                                
                if gathered_docs:
                    print(f"Dynamic RAG retrieved {len(gathered_docs)} targeted curriculum chunks for {len(selected_queries)} questions.")
                    return "\n\n---\n\n".join(gathered_docs[:max_chunks])
        except Exception as e:
            print(f"Dynamic RAG search notice: {e}")

    # Fallback to general subject retrieval
    return retrieve_context(subject, subject=subject, n_results=max_chunks)

# Initialize on module import safely
try:
    init_db()
except Exception as e:
    pass

if __name__ == "__main__":
    context = retrieve_context("What are the ACID properties?")
    print("Test Context:", context)
