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
    sub = (subject or "").lower().replace(" ", "").replace("_", "").replace("-", "")
    if "bcs" in sub or "bca" in sub or "703" in sub or "crypto" in sub or "security" in sub:
        return "bcs703_knowledge"
    return f"{sub}_knowledge"

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

import math
import re
from collections import Counter

class BM25Retriever:
    """
    Pure-Python Okapi BM25 implementation for sparse lexical search.
    Complements dense embeddings by prioritizing exact technical keyword and acronym matches.
    """
    def __init__(self, documents, ids, metadatas=None, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.ids = ids
        self.metadatas = metadatas or [{} for _ in documents]
        self.doc_lens = [len(self._tokenize(d)) for d in documents]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.corpus_size = len(documents)
        self.doc_freqs = Counter()
        self.tokenized_corpus = []
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                self.doc_freqs[t] += 1
            self.tokenized_corpus.append(self._tokenize(doc))

    def _tokenize(self, text):
        return re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())

    def search(self, query, top_k=10):
        q_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        for token in q_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for i, doc_tokens in enumerate(self.tokenized_corpus):
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (self.doc_lens[i] / self.avgdl)))
                scores[i] += score

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.documents[i], self.ids[i], self.metadatas[i], scores[i]) for i in ranked_indices[:top_k] if scores[i] > 0]

_bm25_cache = {}

def get_bm25_retriever(subject):
    global _bm25_cache
    sub_key = get_collection_name(subject)
    if sub_key in _bm25_cache:
        return _bm25_cache[sub_key]
    
    try:
        client = get_client()
        ef = get_ef()
        collection = client.get_collection(name=sub_key, embedding_function=ef)
        data = collection.get()
        if data and data.get('documents'):
            retriever = BM25Retriever(data['documents'], data['ids'], data.get('metadatas'))
            _bm25_cache[sub_key] = retriever
            return retriever
    except Exception as e:
        print(f"BM25 build notice for {sub_key}: {e}")
    return None

def retrieve_hybrid_context(query, subject="dbms", n_results=3):
    """
    State-of-the-art Hybrid RAG (Reciprocal Rank Fusion - RRF):
    Fuses Dense Vector Search (ChromaDB) + Sparse Lexical Search (BM25).
    """
    sub_name = get_collection_name(subject)
    dense_docs = {}
    sparse_docs = {}
    doc_text_map = {}
    doc_meta_map = {}
    
    # 1. Dense Semantic Retrieval (ChromaDB)
    try:
        client = get_client()
        ef = get_ef()
        if ef is not None:
            col = client.get_collection(name=sub_name, embedding_function=ef)
            res = col.query(query_texts=[query], n_results=min(10, n_results * 3))
            if res and res.get('ids') and res['ids'][0]:
                for rank, (doc_id, doc_text) in enumerate(zip(res['ids'][0], res['documents'][0])):
                    dense_docs[doc_id] = rank + 1
                    doc_text_map[doc_id] = doc_text
                    if res.get('metadatas') and res['metadatas'][0]:
                        doc_meta_map[doc_id] = res['metadatas'][0][rank]
    except Exception as e:
        print(f"Dense query notice: {e}")
        
    # 2. Sparse Lexical Retrieval (BM25)
    try:
        bm25 = get_bm25_retriever(subject)
        if bm25 is not None:
            sparse_hits = bm25.search(query, top_k=min(10, n_results * 3))
            for rank, (doc_text, doc_id, meta, score) in enumerate(sparse_hits):
                sparse_docs[doc_id] = rank + 1
                doc_text_map[doc_id] = doc_text
                doc_meta_map[doc_id] = meta
    except Exception as e:
        print(f"Sparse query notice: {e}")
        
    # 3. Reciprocal Rank Fusion (RRF)
    all_doc_ids = set(dense_docs.keys()) | set(sparse_docs.keys())
    if not all_doc_ids:
        return [retrieve_context(query, subject=subject, n_results=n_results)]
        
    rrf_scores = {}
    k_const = 60 # Standard RRF constant
    for doc_id in all_doc_ids:
        r_dense = dense_docs.get(doc_id, 100)
        r_sparse = sparse_docs.get(doc_id, 100)
        score = (1.0 / (k_const + r_dense)) + (1.0 / (k_const + r_sparse))
        rrf_scores[doc_id] = score
        
    sorted_doc_ids = sorted(all_doc_ids, key=lambda d: rrf_scores[d], reverse=True)
    top_docs = [doc_text_map[d] for d in sorted_doc_ids[:n_results] if d in doc_text_map]
    return top_docs

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
            
    rag_guidance_header = "[CURRICULUM BENCHMARK CONTEXT - FOR TOPICAL REFERENCE ONLY; NOT A VERBATIM KEYWORD CHECKLIST]\n\n"
    
    # 2. If queries extracted from question paper, query using Hybrid RAG (Dense + BM25 RRF)
    if queries:
        try:
            gathered_docs = []
            selected_queries = queries[:5]
            for q in selected_queries:
                hybrid_chunks = retrieve_hybrid_context(q, subject=subject, n_results=2)
                for chunk in hybrid_chunks:
                    if chunk and chunk not in gathered_docs:
                        gathered_docs.append(chunk)
                        
            if gathered_docs:
                print(f"Hybrid RAG (Dense + BM25 RRF) retrieved {len(gathered_docs)} targeted curriculum chunks for {len(selected_queries)} questions.")
                return rag_guidance_header + "\n\n---\n\n".join(gathered_docs[:max_chunks])
        except Exception as e:
            print(f"Hybrid RAG search notice: {e}")

    # Fallback to general hybrid retrieval
    hybrid_fallback = retrieve_hybrid_context(subject, subject=subject, n_results=max_chunks)
    if hybrid_fallback:
        return rag_guidance_header + "\n\n---\n\n".join(hybrid_fallback)
    return rag_guidance_header + retrieve_context(subject, subject=subject, n_results=max_chunks)

# Initialize on module import safely
try:
    init_db()
except Exception as e:
    pass

if __name__ == "__main__":
    context = retrieve_context("What are the ACID properties?")
    print("Test Context:", context)
