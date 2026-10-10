import sys
import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from knowledge_engine import CHROMA_DATA_PATH, get_collection_name

def extract_chunks_from_pdf(pdf_path, chunk_size=1000, overlap=200):
    """
    Extracts text from a PDF and splits it into chunks of `chunk_size` characters
    with `overlap` characters to maintain context.
    """
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + " "
            
    # Clean up excess whitespace
    full_text = " ".join(full_text.split())
    print(f"Total characters extracted: {len(full_text)}")
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap # Move forward, leaving overlap
        
    print(f"Created {len(chunks)} text chunks for the database.")
    return chunks

def ingest_pdf(pdf_path, subject):
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        return
        
    try:
        chunks = extract_chunks_from_pdf(pdf_path)
        
        # Connect to existing ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        collection = client.get_or_create_collection(
            name=get_collection_name(subject),
            embedding_function=sentence_transformer_ef
        )
        
        print(f"Embedding and adding chunks to the '{subject}' vector database. This may take a moment...")
        
        # Add chunks in batches to avoid memory overload if textbook is huge
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_ids = [f"pdf_chunk_{uuid.uuid4().hex}" for _ in batch_chunks]
            
            collection.upsert(
                documents=batch_chunks,
                ids=batch_ids
            )
            print(f"Upserted batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
        print(f"Successfully ingested '{pdf_path}' into the '{subject}' knowledge base!")
        print(f"Total documents in database now: {collection.count()}")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest_textbook.py <subject> <path_to_pdf>")
        print("Example: python ingest_textbook.py cn computer_networks.pdf")
        sys.exit(1)
        
    subject_arg = sys.argv[1]
    pdf_file = sys.argv[2]
    ingest_pdf(pdf_file, subject_arg)
