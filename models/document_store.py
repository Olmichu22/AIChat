import os
import json
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentStore():

    def __init__(
        self, collection_name: str = "chats", persist_directory: str = "./chroma_db",
        embedding_model = "light"
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )
        embeddings = {"light" : "paraphrase-multilingual-MiniLM-L12-v2",
                      "heavy" : "intfloat/e5-large-v2"}
        
        self.embedding_model = SentenceTransformer(
            embeddings[embedding_model]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def add_text_document(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """Añade un documento de texto a la base de datos"""
        chunks = self.text_splitter.split_text(text)
        print(f"Adding {len(chunks)} chunks to the collection.")
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk, normalize_embeddings=True).tolist()
            doc_id = f"{metadata.get('file_name', 'text')}_{i}"
            print(f"Adding chunk {i+1}/{len(chunks)} with ID {doc_id}")
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_id'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            print(f"Metadata for chunk {i+1}: {chunk_metadata}")

            self.collection.add(
          embeddings=[embedding],
          documents=[chunk],
          metadatas=[chunk_metadata],
          ids=[doc_id]
      )
            print(f"Chunk {i+1} added with ID {doc_id} and metadata {chunk_metadata}")
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Busca documentos similares a la consulta"""
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()

        results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

        return [{
        'document': doc,
        'metadata': meta,
        'distance': dist
    } for doc, meta, dist in zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )]
