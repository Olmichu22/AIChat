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
    
    def add_pdf_document(self, pdf_path: str, metadata: Dict[str, Any] = None) -> None:
    """
    Lee un PDF, lo divide en chunks y lo añade a la colección de ChromaDB.
    Divide por páginas y, dentro de cada página, aplica el mismo splitter que para texto.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No se encontró el PDF: {pdf_path}")
    
    file_name = os.path.basename(pdf_path)
    base_metadata = metadata.copy() if metadata else {}
    base_metadata.setdefault("file_name", file_name)
    base_metadata.setdefault("source", "pdf")
    base_metadata.setdefault("path", os.path.abspath(pdf_path))

    # Leer PDF
    reader = PyPDF2.PdfReader(pdf_path)
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")  # intenta desencriptar sin contraseña (algunos PDFs permiten)
        except Exception:
            print(f"Advertencia: el PDF '{file_name}' está cifrado y no se pudo desencriptar.")

    total_pages = len(reader.pages)
    print(f"Procesando '{file_name}' con {total_pages} páginas...")

    total_chunks_global = 0
    for page_idx in range(total_pages):
        page_number = page_idx + 1
        try:
            page_text = reader.pages[page_idx].extract_text() or ""
        except Exception as e:
            print(f"⚠️  No se pudo extraer texto de la página {page_number}: {e}")
            page_text = ""

        # Saltar páginas sin texto
        if not page_text.strip():
            print(f"Página {page_number} vacía o sin texto extraíble, se omite.")
            continue

        # Trocear la página
        page_chunks = self.text_splitter.split_text(page_text)
        print(f"Página {page_number}: {len(page_chunks)} chunks.")

        # Añadir cada chunk
        for i, chunk in enumerate(page_chunks):
            embedding = self.embedding_model.encode(
                chunk, normalize_embeddings=True
            ).tolist()

            # ID único por archivo/página/chunk
            doc_id = f"{file_name}_p{page_number}_c{i}"

            # Metadatos del chunk
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "page": page_number,
                "chunk_id": i,
                "total_chunks_on_page": len(page_chunks),
            })

            print(
                f"Añadiendo chunk p{page_number} c{i+1}/{len(page_chunks)} "
                f"con ID {doc_id}"
            )
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[doc_id],
            )
            total_chunks_global += 1

    print(f"Listo. Se añadieron {total_chunks_global} chunks en total desde '{file_name}'.")

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
