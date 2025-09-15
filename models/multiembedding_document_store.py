import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import math
from collections import defaultdict, Counter
import string

STOPWORDS = {
    # stopwords EN/ES mínimas (puedes ampliar)
    "the","a","an","and","or","of","to","in","on","for","with","by","is","are","was","were","be","as","at","that","this","it",
    "de","la","el","y","o","del","con","por","para","en","un","una","los","las","es","son","era","eran","ser","como","al","lo","se"
}

class DocumentStore():

    def __init__(
        self,
        collection_name: str = "chats",
        persist_directory: str = "./chroma_db",
        embedding_models: Optional[Dict[str, str]] = None,  # key -> HF model name
        multi_vector_config: Optional[Dict[str, Any]] = None,
    ):
        """
        embedding_models: dict key->modelo HF (SentenceTransformer).
        multi_vector_config: qué vistas indexar por chunk y pesos de fusión.
            {
              "variants": {
                 "chunk":    {"enabled": True,  "weight": 1.0},
                 "title":    {"enabled": True,  "weight": 0.3},
                 "summary":  {"enabled": True,  "weight": 0.6},
                 "keywords": {"enabled": True,  "weight": 0.4, "k": 12},
                 "sentence": {"enabled": True,  "weight": 0.8, "max_sentences": 6, "min_len": 20}
              },
              "fusion": "rrf",  # "rrf" o "sum_inverse_distance"
              "rrf_k": 60,      # parámetro de Reciprocal Rank Fusion
              "n_per_variant": 8  # top-N por variante antes de fusionar
            }
        """
        self.client = chromadb.PersistentClient(path=persist_directory)

        if embedding_models is None:
            embedding_models = {
                "light": "paraphrase-multilingual-MiniLM-L12-v2",
                "heavy": "intfloat/e5-large-v2"
            }

        # colecciones por modelo
        self.collections: Dict[str, chromadb.api.models.Collection] = {}
        for key in embedding_models.keys():
            col_name = f"{collection_name}_{key}"
            self.collections[key] = self.client.get_or_create_collection(
                name=col_name,
                configuration={"hnsw": {"space": "cosine"}},
            )

        self.embedding_models = {
            key: SentenceTransformer(model_name)
            for key, model_name in embedding_models.items()
        }

        # primaria (para leer texto/metadata "chunk")
        self.primary_key = next(iter(self.collections.keys()))
        self.primary_collection = self.collections[self.primary_key]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # configuración multi-vector por defecto
        if multi_vector_config is None:
            multi_vector_config = {
                "variants": {
                    "chunk":    {"enabled": True,  "weight": 1.0},
                    "title":    {"enabled": True,  "weight": 0.3},
                    "summary":  {"enabled": True,  "weight": 0.6},
                    "keywords": {"enabled": True,  "weight": 0.4, "k": 12},
                    "sentence": {"enabled": True,  "weight": 0.8, "max_sentences": 6, "min_len": 20}
                },
                "fusion": "rrf",
                "rrf_k": 60,
                "n_per_variant": 8
            }
        self.mv = multi_vector_config

    # ---------- helpers de variantes ----------

    def _split_sentences(self, text: str) -> List[str]:
        # Split muy simple por puntuación (ajusta si usas spaCy/punkt)
        sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        # limpia
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        return sentences

    def _pick_sentences(self, text: str, max_sentences: int, min_len: int) -> List[str]:
        sents = self._split_sentences(text)
        # heurística: prioriza frases más informativas (longitud)
        sents = [s for s in sents if len(s) >= min_len]
        sents.sort(key=len, reverse=True)
        return sents[:max_sentences]

    def _simple_summary(self, text: str, max_sentences: int = 3) -> str:
        # resumen heurístico: toma las frases más largas (proxy de saliencia)
        sents = self._split_sentences(text)
        ranked = sorted(sents, key=len, reverse=True)[:max_sentences]
        ranked = sorted(ranked, key=lambda s: sents.index(s))  # mantiene algo de orden original
        return " ".join(ranked) if ranked else text[:300]

    def _keyword_string(self, text: str, k: int = 12) -> str:
        # extracción muy simple de keywords (frecuencia sin stopwords)
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text.lower())
        tokens = [t for t in tokens if t not in STOPWORDS and t not in string.punctuation and len(t) > 2]
        freq = Counter(tokens)
        words = [w for w, _ in freq.most_common(k)]
        return " ".join(words)

    def _title_from_meta(self, metadata: Optional[Dict[str, Any]]) -> str:
        if not metadata: 
            return ""
        # usa file_name o title si existen
        return str(metadata.get("title") or metadata.get("file_name") or "")[:200]

    # ---------- indexación multi-vector ----------

    def add_text_document(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """Indexa el documento en todas las colecciones y en múltiples variantes por chunk."""
        chunks = self.text_splitter.split_text(text)
        print(f"Adding {len(chunks)} chunks x {len(self.collections)} models, multi-vector variants.")

        title = self._title_from_meta(metadata)
        for i, chunk in enumerate(chunks):
            base_id = f"{metadata.get('file_name','text')}_{i}_{uuid.uuid4().hex[:8]}" if metadata else f"text_{i}_{uuid.uuid4().hex[:8]}"

            base_meta = (metadata.copy() if metadata else {})
            base_meta.update({'chunk_id': i, 'total_chunks': len(chunks), 'base_id': base_id})

            # prepara variantes (texto a embeber)
            variants: List[Tuple[str, str, Optional[int]]] = []  # (variant_name, text, subindex)

            if self.mv["variants"]["chunk"]["enabled"]:
                variants.append(("chunk", chunk, None))

            if self.mv["variants"]["title"]["enabled"] and title:
                variants.append(("title", title, None))

            if self.mv["variants"]["summary"]["enabled"]:
                summary = self._simple_summary(chunk, max_sentences=3)
                variants.append(("summary", summary, None))

            if self.mv["variants"]["keywords"]["enabled"]:
                kw = self._keyword_string(chunk, k=self.mv["variants"]["keywords"].get("k", 12))
                if kw:
                    variants.append(("keywords", kw, None))

            if self.mv["variants"]["sentence"]["enabled"]:
                max_s = self.mv["variants"]["sentence"].get("max_sentences", 6)
                min_len = self.mv["variants"]["sentence"].get("min_len", 20)
                sent_list = self._pick_sentences(chunk, max_s, min_len)
                for si, s in enumerate(sent_list):
                    variants.append(("sentence", s, si))

            # indexa en todas las colecciones (modelos)
            for model_key, model in self.embedding_models.items():
                collection = self.collections[model_key]
                embeds = []
                docs = []
                metas = []
                ids = []

                for (vname, vtext, subidx) in variants:
                    emb = model.encode(vtext, normalize_embeddings=True).tolist()
                    doc_id = f"{base_id}::v={vname}" + (f"::i={subidx}" if subidx is not None else "")
                    mv_meta = base_meta.copy()
                    mv_meta.update({"variant": vname, "subindex": subidx})
                    embeds.append(emb)
                    docs.append(vtext)
                    metas.append(mv_meta)
                    ids.append(doc_id)

                collection.add(embeddings=embeds, documents=docs, metadatas=metas, ids=ids)
                print(f"[{model_key}] Added chunk {i+1}/{len(chunks)} as {len(variants)} variants (base_id={base_id})")

    # ---------- búsqueda y fusión ----------

    def _score_from_distance(self, dist: float, eps: float = 1e-6) -> float:
        return 1.0 / (dist + eps)

    def search_similar_documents(
        self,
        query: str,
        n_results: int = 5,
        models: Optional[List[str]] = None,
        combine: Optional[str] = None,   # override de fusión (si quieres)
        restrict_variants: Optional[List[str]] = None,  # p.ej. ["chunk","sentence"]
        per_variant_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Consulta multi-modelo + multi-variant y fusiona por base_id.
        """
        if models is None:
            models = list(self.embedding_models.keys())

        fusion = combine or self.mv.get("fusion", "rrf")
        rrf_k = self.mv.get("rrf_k", 60)
        n_per_variant = per_variant_n or self.mv.get("n_per_variant", 8)

        # pesos por variante
        var_cfg = self.mv["variants"]
        use_variants = [v for v, cfg in var_cfg.items() if cfg.get("enabled", False)]
        if restrict_variants:
            use_variants = [v for v in use_variants if v in restrict_variants]

        # acumuladores
        agg_scores: Dict[str, float] = defaultdict(float)      # base_id -> score
        best_hit_for_base: Dict[str, Tuple[str, str]] = {}     # base_id -> (model_key, doc_id) para rescatar texto
        rank_lists: Dict[str, Dict[str, int]] = defaultdict(dict)  # variant -> base_id -> best_rank

        # ejecuta por modelo y por variante
        for model_key in models:
            if model_key not in self.embedding_models:
                print(f"Model key {model_key} no encontrado, saltando.")
                continue
            model = self.embedding_models[model_key]
            collection = self.collections[model_key]
            q_emb = model.encode(query, normalize_embeddings=True).tolist()

            for vname in use_variants:
                # filtra por metadata variant=vname
                results = collection.query(
                    query_embeddings=[q_emb],
                    n_results=n_per_variant,
                    where={"variant": vname}
                )

                ids = results.get('ids', [[]])[0]
                distances = results.get('distances', [[]])[0]

                # build ranking por base_id
                for rank, (doc_id, dist) in enumerate(zip(ids, distances), start=1):
                    if dist is None or doc_id is None:
                        continue
                    # extrae base_id del doc_id (antes de "::v=")
                    base_id = doc_id.split("::v=")[0]
                    weight = var_cfg.get(vname, {}).get("weight", 1.0)

                    if fusion == "rrf":
                        score = weight * (1.0 / (rrf_k + rank))
                    else:
                        score = weight * self._score_from_distance(dist)

                    if base_id not in best_hit_for_base:
                        best_hit_for_base[base_id] = (model_key, doc_id)

                    # guarda mejor rank por variante (para no sobrerrepresentar muchas sub-frases)
                    if base_id not in rank_lists[vname] or rank < rank_lists[vname][base_id]:
                        rank_lists[vname][base_id] = rank

                    agg_scores[base_id] += score

        # si usaste RRF, ya está fusionado; si quisieras “máximo por variante” adicional, podrías normalizar aquí
        ranked = sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)[:n_results]

        results_out: List[Dict] = []
        for base_id, agg_score in ranked:
            # recuperar texto/metadata del variant "chunk" en la colección primaria
            # como insertamos todo con metadatos base_id, hacemos get por filtro
            primary = self.primary_collection.get(where={"base_id": base_id, "variant": "chunk"}, limit=1)
            docs = primary.get('documents', [])
            metas = primary.get('metadatas', [])

            document = docs[0] if docs else None
            metadata = metas[0] if metas else {"base_id": base_id}

            # también puedes devolver evidencia de qué variantes contribuyeron (opcional)
            results_out.append({
                'document': document,
                'metadata': metadata,
                'score': agg_score,
                'base_id': base_id
            })

        return results_out
