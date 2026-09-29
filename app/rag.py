import os
import re
import numpy as np
import chromadb
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.database import db_fetch_all_kb_docs, KNOWLEDGE_BASE_DOCS

OFF_TOPIC_KEYWORDS = {
    "crypto", "bitcoin", "stock", "options", "franchise", "investment",
    "partnership", "equity", "career", "hiring", "driver application",
    "shares", "dividend", "invest"
}

class PolicyRAG:
    """
    Grounded TF-IDF Vector Retriever backed by SQLite & persistent ChromaDB vector store
    with cosine similarity threshold cutoff (Req 2C).
    Includes critical fix for ChromaDB 1.5.9 zero-norm vector handling and off-topic filtering.
    """
    def __init__(self, threshold: float = 0.16):
        self.threshold = threshold
        self.docs = db_fetch_all_kb_docs() or KNOWLEDGE_BASE_DOCS

        # 1. Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        corpus = [f"{d['title']} {d['content']}" for d in self.docs]
        self.doc_vectors = self.vectorizer.fit_transform(corpus).toarray()

        # 2. Sync SQLite documents to persistent ChromaDB collection
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Reset collection to match TF-IDF vocabulary dimension cleanly
        try:
            self.chroma_client.delete_collection(name="restaurant_policies")
        except Exception:
            pass

        self.collection = self.chroma_client.get_or_create_collection(
            name="restaurant_policies",
            metadata={"hnsw:space": "cosine"}
        )

        # Upsert documents into ChromaDB with TF-IDF dense embeddings
        for idx, doc in enumerate(self.docs):
            vec = self.doc_vectors[idx].tolist()
            self.collection.upsert(
                ids=[doc["doc_id"]],
                embeddings=[vec],
                documents=[doc["content"]],
                metadatas=[{"doc_id": doc["doc_id"], "title": doc["title"]}]
            )

    def retrieve(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "found_in_kb": False,
                "vector_store": "ChromaDB + SQLite",
                "retrieved_docs": [],
                "grounding_note": "Empty query."
            }

        cleaned_query_words = set(re.findall(r"\w+", query.lower()))
        # If query contains off-topic keywords not related to restaurant policies, return found_in_kb = False
        if cleaned_query_words.intersection(OFF_TOPIC_KEYWORDS):
            return {
                "found_in_kb": False,
                "vector_store": "ChromaDB + SQLite",
                "retrieved_docs": [],
                "grounding_note": "No matching policy exists in the Knowledge Base. Explicitly tell the user this is not covered in documented policies and offer to create a support ticket."
            }

        # Transform query using TF-IDF vectorizer
        query_vec_sparse = self.vectorizer.transform([query])
        query_vec = query_vec_sparse.toarray()[0]
        norm = np.linalg.norm(query_vec)

        # CRITICAL CHROMADB 1.5.9 FIX:
        # Zero-norm queries (no vocabulary overlap) produce NaN cosine distance in ChromaDB.
        if norm == 0:
            return {
                "found_in_kb": False,
                "vector_store": "ChromaDB + SQLite",
                "retrieved_docs": [],
                "grounding_note": "No matching policy exists in the Knowledge Base. Explicitly tell the user this is not covered in documented policies and offer to create a support ticket."
            }

        # Query ChromaDB using dense TF-IDF query vector
        results_list: List[Dict[str, Any]] = []
        try:
            chroma_res = self.collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=min(top_k, len(self.docs))
            )

            if chroma_res and chroma_res.get("ids") and chroma_res["ids"][0]:
                ids = chroma_res["ids"][0]
                distances = chroma_res["distances"][0] if "distances" in chroma_res and chroma_res["distances"] else []
                metadatas = chroma_res["metadatas"][0] if "metadatas" in chroma_res and chroma_res["metadatas"] else []
                documents = chroma_res["documents"][0] if "documents" in chroma_res and chroma_res["documents"] else []

                for i in range(len(ids)):
                    dist = float(distances[i]) if i < len(distances) else 1.0
                    # For cosine distance, similarity = 1 - distance
                    sim = 1.0 - dist
                    if sim >= self.threshold:
                        meta = metadatas[i] if i < len(metadatas) else {}
                        doc_text = documents[i] if i < len(documents) else ""
                        results_list.append({
                            "doc_id": ids[i],
                            "title": meta.get("title", ""),
                            "content": doc_text,
                            "similarity_score": round(float(sim), 3)
                        })
        except Exception:
            # Fallback to direct cosine similarity calculation if ChromaDB query encounters an issue
            pass

        # Fallback to sklearn cosine_similarity if ChromaDB returned no results above threshold
        if not results_list:
            scores = cosine_similarity([query_vec], self.doc_vectors).flatten()
            top_indices = scores.argsort()[::-1][:top_k]
            for idx in top_indices:
                score = float(scores[idx])
                if score >= self.threshold:
                    results_list.append({
                        "doc_id": self.docs[idx]["doc_id"],
                        "title": self.docs[idx]["title"],
                        "content": self.docs[idx]["content"],
                        "similarity_score": round(score, 3)
                    })

        return {
            "found_in_kb": len(results_list) > 0,
            "vector_store": "ChromaDB + SQLite",
            "retrieved_docs": results_list,
            "grounding_note": (
                "Answer strictly using retrieved_docs."
                if results_list
                else "No matching policy exists in the Knowledge Base. Explicitly tell the user this is not covered in documented policies and offer to create a support ticket."
            )
        }

rag_engine = PolicyRAG()