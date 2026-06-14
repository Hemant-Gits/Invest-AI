"""FAISS vector store for financial document retrieval."""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document


class FinancialVectorStore:
    """Manage FAISS index for stock intelligence documents."""

    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        self.documents: list[Document] = []

    def _init_embeddings(self):
        if self.embeddings is not None:
            return
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception:
            self.embeddings = None

    def build_index(self, documents: list[Document]) -> bool:
        self.documents = documents
        if not documents:
            return False

        self._init_embeddings()
        if self.embeddings is None:
            return False

        try:
            from langchain_community.vectorstores import FAISS
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            return True
        except Exception:
            return False

    def search(self, query: str, k: int = 5) -> list[Document]:
        if self.vectorstore is None:
            return self._fallback_search(query, k)
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception:
            return self._fallback_search(query, k)

    def _fallback_search(self, query: str, k: int) -> list[Document]:
        query_lower = query.lower()
        scored = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def is_ready(self) -> bool:
        return self.vectorstore is not None or len(self.documents) > 0
