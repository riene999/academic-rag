import math
import re
from collections import Counter
from typing import Dict, List

from src.rag.retriever import RetrievedChunk
from src.utils.pdf_parser import Document


class BM25Retriever:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.documents: List[Document] = []
        self._tokenized_docs: List[List[str]] = []
        self._doc_freqs: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._avg_doc_len = 0.0

    def load_documents(self, documents: List[Document]) -> None:
        self.documents = list(documents)
        self._rebuild()

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        self.documents.extend(documents)
        self._rebuild()

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[RetrievedChunk]:
        if not self.documents:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored = []
        for idx, tokens in enumerate(self._tokenized_docs):
            score = self._score(query_terms, tokens)
            if score >= score_threshold and score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        results: List[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(scored[:top_k], start=1):
            results.append(
                RetrievedChunk(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        return results

    def _rebuild(self) -> None:
        self._tokenized_docs = [self._tokenize(doc.content) for doc in self.documents]
        self._doc_freqs = {}
        total_len = 0

        for tokens in self._tokenized_docs:
            total_len += len(tokens)
            for term in set(tokens):
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        doc_count = len(self._tokenized_docs)
        self._avg_doc_len = total_len / doc_count if doc_count else 0.0
        self._idf = {
            term: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in self._doc_freqs.items()
        }

    def _score(self, query_terms: List[str], doc_tokens: List[str]) -> float:
        if not doc_tokens:
            return 0.0

        frequencies = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0
        for term in query_terms:
            freq = frequencies.get(term, 0)
            if freq == 0:
                continue

            idf = self._idf.get(term, 0.0)
            denominator = freq + self.k1 * (
                1.0 - self.b + self.b * doc_len / max(self._avg_doc_len, 1e-9)
            )
            score += idf * (freq * (self.k1 + 1.0)) / denominator
        return score

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        ascii_terms = re.findall(r"[a-z0-9_]+", text)
        cjk_terms = re.findall(r"[\u4e00-\u9fff]", text)
        return ascii_terms + cjk_terms
