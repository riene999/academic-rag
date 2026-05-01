import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import PyPDF2
from loguru import logger


@dataclass
class Document:
    content: str
    metadata: Dict
    chunk_id: str


class PDFParser:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_path: str, source_name: str = None) -> List[Document]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        logger.info("Parsing PDF: {}", path.name)
        full_text = ""
        page_map = []
        source = source_name or path.name
        paper_title = Path(source).stem

        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            paper_title = extract_paper_title(reader, fallback=source)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                start = len(full_text)
                full_text += text + "\n"
                page_map.append((start, len(full_text), page_num + 1))

        chunks = self._split_text(full_text)
        documents = []
        for idx, chunk in enumerate(chunks):
            chunk_start = full_text.find(chunk[:50])
            page_num = self._find_page(chunk_start, page_map)
            documents.append(
                Document(
                    content=chunk,
                    metadata={
                        "source": source,
                        "paper_title": paper_title,
                        "page": page_num,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                    },
                    chunk_id=f"{path.stem}_chunk_{idx}",
                )
            )

        logger.info("Parsed {} chunks from {}", len(documents), path.name)
        return documents

    def parse_text(self, text: str, source_name: str = "manual_input") -> List[Document]:
        chunks = self._split_text(text)
        return [
            Document(
                content=chunk,
                metadata={
                    "source": source_name,
                    "paper_title": source_name,
                    "chunk_index": idx,
                },
                chunk_id=f"{source_name}_chunk_{idx}",
            )
            for idx, chunk in enumerate(chunks)
        ]

    def _split_text(self, text: str) -> List[str]:
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        separators = ["\n\n", "\n", "。", "；", "，", ".", "!", "?", " ", ""]

        for sep in separators:
            if sep and sep not in text:
                continue
            if sep:
                parts = text.split(sep)
                current_chunk = ""
                for part in parts:
                    addition = part + sep
                    if len(current_chunk) + len(addition) <= self.chunk_size:
                        current_chunk += addition
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = addition
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
            else:
                step = max(1, self.chunk_size - self.chunk_overlap)
                for idx in range(0, len(text), step):
                    chunks.append(text[idx : idx + self.chunk_size])
            break

        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for idx in range(1, len(chunks)):
                overlap_text = chunks[idx - 1][-self.chunk_overlap :]
                overlapped.append(overlap_text + chunks[idx])
            return overlapped

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _find_page(self, char_pos: int, page_map: List) -> int:
        for start, end, page in page_map:
            if start <= char_pos < end:
                return page
        return 1


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", Path(value or "").stem).lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_paper_title(reader: PyPDF2.PdfReader, fallback: str) -> str:
    metadata_title = ""
    try:
        metadata_title = str(getattr(reader.metadata, "title", "") or "").strip()
    except Exception:
        metadata_title = ""

    if _looks_like_title(metadata_title):
        return unicodedata.normalize("NFKC", metadata_title)

    first_page_text = ""
    try:
        if reader.pages:
            first_page_text = reader.pages[0].extract_text() or ""
    except Exception:
        first_page_text = ""

    title = _extract_title_from_first_page(first_page_text)
    if title:
        return unicodedata.normalize("NFKC", title)
    return unicodedata.normalize("NFKC", Path(fallback).stem)


def _extract_title_from_first_page(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    abstract_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if re.fullmatch(r"(?i)\s*abstract\s*[:.\-]?\s*", line)
            or re.match(r"(?i)\s*abstract\s*[:.\-]\s+", line)
        ),
        min(len(lines), 14),
    )
    candidates = [
        line
        for line in lines[:abstract_idx]
        if _looks_like_title(line)
        and not re.search(
            r"(?i)(arxiv|proceedings|conference|workshop|university|@|copyright|doi)",
            line,
        )
    ]
    if not candidates:
        return ""

    joined = []
    for line in candidates[:3]:
        joined.append(line)
        title = " ".join(joined)
        if len(title) >= 30:
            return title[:240]
    return " ".join(joined)[:240]


def _looks_like_title(value: str) -> bool:
    value = re.sub(r"\s+", " ", value or "").strip()
    if len(value) < 12 or len(value) > 260:
        return False
    if re.search(r"@|https?://|doi\.org", value, re.I):
        return False
    return len(value.split()) >= 3
