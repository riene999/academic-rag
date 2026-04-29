import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import PyPDF2
from loguru import logger


@dataclass
class Document:
    content: str          # 文本内容
    metadata: Dict        # 元信息：来源文件、页码、chunk_id等
    chunk_id: str         # 唯一标识


class PDFParser:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(self, pdf_path: str, source_name: str = None) -> List[Document]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        logger.info(f"解析PDF: {path.name}")
        full_text = ""
        page_map = []  # 记录每个字符对应的页码，用于元信息

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                start = len(full_text)
                full_text += text + "\n"
                page_map.append((start, len(full_text), page_num + 1))

        chunks = self._split_text(full_text)
        documents = []

        for i, chunk in enumerate(chunks):
            # 找到chunk对应的页码
            chunk_start = full_text.find(chunk[:50])  # 用前50字符定位
            page_num = self._find_page(chunk_start, page_map)

            documents.append(Document(
                content=chunk,
                metadata={
                    "source": source_name or path.name,
                    "page": page_num,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
                chunk_id=f"{path.stem}_chunk_{i}",
            ))

        logger.info(f"解析完成：{len(documents)} 个chunks")
        return documents

    def parse_text(self, text: str, source_name: str = "manual_input") -> List[Document]:
        chunks = self._split_text(text)
        return [
            Document(
                content=chunk,
                metadata={"source": source_name, "chunk_index": i},
                chunk_id=f"{source_name}_chunk_{i}",
            )
            for i, chunk in enumerate(chunks)
        ]

    def _split_text(self, text: str) -> List[str]:
        # 递归字符分割策略（参考LangChain RecursiveCharacterTextSplitter）
        # 优先按段落切，其次按句子切，最后按字符切
        # 保证每个chunk不超过chunk_size，相邻chunk有chunk_overlap重叠

        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        # 按段落优先分割
        separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                current_chunk = ""
                for part in parts:
                    if len(current_chunk) + len(part) + len(sep) <= self.chunk_size:
                        current_chunk += part + sep
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = part + sep
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                break

        if not chunks:
            # 兜底：强制按长度切
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i:i + self.chunk_size])

        # 添加overlap：每个chunk的开头加上上一个chunk的末尾
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                overlap_text = chunks[i-1][-self.chunk_overlap:]
                overlapped.append(overlap_text + chunks[i])
            return overlapped

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)          # 合并多余空白
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # 修复断行连字符
        return text.strip()

    def _find_page(self, char_pos: int, page_map: List) -> int:
        for start, end, page in page_map:
            if start <= char_pos < end:
                return page
        return 1
