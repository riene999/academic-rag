"""SQLite-backed metadata storage for indexed documents and chunks."""

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List

from src.utils.pdf_parser import Document


class SQLiteDocumentStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER,
                    metadata_json TEXT NOT NULL,
                    vector_index INTEGER NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS index_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_source_page ON chunks(page, chunk_index)"
            )

    def has_chunks(self) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
            return bool(row and row["count"] > 0)

    def replace_all_documents(self, documents: Iterable[Document], reason: str = "save") -> None:
        docs = list(documents)
        with self._connect() as connection:
            connection.execute("BEGIN")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")

            source_to_document_id: dict[str, int] = {}
            for vector_index, doc in enumerate(docs):
                source_name = str(doc.metadata.get("source") or "unknown")
                document_id = source_to_document_id.get(source_name)
                if document_id is None:
                    cursor = connection.execute(
                        "INSERT INTO documents (source_name, status) VALUES (?, 'active')",
                        (source_name,),
                    )
                    document_id = int(cursor.lastrowid)
                    source_to_document_id[source_name] = document_id

                metadata_json = json.dumps(doc.metadata or {}, ensure_ascii=False)
                connection.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        chunk_id,
                        content,
                        page,
                        chunk_index,
                        metadata_json,
                        vector_index
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        doc.chunk_id,
                        doc.content,
                        doc.metadata.get("page"),
                        doc.metadata.get("chunk_index"),
                        metadata_json,
                        vector_index,
                    ),
                )

            next_version = self.current_version(connection) + 1
            connection.execute(
                "INSERT INTO index_versions (version, reason) VALUES (?, ?)",
                (next_version, reason),
            )
            connection.commit()

    def load_documents(self) -> List[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, content, metadata_json, vector_index
                FROM chunks
                ORDER BY vector_index ASC
                """
            ).fetchall()

        return [
            Document(
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
                chunk_id=row["chunk_id"],
            )
            for row in rows
        ]

    def current_version(self, connection: sqlite3.Connection | None = None) -> int:
        owns_connection = connection is None
        if connection is None:
            connection = self._connect()
        try:
            row = connection.execute(
                "SELECT version FROM index_versions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return int(row["version"]) if row else 0
        finally:
            if owns_connection:
                connection.close()

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    d.id,
                    d.source_name,
                    d.status,
                    d.created_at,
                    d.updated_at,
                    COUNT(c.id) AS chunk_count,
                    MIN(c.page) AS first_page,
                    MAX(c.page) AS last_page
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC, d.id DESC
                """
            ).fetchall()

        return [dict(row) for row in rows]
