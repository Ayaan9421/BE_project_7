import io
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import kuzu
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import CHUNK_SIZE, CHUNK_STRIDE, EMBED_DIM


def clean_db_path(db_path: str):
    """Safely remove existing Kuzu database files/directories."""
    p = Path(db_path)
    wal = Path(f"{db_path}.wal")
    lock = Path(f"{db_path}.lock")
    for target in [p, wal, lock]:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_file():
            try:
                target.unlink(missing_ok=True)
            except Exception:
                pass


def chunk_text(text: str, size: int = CHUNK_SIZE, stride: int = CHUNK_STRIDE) -> List[str]:
    """Sliding-window, sentence-aware chunker identical to BE_RAG.ipynb."""
    if not text or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= size:
            current = (current + " " + sent).strip()
            continue

        if current:
            chunks.append(current)

        if len(sent) > size:
            for i in range(0, len(sent), size - stride):
                chunks.append(sent[i : i + size])
            current = ""
        else:
            current = sent

    if current:
        chunks.append(current)
    return chunks


def parse_uploaded_file(file_obj) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses an uploaded file (.pdf, .txt, or .json).
    Returns:
    - pages: Flat list of {source_file, page_number, text}
    - explicit_triples: List of {head, relation, tail, source_file, page_number} if present in JSON.
    """
    filename = file_obj.name
    ext = Path(filename).suffix.lower()
    content = file_obj.read()

    pages: List[Dict[str, Any]] = []
    explicit_triples: List[Dict[str, Any]] = []

    if ext == ".txt":
        text_str = content.decode("utf-8", errors="replace")
        # Split on form feed or treat as single page
        parts = text_str.split("\f")
        for idx, part in enumerate(parts, start=1):
            if part.strip():
                pages.append({
                    "source_file": filename,
                    "page_number": idx,
                    "text": part.strip(),
                })

    elif ext == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "source_file": filename,
                    "page_number": idx,
                    "text": text.strip(),
                })

    elif ext == ".json":
        data = json.loads(content.decode("utf-8", errors="replace"))

        # Check if dataset format mirrors colab
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Format 1: {source_file, text: [{text, page_number}]}
                    if "source_file" in item and "text" in item and isinstance(item["text"], list):
                        src = item["source_file"]
                        for page_obj in item["text"]:
                            pages.append({
                                "source_file": src,
                                "page_number": int(page_obj.get("page_number", 1)),
                                "text": str(page_obj.get("text", "")).strip(),
                            })
                    # Format 2: Flat page list {source_file, page_number, text}
                    elif "source_file" in item and "text" in item and isinstance(item["text"], str):
                        pages.append({
                            "source_file": str(item["source_file"]),
                            "page_number": int(item.get("page_number", 1)),
                            "text": str(item["text"]).strip(),
                        })
                    # Format 3: Explicit KG triples {head, relation, tail}
                    elif "head" in item and "relation" in item and "tail" in item:
                        explicit_triples.append({
                            "head": str(item["head"]).strip().lower(),
                            "relation": str(item["relation"]).strip().lower(),
                            "tail": str(item["tail"]).strip().lower(),
                            "source_file": str(item.get("source_file", filename)),
                            "page_number": int(item.get("page_number", 1)),
                        })

    return pages, explicit_triples


def extract_triples_from_text(
    pages: List[Dict[str, Any]],
    max_triples: int = 500,
) -> List[Dict[str, Any]]:
    """
    Extracts informative entity relationship triples from document text when
    explicit triples are not provided. Uses syntactic relation patterns.
    """
    triples = []
    patterns = [
        (r"\b([A-Za-z0-9_\- ]{3,35})\s+is\s+(?:a|an)\s+([A-Za-z0-9_\- ]{3,35})\b", "is a"),
        (r"\b([A-Za-z0-9_\- ]{3,35})\s+part\s+of\s+([A-Za-z0-9_\- ]{3,35})\b", "part of"),
        (r"\b([A-Za-z0-9_\- ]{3,35})\s+used\s+by\s+([A-Za-z0-9_\- ]{3,35})\b", "used by"),
        (r"\b([A-Za-z0-9_\- ]{3,35})\s+subclass\s+of\s+([A-Za-z0-9_\- ]{3,35})\b", "subclass of"),
        (r"\b([A-Za-z0-9_\- ]{3,35})\s+leads\s+to\s+([A-Za-z0-9_\- ]{3,35})\b", "leads to"),
        (r"\b([A-Za-z0-9_\- ]{3,35})\s+consists\s+of\s+([A-Za-z0-9_\- ]{3,35})\b", "consists of"),
        (r"\b([A-Za-z0-9_\- ]{3,35})\s+affects\s+([A-Za-z0-9_\- ]{3,35})\b", "affects"),
        (r"\b([A-Za-z0-9_\- ]{3,35})\s+requires\s+([A-Za-z0-9_\- ]{3,35})\b", "requires"),
    ]

    stop_entities = {"this", "that", "these", "those", "which", "there", "it", "they", "we", "you", "he", "she"}

    for p in pages:
        text = p["text"]
        src = p["source_file"]
        page_no = p["page_number"]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            for pat, rel in patterns:
                matches = re.findall(pat, sent, re.IGNORECASE)
                for h, t in matches:
                    head_clean = h.strip().lower()
                    tail_clean = t.strip().lower()
                    if (
                        len(head_clean) > 2
                        and len(tail_clean) > 2
                        and head_clean not in stop_entities
                        and tail_clean not in stop_entities
                        and head_clean != tail_clean
                    ):
                        triples.append({
                            "head": head_clean,
                            "relation": rel,
                            "tail": tail_clean,
                            "source_file": src,
                            "page_number": page_no,
                        })
                        if len(triples) >= max_triples:
                            return triples
    return triples


def build_databases(
    uploaded_files: list,
    db_prefix: str,
    embed_model: SentenceTransformer,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Processes uploaded documents and creates two KuzuDB database instances:
    1. <db_prefix>_rag.kuzu (Vector Chunk DB)
    2. <db_prefix>_kg.kuzu (Knowledge Graph DB with HNSW index)
    """
    def update_progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    update_progress(0.05, "Parsing uploaded documents...")
    all_pages: List[Dict[str, Any]] = []
    all_triples: List[Dict[str, Any]] = []

    for file_obj in uploaded_files:
        pages, triples = parse_uploaded_file(file_obj)
        all_pages.extend(pages)
        all_triples.extend(triples)

    if not all_pages and not all_triples:
        raise ValueError("No valid text or JSON data found in uploaded files.")

    # ---------------------------------------------------------
    # 1. Build Vector RAG Database
    # ---------------------------------------------------------
    update_progress(0.15, "Chunking documents for Vector DB...")
    all_chunks = []
    for page in all_pages:
        for chunk in chunk_text(page["text"]):
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk,
                "source_file": page["source_file"],
                "page_number": page["page_number"],
            })

    total_chunks = len(all_chunks)
    rag_db_path = f"{db_prefix}_rag.kuzu"
    clean_db_path(rag_db_path)

    update_progress(0.25, f"Computing embeddings for {total_chunks} chunks...")
    rag_db = kuzu.Database(rag_db_path)
    rag_conn = kuzu.Connection(rag_db)

    rag_conn.execute(f"""
        CREATE NODE TABLE IF NOT EXISTS Chunk(
            chunk_id    STRING PRIMARY KEY,
            text        STRING,
            source_file STRING,
            page_number INT64,
            embedding   FLOAT[{EMBED_DIM}]
        );
    """)

    if total_chunks > 0:
        texts = [c["text"] for c in all_chunks]
        chunk_embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=False)

        update_progress(0.50, f"Inserting {total_chunks} chunks into {rag_db_path}...")
        INSERT_CHUNK = """
            CREATE (:Chunk {
                chunk_id:    $chunk_id,
                text:        $text,
                source_file: $source_file,
                page_number: $page_number,
                embedding:   $embedding
            });
        """
        BATCH_SIZE = 500
        for start in range(0, total_chunks, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_chunks)
            for i in range(start, end):
                c = all_chunks[i]
                rag_conn.execute(
                    INSERT_CHUNK,
                    parameters={
                        "chunk_id": c["chunk_id"],
                        "text": c["text"],
                        "source_file": c["source_file"],
                        "page_number": c["page_number"],
                        "embedding": chunk_embeddings[i].tolist(),
                    },
                )

    del rag_conn
    del rag_db

    # ---------------------------------------------------------
    # 2. Build Knowledge Graph Database
    # ---------------------------------------------------------
    update_progress(0.65, "Extracting entities & relations for Knowledge Graph...")
    if not all_triples and all_pages:
        all_triples = extract_triples_from_text(all_pages)

    kg_db_path = f"{db_prefix}_kg.kuzu"
    clean_db_path(kg_db_path)

    kg_db = kuzu.Database(kg_db_path)
    kg_conn = kuzu.Connection(kg_db)

    kg_conn.execute(f"""
        CREATE NODE TABLE IF NOT EXISTS Entity(
            entity_id STRING PRIMARY KEY,
            name STRING,
            embedding FLOAT[{EMBED_DIM}]
        );
    """)

    kg_conn.execute("""
        CREATE REL TABLE IF NOT EXISTS Graph(
            FROM Entity TO Entity,
            relation STRING,
            source_file STRING,
            page_number INT64
        );
    """)

    # Collect unique entities
    unique_entities: Dict[str, str] = {}  # name -> entity_id
    for t in all_triples:
        h = t["head"]
        tail = t["tail"]
        if h not in unique_entities:
            unique_entities[h] = str(uuid.uuid4())
        if tail not in unique_entities:
            unique_entities[tail] = str(uuid.uuid4())

    total_entities = len(unique_entities)
    total_edges = len(all_triples)

    update_progress(0.75, f"Computing embeddings for {total_entities} entities...")
    if total_entities > 0:
        entity_names = list(unique_entities.keys())
        entity_embs = embed_model.encode(entity_names, batch_size=64, normalize_embeddings=True)

        INSERT_ENTITY = """
            CREATE (:Entity {
                entity_id: $entity_id,
                name:      $name,
                embedding: $embedding
            });
        """
        for name, emb in zip(entity_names, entity_embs):
            kg_conn.execute(
                INSERT_ENTITY,
                parameters={
                    "entity_id": unique_entities[name],
                    "name": name,
                    "embedding": emb.tolist(),
                },
            )

        update_progress(0.85, f"Inserting {total_edges} graph relationships...")
        INSERT_EDGE = """
            MATCH (h:Entity {entity_id: $head_id}), (t:Entity {entity_id: $tail_id})
            CREATE (h)-[:Graph {
                relation:    $relation,
                source_file: $source_file,
                page_number: $page_number
            }]->(t);
        """
        for t in all_triples:
            kg_conn.execute(
                INSERT_EDGE,
                parameters={
                    "head_id": unique_entities[t["head"]],
                    "tail_id": unique_entities[t["tail"]],
                    "relation": t["relation"],
                    "source_file": t["source_file"],
                    "page_number": t["page_number"],
                },
            )

        update_progress(0.92, "Creating HNSW vector index on Entity embeddings...")
        try:
            kg_conn.execute("""
                CALL CREATE_VECTOR_INDEX('Entity', 'entity_hnsw', 'embedding', metric := 'cosine');
            """)
        except Exception as e:
            # If already exists or error
            print(f"Vector index notice: {e}")

    del kg_conn
    del kg_db

    update_progress(1.0, "Database initialization complete!")

    stats = {
        "total_pages": len(all_pages),
        "total_chunks": total_chunks,
        "total_entities": total_entities,
        "total_edges": total_edges,
        "rag_db_path": rag_db_path,
        "kg_db_path": kg_db_path,
    }
    return rag_db_path, kg_db_path, stats
