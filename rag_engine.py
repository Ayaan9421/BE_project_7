import os
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import kuzu
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import OPENROUTER_BASE_URL, DEFAULT_OPENROUTER_MODEL

_EMBED_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_CHUNK_CACHE: Dict[str, Tuple[List[Dict[str, Any]], np.ndarray]] = {}
_DB_CACHE: Dict[str, kuzu.Database] = {}


def get_embed_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the SentenceTransformer model in memory."""
    if model_name not in _EMBED_MODEL_CACHE:
        _EMBED_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBED_MODEL_CACHE[model_name]


def get_kuzu_connection(db_path: str, read_only: bool = True) -> Tuple[kuzu.Database, kuzu.Connection]:
    """Create or reuse a Kùzu database and connection instance in read_only mode."""
    if db_path not in _DB_CACHE:
        _DB_CACHE[db_path] = kuzu.Database(db_path, read_only=read_only)
    db = _DB_CACHE[db_path]
    conn = kuzu.Connection(db)
    return db, conn


def load_chunks_and_embeddings(conn: kuzu.Connection, db_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Fetch all chunks and embeddings from KuzuDB with memory caching for sub-second retrieval.
    """
    if db_path in _CHUNK_CACHE:
        return _CHUNK_CACHE[db_path]

    result = conn.execute("MATCH (c:Chunk) RETURN c.chunk_id, c.text, c.source_file, c.page_number, c.embedding")
    df = result.get_as_df()

    if df.empty:
        return [], np.empty((0, 384))

    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "chunk_id": str(row["c.chunk_id"]),
            "text": str(row["c.text"]),
            "source": str(row["c.source_file"]),
            "page": int(row["c.page_number"]),
        })

    emb_matrix = np.vstack(df["c.embedding"].values).astype(np.float32)
    _CHUNK_CACHE[db_path] = (metadata, emb_matrix)
    return metadata, emb_matrix


def clear_chunk_cache(db_path: str = None):
    """Clear memory chunk cache and database cache."""
    if db_path:
        _CHUNK_CACHE.pop(db_path, None)
        _DB_CACHE.pop(db_path, None)
    else:
        _CHUNK_CACHE.clear()
        _DB_CACHE.clear()


def build_openrouter_llm(
    api_key: str = None,
    model: str = DEFAULT_OPENROUTER_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
):
    """
    Constructs a ChatOpenAI wrapper configured for OpenRouter API.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None

    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Dual RAG Engine",
        },
    )


def query_semantic_rag(
    question: str,
    db_path: str,
    embed_model: SentenceTransformer,
    openrouter_api_key: str = None,
    model: str = DEFAULT_OPENROUTER_MODEL,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Performs Semantic RAG:
    1. Encodes question into 384-d vector.
    2. Computes Cosine Similarity against Chunk embeddings in KuzuDB.
    3. Retrieves top-k chunks.
    4. Generates response using OpenRouter API.
    """
    start_time = time.time()

    if not os.path.exists(db_path):
        return {
            "top_k": [],
            "context": "",
            "llm_answer": (
                f"ℹ️ **Vector database not found at `{db_path}`.**\n\n"
                "Please build or initialize a database using the **'👉 Try it on your own files'** section below."
            ),
            "execution_time": 0.0,
            "error": "Database not found on disk",
            "prompt_sent": "",
        }

    db, conn = get_kuzu_connection(db_path, read_only=True)

    try:
        # Step 1: Embed question
        q_vec = np.array(embed_model.encode(question).tolist(), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-9

        # Step 2: Fetch chunks & compute cosine similarity
        metadata, emb_matrix = load_chunks_and_embeddings(conn, db_path)

        if len(metadata) == 0:
            return {
                "top_k": [],
                "context": "No chunks found in database.",
                "llm_answer": "No chunks available in the database.",
                "execution_time": time.time() - start_time,
                "error": None,
                "prompt_sent": "",
            }

        norms = np.linalg.norm(emb_matrix, axis=1) + 1e-9
        sims = np.dot(emb_matrix, q_vec) / (norms * q_norm)

        top_indices = np.argsort(sims)[::-1][:k]
        top_k = []
        for idx in top_indices:
            item = dict(metadata[idx])
            item["score"] = float(sims[idx])
            top_k.append(item)

        # Step 3: Format retrieved chunks into context string
        context = "\n\n".join([item["text"] for item in top_k])

        # Step 4: Prompt Template as in BE_RAG.ipynb
        template = """You are a knowledgeable assistant. Use the retrieved chunks below to structure and generate a comprehensive answer.:
{context}

Question: {question}
"""
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt_sent = template.format(context=context, question=question)

        # Step 5: LLM Execution via OpenRouter
        llm_answer = ""
        llm_error = None

        llm = build_openrouter_llm(api_key=openrouter_api_key, model=model)
        if llm is None:
            llm_answer = (
                "ℹ️ **OpenRouter API Key required for answer generation.**\n\n"
                "Add your `OPENROUTER_API_KEY` to `.env` or input it in the sidebar settings. "
                "Retrieved chunks and similarity scores are shown below."
            )
        else:
            try:
                rag_chain = prompt_template | llm | StrOutputParser()
                response = rag_chain.invoke({"context": context, "question": question})
                llm_answer = response.strip() if isinstance(response, str) else str(response)
                if not llm_answer:
                    llm_answer = "*(Model returned an empty response. Please check model parameters or try another model in settings.)*"
            except Exception as e:
                err_msg = str(e)
                llm_error = err_msg
                llm_answer = (
                    f"⚠️ **OpenRouter API Error:**\n\n`{err_msg}`\n\n"
                    "*(Please check your OpenRouter API key and model selection in the sidebar.)*"
                )

        execution_time = time.time() - start_time
        return {
            "top_k": top_k,
            "context": context,
            "llm_answer": llm_answer,
            "execution_time": execution_time,
            "error": llm_error,
            "prompt_sent": prompt_sent,
        }
    finally:
        del conn
