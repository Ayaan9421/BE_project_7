import os
import time
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import kuzu
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import OPENROUTER_BASE_URL, DEFAULT_OPENROUTER_MODEL

_KG_DB_CACHE: Dict[str, kuzu.Database] = {}


def get_kuzu_connection(db_path: str, read_only: bool = True) -> Tuple[kuzu.Database, kuzu.Connection]:
    """Create or reuse a Kùzu database and connection instance in read_only mode."""
    if db_path not in _KG_DB_CACHE:
        _KG_DB_CACHE[db_path] = kuzu.Database(db_path, read_only=read_only)
    db = _KG_DB_CACHE[db_path]
    conn = kuzu.Connection(db)
    return db, conn


def clear_kg_cache(db_path: str = None):
    """Clear cached database instances."""
    if db_path:
        _KG_DB_CACHE.pop(db_path, None)
    else:
        _KG_DB_CACHE.clear()


def get_seed_nodes(
    query: str,
    conn: kuzu.Connection,
    embed_model: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Fetch seed nodes from HNSW vector index search over Entity nodes in KuzuDB.
    """
    q_vec = embed_model.encode(query, normalize_embeddings=True).tolist()

    result = conn.execute(
        """
        CALL QUERY_VECTOR_INDEX('Entity', 'entity_hnsw', $q_vec, $top_k)
        YIELD node, distance
        RETURN node.entity_id, node.name, distance
        ORDER BY distance ASC
    """,
        {"q_vec": q_vec, "top_k": top_k},
    )

    df = result.get_as_df().rename(
        columns={
            "node.entity_id": "entity_id",
            "node.name": "name",
        }
    )
    if not df.empty and "distance" in df.columns:
        df["similarity_score"] = (1.0 - df["distance"]).clip(lower=0.0, upper=1.0)
    return df


def get_subgraph(
    seed_entity_ids: List[str],
    conn: kuzu.Connection,
    hops: int = 2,
) -> pd.DataFrame:
    """
    Traverse multi-hop relationships starting from seed entities.
    """
    if not seed_entity_ids:
        return pd.DataFrame(columns=["head", "relation", "tail", "source", "page"])

    id_list = "[" + ", ".join(f'"{eid}"' for eid in seed_entity_ids) + "]"

    query = f"""
        MATCH (h:Entity)-[r:Graph*1..{hops}]->(t:Entity)
        WHERE h.entity_id IN {id_list}
        UNWIND relationships(r) AS rel
        RETURN DISTINCT
            h.name          AS head,
            rel.relation    AS relation,
            t.name          AS tail,
            rel.source_file AS source,
            rel.page_number AS page
    """
    result = conn.execute(query)
    return result.get_as_df()


def subgraph_to_context(subgraph_df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    """
    Converts a subgraph DataFrame into a clean, LLM-readable context string.
    - Deduplicates identical (head, relation, tail) triples
    - Groups triples by source file
    - Within each group, sorts by head entity then relation
    """
    if subgraph_df is None or subgraph_df.empty:
        return "No relevant knowledge graph context found.", {
            "total_raw": 0,
            "total_deduped": 0,
            "duplicates_removed": 0,
            "sources": [],
        }

    original_count = len(subgraph_df)

    # 1. Deduplicate on triple itself; keep first occurrence
    deduped = (
        subgraph_df.drop_duplicates(subset=["head", "relation", "tail"])
        .sort_values(["source", "page", "head", "relation"])
    )

    total_deduped = len(deduped)
    duplicates_removed = original_count - total_deduped

    # 2. Group by source file and build sections
    sections = []
    sources = []
    for source, group in deduped.groupby("source", sort=True):
        sources.append(str(source))
        header = f"[Source: {source}]"
        triples = []
        for _, row in group.iterrows():
            triples.append(
                f"  - {row['head']} --[{row['relation']}]--> {row['tail']}  (p.{row['page']})"
            )
        sections.append(header + "\n" + "\n".join(triples))

    summary = f"Knowledge Graph Context — {total_deduped} triples"
    if duplicates_removed > 0:
        summary += f" ({duplicates_removed} duplicates removed)"

    formatted_context = summary + "\n\n" + "\n\n".join(sections)

    metadata = {
        "total_raw": original_count,
        "total_deduped": total_deduped,
        "duplicates_removed": duplicates_removed,
        "sources": sources,
    }
    return formatted_context, metadata


def build_openrouter_graph_llm(
    api_key: str = None,
    model: str = DEFAULT_OPENROUTER_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
):
    """
    Constructs ChatOpenAI wrapper configured for OpenRouter API.
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


def query_graph_rag(
    question: str,
    db_path: str,
    embed_model: SentenceTransformer,
    openrouter_api_key: str = None,
    model: str = DEFAULT_OPENROUTER_MODEL,
    top_k: int = 5,
    hops: int = 2,
) -> Dict[str, Any]:
    """
    Performs Knowledge Graph RAG:
    1. Fetches seed nodes via HNSW vector index search.
    2. Retrieves multi-hop subgraph.
    3. Flattens subgraph to structured context.
    4. Generates grounded answer via OpenRouter API.
    """
    start_time = time.time()
    db, conn = get_kuzu_connection(db_path, read_only=True)

    try:
        # Step 1: Seed nodes
        seeds_df = get_seed_nodes(question, conn, embed_model, top_k=top_k)
        seed_ids = seeds_df["entity_id"].tolist() if not seeds_df.empty else []

        # Step 2: Subgraph
        subgraph_df = get_subgraph(seed_ids, conn, hops=hops)

        # Step 3: Flatten Graph to Context
        flattened_context, transform_stats = subgraph_to_context(subgraph_df)

        # Step 4: Prompt Template as in GraphRAG.ipynb
        generation_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a knowledgeable assistant. Use ONLY the knowledge graph triples below to answer the question.
Each triple is in the form: (head) -[relation]-> (tail)  [src: <file>, p.<page>]
If the triples do not contain enough information, say so clearly.

Knowledge Graph Context:
{context}

Question: {question}

Answer:""",
        )
        prompt_sent = generation_prompt.format(context=flattened_context, question=question)

        # Step 5: LLM Generation via OpenRouter
        llm_answer = ""
        llm_error = None

        llm = build_openrouter_graph_llm(api_key=openrouter_api_key, model=model)
        if llm is None:
            llm_answer = (
                "ℹ️ **OpenRouter API Key required for answer generation.**\n\n"
                "Add your `OPENROUTER_API_KEY` to `.env` or input it in the sidebar settings. "
                "Seed nodes and multi-hop triples are shown below."
            )
        else:
            try:
                graph_chain = generation_prompt | llm | StrOutputParser()
                response = graph_chain.invoke({"context": flattened_context, "question": question})
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
            "seeds_df": seeds_df,
            "selected_seed_ids": seed_ids,
            "subgraph_df": subgraph_df,
            "flattened_context": flattened_context,
            "transform_stats": transform_stats,
            "llm_answer": llm_answer,
            "execution_time": execution_time,
            "error": llm_error,
            "prompt_sent": prompt_sent,
        }
    finally:
        del conn
