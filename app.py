import os
import glob
import json
import urllib.request
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

from config import (
    DEFAULT_RAG_DB,
    DEFAULT_KG_DB,
    FEATURED_MODELS,
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_TOP_K_CHUNKS,
    DEFAULT_GRAPH_TOP_K,
    DEFAULT_GRAPH_HOPS,
    EMBED_MODEL_NAME,
)
from rag_engine import get_embed_model, query_semantic_rag, clear_chunk_cache
from graph_engine import query_graph_rag, clear_kg_cache
from visualization import create_subgraph_network_html
from ingestion import build_databases

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dual RAG Engine: Semantic vs. Knowledge Graph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Clean, Modern Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean layout */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Minimalist Stat Pill */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 0.86rem;
        color: #94a3b8;
    }
    .stat-pill b {
        color: #38bdf8;
    }

    /* Chunk Grid Card Styling */
    .chunk-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 10px;
        padding: 14px;
        height: 235px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 8px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .chunk-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        padding-bottom: 4px;
    }
    .chunk-rank {
        font-weight: 700;
        font-size: 0.85rem;
        color: #f8fafc;
    }
    .chunk-score {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border-radius: 5px;
        padding: 2px 7px;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .chunk-meta {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .chunk-text-body {
        font-size: 0.81rem;
        line-height: 1.45;
        color: #cbd5e1;
        overflow-y: auto;
        flex-grow: 1;
        padding-right: 4px;
        word-break: break-word;
    }
    .chunk-text-body::-webkit-scrollbar {
        width: 4px;
    }
    .chunk-text-body::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.3);
        border-radius: 4px;
    }

    /* Clean Answer Box */
    .clean-answer-box {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 3px solid #38bdf8;
        border-radius: 8px;
        padding: 18px 22px;
        font-size: 1rem;
        line-height: 1.6;
        margin: 12px 0 20px 0;
    }
    .clean-answer-box-kg {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 3px solid #a855f7;
        border-radius: 8px;
        padding: 18px 22px;
        font-size: 1rem;
        line-height: 1.6;
        margin: 12px 0 20px 0;
    }

    /* Seed badge */
    .seed-badge {
        display: inline-block;
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px 4px 3px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helper: Database Discovery
# -----------------------------------------------------------------------------
def discover_databases():
    """Find all matching _rag.kuzu and _kg.kuzu files/directories."""
    rag_dbs = glob.glob("*_rag.kuzu")
    kg_dbs = glob.glob("*_kg.kuzu")

    if os.path.exists(DEFAULT_RAG_DB) and DEFAULT_RAG_DB not in rag_dbs:
        rag_dbs.append(DEFAULT_RAG_DB)
    if os.path.exists(DEFAULT_KG_DB) and DEFAULT_KG_DB not in kg_dbs:
        kg_dbs.append(DEFAULT_KG_DB)

    return sorted(list(set(rag_dbs))), sorted(list(set(kg_dbs)))


# -----------------------------------------------------------------------------
# Helper: Fetch Models List
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_available_models():
    """Returns featured models (openai/gpt-oss-20b, gpt-oss-120b) + active free models."""
    models_list = list(FEATURED_MODELS)
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            live_free = [
                m["id"]
                for m in data.get("data", [])
                if ":free" in m["id"]
                or (m.get("pricing", {}).get("prompt") == "0" and m.get("pricing", {}).get("completion") == "0")
            ]
            for lf in sorted(live_free):
                if lf not in models_list:
                    models_list.append(lf)
    except Exception:
        pass
    return models_list


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "semantic_result" not in st.session_state:
    st.session_state["semantic_result"] = None
if "graph_result" not in st.session_state:
    st.session_state["graph_result"] = None
if "has_run" not in st.session_state:
    st.session_state["has_run"] = False

# -----------------------------------------------------------------------------
# Sidebar: Engine Settings & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    # 1. OpenRouter API Key
    st.markdown("##### 🔑 OpenRouter API")
    openrouter_env_key = ""
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            openrouter_env_key = str(st.secrets["OPENROUTER_API_KEY"]).strip()
    except Exception:
        pass
    if not openrouter_env_key:
        openrouter_env_key = os.environ.get("OPENROUTER_API_KEY", "")

    user_api_key = st.text_input(
        "OpenRouter API Key",
        value=openrouter_env_key,
        type="password",
        placeholder="sk-or-v1-...",
        help="Reads from Streamlit Secrets or OPENROUTER_API_KEY in .env by default.",
    )
    if user_api_key:
        os.environ["OPENROUTER_API_KEY"] = user_api_key

    # 2. Model Selection (gpt-oss-20b first and default!)
    st.markdown("##### 🤖 Model Selection")
    all_models = get_available_models()
    default_idx = 0
    if DEFAULT_OPENROUTER_MODEL in all_models:
        default_idx = all_models.index(DEFAULT_OPENROUTER_MODEL)

    selected_model = st.selectbox(
        "LLM Model",
        options=all_models,
        index=default_idx,
        help="Featuring openai/gpt-oss-20b, openai/gpt-oss-120b, and free OpenRouter models.",
    )

    st.divider()

    # 3. Active Databases Selection
    st.markdown("##### 🗄️ Active Databases")
    rag_options, kg_options = discover_databases()

    default_rag_idx = 0
    for i, r in enumerate(rag_options):
        if "cleaned_dataset" in r or r == DEFAULT_RAG_DB:
            default_rag_idx = i
            break

    selected_rag_db = st.selectbox(
        "Vector DB (*_rag.kuzu)",
        options=rag_options if rag_options else [DEFAULT_RAG_DB],
        index=default_rag_idx if rag_options else 0,
    )

    default_kg_idx = 0
    for i, k in enumerate(kg_options):
        if "complete_dataset" in k or k == DEFAULT_KG_DB:
            default_kg_idx = i
            break

    selected_kg_db = st.selectbox(
        "Knowledge Graph DB (*_kg.kuzu)",
        options=kg_options if kg_options else [DEFAULT_KG_DB],
        index=default_kg_idx if kg_options else 0,
    )

    st.divider()

    # 4. Retrieval Sliders
    st.markdown("##### 🎛️ Parameters")
    top_k_chunks = st.slider("Top Chunks (Semantic)", min_value=1, max_value=20, value=DEFAULT_TOP_K_CHUNKS)
    graph_top_k = st.slider("Seed Entities (Graph)", min_value=1, max_value=10, value=DEFAULT_GRAPH_TOP_K)
    graph_hops = st.slider("Traversal Hops (Graph)", min_value=1, max_value=3, value=DEFAULT_GRAPH_HOPS)

    st.divider()
    st.caption("⚡ Dual RAG Engine | KùzuDB + OpenRouter")

# -----------------------------------------------------------------------------
# Top Layout: Clean Title & Minimalist Subtitle
# -----------------------------------------------------------------------------
st.title("Dual RAG Engine: Semantic vs. Knowledge Graph")
st.caption(
    "Compare dense vector chunk similarity search with multi-hop property graph traversal using KùzuDB and OpenRouter models."
)

# -----------------------------------------------------------------------------
# Predefined Questions (100 curated questions across 9 categories)
# -----------------------------------------------------------------------------
PREDEFINED_QUESTIONS = {
    "A. Multi-hop reasoning within one subject": [
        "How do functional dependencies lead to normalization, and how does the resulting schema affect index design?",
        "Explain the chain from ER/EER modeling to relational schema mapping to normal form violations.",
        "How does a transaction's isolation level affect which concurrency control protocol is appropriate, and how does that in turn affect recovery logging?",
        "Trace how a deadlock is detected, and what recovery mechanism is triggered once it's broken.",
        "How do B+-tree indexes support query optimization decisions during query processing?",
        "How does buffer management strategy affect both file organization choice and recovery/logging overhead?",
        "What is the relationship between multivalued dependencies, 4NF, and join dependencies leading to 5NF?",
        "How does the choice of hashing (static vs dynamic) affect both file organization and index maintenance during insert operations?",
        "How do database security mechanisms (access control) interact with concurrency control on shared resources?",
        "Explain how relational algebra operators map to SQL query constructs and then to query execution plans.",
    ],
    "B. Cross-topic bridges within a subject": [
        "How does distributed database fragmentation and replication affect transaction concurrency control protocols?",
        "How do NoSQL data models relate to the CAP theorem trade-offs discussed alongside distributed databases?",
        "How does the MapReduce programming model, as covered in the DBMS big-data chapter, connect to Hadoop's HDFS architecture?",
        "How do active database triggers interact with transaction rollback and recovery?",
        "How does object-relational mapping (ORDBMS) relate to XML data representation in the same book?",
        "How is spatial/temporal database indexing different from the standard B-tree indexing described earlier in the same text?",
        "How does database security link to distributed database authentication mechanisms?",
        "How does information retrieval (IR) evaluation (precision/recall) relate to indexing techniques used for Web search discussed in the same book?",
        "How does query optimization's cost model depend on statistics collected during database tuning?",
        "How does the data mining chapter's association rule mining connect to the data warehousing chapter's OLAP operations in the same text?",
    ],
    "C. Cross-book comparisons": [
        "Compare how DBMS and ADBMS define 'transaction atomicity' versus how BDA treats fault tolerance in distributed computation (MapReduce node failure).",
        "Compare B-tree/B+-tree indexing (DBMS) with the indexing techniques used for near-duplicate detection (BDA's shingling + LSH).",
        "Compare how DWM defines a 'data cube' with how ADBMS defines OLAP support in advanced database models.",
        "Compare clustering algorithms in DWM (k-means, hierarchical, CURE) with clustering of social-network graphs in BDA.",
        "Compare 'association rule mining' as presented in DWM with 'frequent itemsets'/Apriori as presented in BDA.",
        "Compare distributed database replication (ADBMS) with data replication for fault tolerance in MapReduce/HDFS (BDA).",
        "Compare normalization theory (DBMS) with dimensional modeling / star schema design (DWM) — why do warehouses often denormalize?",
        "Compare recommendation systems in BDA with the classification/prediction methods in DWM.",
        "Compare hashing for file organization (DBMS) with hash functions used in stream processing algorithms (BDA, e.g., DGIM, Bloom filters).",
        "Compare security/access control in ADBMS with privacy issues raised around large-scale data mining in BDA/DWM.",
        "Compare PageRank's graph-based ranking (BDA) with database indexing's goal of fast retrieval (DBMS) — what problem does each solve?",
        "Compare NoSQL 'four categories' (ADBMS) with the 'big data v's' framing typically used to justify NoSQL adoption (BDA/DWM).",
        "Compare OLAP star/snowflake schemas (DWM) with object-relational/XML models (ADBMS) as alternative advanced data models.",
        "Compare outlier/anomaly detection (DWM) with spam/fraud detection use-cases discussed in BDA (e.g., link/click fraud).",
        "Compare dimensionality reduction (SVD/LSI in BDA) with attribute subset selection / PCA as a data reduction technique in DWM.",
    ],
    "D. Aggregation / list everything related to X": [
        "List every algorithm across all four books that relies on hashing, and state what each uses it for.",
        "List every technique across the corpus used to reduce dataset size or dimensionality (data reduction).",
        "What are all the distributed/parallel processing concepts mentioned across ADBMS and BDA (e.g., fragmentation, MapReduce, replication)?",
        "List every clustering algorithm mentioned across BDA and DWM.",
        "List every concurrency-control or synchronization mechanism mentioned across DBMS and ADBMS.",
        "What are all the index structures discussed across DBMS, ADBMS, and BDA (B-tree, hash index, bitmap index, LSH, grid file)?",
        "List every evaluation metric mentioned across the books for measuring mining/retrieval quality (precision, recall, confidence, support, RMSE).",
        "What are all the graph-related algorithms mentioned across BDA (PageRank, social network clustering, matching)?",
        "List every normalization or schema-design concept from DBMS alongside every schema concept from DWM (star, snowflake, fact constellation).",
        "What are all the types of 'big data' storage/processing systems named across ADBMS and BDA (NoSQL categories, Hadoop, HDFS, Hive, Pig)?",
        "List every data structure introduced specifically for streaming or approximate computation across BDA.",
        "What are all the security-related topics spread across ADBMS (access control, encryption, distributed security)?",
        "List every kind of 'advanced data model' mentioned across ADBMS and DWM (temporal, spatial, multimedia, deductive, active).",
        "What machine-learning algorithms appear across BDA and DWM (perceptron, SVM, gradient descent, decision trees, k-NN)?",
        "List every optimization or cost-based technique mentioned across DBMS's query optimization chapter and BDA's algorithmic efficiency discussions.",
    ],
    "E. Dependency-chain / 'why' questions": [
        "Why does choosing an isolation level affect which recovery technique a DBMS must use?",
        "Why does the CAP theorem push NoSQL systems away from strict ACID guarantees used in traditional DBMS?",
        "Why does MapReduce's design assume node failures, and how does that shape its execution model?",
        "Why does data warehousing favor a denormalized star schema over the 3NF designs taught for OLTP systems?",
        "Why is the Apriori algorithm's 'monotonicity of itemsets' property essential for its pruning strategy?",
        "Why do LSH techniques trade off exactness for scalability compared to exact index lookups in a DBMS?",
        "Why does PageRank need 'teleporting' to handle dead ends and spider traps?",
        "Why does 2PL (two-phase locking) guarantee serializability, and what failure mode does it not prevent?",
        "Why does distributed database replication improve availability but complicate consistency?",
        "Why is dimensionality reduction (SVD) useful before applying clustering or recommendation algorithms?",
        "Why does bitmap indexing work well for OLAP queries but poorly for high-cardinality OLTP columns?",
        "Why does the decaying window model matter for mining data streams rather than static batch mining?",
        "Why does query optimization rely on selectivity estimates, and how do outdated statistics break this?",
        "Why is a multivalued dependency insufficient to prevent redundancy without 4NF, motivating join dependencies and 5NF?",
        "Why does the Balance Algorithm outperform greedy in the online bipartite matching / AdWords problem?",
    ],
    "F. Compare and contrast": [
        "Contrast transaction processing (DBMS) with batch analytical processing (DWM/BDA) in terms of latency, consistency, and use case.",
        "Contrast structured query languages (SQL) with the algorithmic/procedural style of MapReduce jobs.",
        "Contrast schema-on-write (traditional DBMS) with schema-on-read (NoSQL/big data systems).",
        "Contrast supervised classification (DWM) with unsupervised clustering (BDA) using the same 'labeling' concept from both books.",
        "Contrast centralized recovery/logging (DBMS) with fault-tolerant recomputation in distributed frameworks (BDA/ADBMS).",
        "Contrast relational algebra operators with graph traversal operations implicit in social network analysis (BDA).",
        "Contrast OLTP system design goals with OLAP/data warehouse design goals as presented across DBMS and DWM.",
        "Contrast market-basket association rules (BDA) with sequential pattern / multilevel association mining (DWM).",
        "Contrast a data mart vs. an enterprise data warehouse (DWM) with a fragmented vs. replicated distributed database (ADBMS).",
        "Contrast the accuracy-vs-speed trade-off in approximate stream algorithms (BDA) with exact ACID guarantees in classical DBMS.",
    ],
    "G. Synthesis / design a system": [
        "Design a system architecture that combines a transactional DBMS with a data warehouse for analytics — what components from each book would you use?",
        "How would you design indexing for a hybrid workload combining OLTP queries and OLAP-style aggregations?",
        "Propose a pipeline to detect near-duplicate documents at scale, citing techniques from both DBMS-style indexing and BDA's shingling/LSH.",
        "Design a recommendation engine architecture that uses distributed storage concepts from ADBMS and the recommendation algorithms from BDA.",
        "How would you architect a system to maintain consistency across replicated nodes while also supporting fast big-data analytics jobs?",
        "Propose a strategy for securing a distributed, big-data warehouse combining access control (ADBMS) and privacy-preserving data mining (DWM).",
        "Design a concurrency control scheme suitable for a system also running long analytical MapReduce jobs concurrently with OLTP transactions.",
        "How would clustering (DWM/BDA) be used to optimize physical database design/index selection (DBMS/ADBMS)?",
        "Propose an ETL pipeline connecting an operational DBMS to a data warehouse's star schema, citing normalization and denormalization trade-offs.",
        "Design a fault-tolerant data pipeline that uses HDFS replication (BDA) alongside a NoSQL system's own replication strategy (ADBMS) — what redundancy is avoided or duplicated?",
    ],
    "H. Enumerated relationship / entity-linking": [
        "Which chapters or concepts across these four books are prerequisites for understanding data warehousing's OLAP implementation?",
        "Which specific algorithms in BDA depend on hash functions first introduced conceptually in DBMS?",
        "Which concepts in ADBMS's distributed database chapter are direct extensions of the single-node concurrency control chapter in DBMS?",
        "Which data mining techniques in DWM require the data cube structures defined earlier in the same book, and which are independent?",
        "Which big-data storage systems in ADBMS map onto categories of NoSQL, and which of BDA's algorithms are commonly run on top of them?",
        "What entities/topics does 'indexing' connect to across all four books (B-tree, bitmap, hash, inverted index, LSH)?",
        "What entities/topics does 'similarity' connect to across BDA and DWM (Jaccard similarity, cosine similarity, distance measures, clustering)?",
        "What entities/topics does 'consistency' connect to across DBMS, ADBMS, and BDA (ACID, CAP, eventual consistency, replication protocols)?",
        "What entities/topics does 'aggregation' connect to across DWM and BDA (data cube aggregation, MapReduce grouping/aggregation)?",
        "What entities/topics does 'graph' connect to across BDA (PageRank, social networks, matching) versus DBMS's ER diagrams and EER graphs?",
    ],
    "I. Negative/absence questions": [
        "Does the transaction recovery protocol in DBMS have any direct dependency on the PageRank algorithm in BDA?",
        "Is there a described relationship between XML data modeling (ADBMS) and frequent pattern mining (DWM), or are they independent topics?",
        "Does data warehousing's bitmap indexing depend on the locality-sensitive hashing techniques from BDA, or are they unrelated indexing methods?",
        "Are the concurrency control protocols in DBMS reused unmodified in the distributed database chapter, or do they require fundamental changes?",
        "Does the Apriori algorithm (BDA/DWM) have any structural dependency on B+-tree indexing (DBMS), or do they solve unrelated problems?",
    ],
}

# Build a flat list for the selectbox: "— Type your own question —" + grouped questions
CUSTOM_QUESTION_OPTION = "— Type your own question —"

def _build_question_options():
    """Build a flat list of questions with category headers for the selectbox."""
    options = [CUSTOM_QUESTION_OPTION]
    for category, questions in PREDEFINED_QUESTIONS.items():
        for q in questions:
            options.append(q)
    return options

QUESTION_OPTIONS = _build_question_options()

# Build category lookup for formatting
_question_to_category = {}
for _cat, _qs in PREDEFINED_QUESTIONS.items():
    for _q in _qs:
        _question_to_category[_q] = _cat

# -----------------------------------------------------------------------------
# Clean Global Query Box (Dropdown + Custom Text Input)
# -----------------------------------------------------------------------------
st.markdown("##### 📋 Select a question or type your own")

selected_question = st.selectbox(
    "Choose a predefined question",
    options=QUESTION_OPTIONS,
    index=0,
    format_func=lambda x: x if x == CUSTOM_QUESTION_OPTION else f"[{_question_to_category.get(x, '').split('.')[0].strip()}] {x[:120]}{'...' if len(x) > 120 else ''}",
    label_visibility="collapsed",
    help="Pick from 100 curated questions across 9 categories, or select the first option to type your own.",
)

with st.form("query_form", clear_on_submit=False):
    if selected_question == CUSTOM_QUESTION_OPTION:
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            query_input = st.text_input(
                "Query Box",
                value="",
                placeholder="Type your own question here...",
                label_visibility="collapsed",
            )
        with col_btn:
            execute_button = st.form_submit_button("Search", type="primary", use_container_width=True)
    else:
        query_input = selected_question
        st.markdown(
            f'<div style="background: rgba(30,41,59,0.45); border: 1px solid rgba(148,163,184,0.18); '
            f'border-radius: 8px; padding: 10px 14px; font-size: 0.95rem; color: #e2e8f0; margin-bottom: 8px;">'
            f'<span style="color: #94a3b8; font-size: 0.78rem;">{_question_to_category.get(selected_question, "")}</span><br/>'
            f'{selected_question}</div>',
            unsafe_allow_html=True,
        )
        execute_button = st.form_submit_button("Search", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Query Execution Logic
# -----------------------------------------------------------------------------
if execute_button and query_input.strip():
    embed_model = get_embed_model(EMBED_MODEL_NAME)

    with st.spinner("Executing Semantic and Graph pipelines..."):
        # 1. Semantic RAG
        semantic_res = query_semantic_rag(
            question=query_input.strip(),
            db_path=selected_rag_db,
            embed_model=embed_model,
            openrouter_api_key=user_api_key,
            model=selected_model,
            k=top_k_chunks,
        )
        st.session_state["semantic_result"] = semantic_res

        # 2. Graph RAG
        graph_res = query_graph_rag(
            question=query_input.strip(),
            db_path=selected_kg_db,
            embed_model=embed_model,
            openrouter_api_key=user_api_key,
            model=selected_model,
            top_k=graph_top_k,
            hops=graph_hops,
        )
        st.session_state["graph_result"] = graph_res
        st.session_state["has_run"] = True

# -----------------------------------------------------------------------------
# Main Tabs: Semantic RAG vs. Graph RAG
# -----------------------------------------------------------------------------
tab_semantic, tab_graph = st.tabs(["Semantic RAG", "Graph RAG"])

# =============================================================================
# TAB 1: SEMANTIC RAG
# =============================================================================
with tab_semantic:
    sem_data = st.session_state.get("semantic_result")

    if sem_data is None:
        st.info("Enter your question above and click **Search** to run the Semantic RAG pipeline.")
    else:
        top_chunks = sem_data.get("top_k", [])
        exec_time = sem_data.get("execution_time", 0)
        top_score = top_chunks[0]["score"] if top_chunks else 0.0

        # Compact stats summary
        st.markdown(
            f"""
            <div style="margin-bottom: 12px; display: flex; gap: 10px; flex-wrap: wrap;">
                <div class="stat-pill">Chunks: <b>{len(top_chunks)}</b></div>
                <div class="stat-pill">Top Score: <b>{top_score:.4f}</b></div>
                <div class="stat-pill">Model: <b>{selected_model}</b></div>
                <div class="stat-pill">Latency: <b>{exec_time:.2f}s</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Generated Answer
        st.markdown("#### Generated Answer")
        st.markdown(
            f'<div class="clean-answer-box">{sem_data.get("llm_answer", "")}</div>',
            unsafe_allow_html=True,
        )

        # Prompt Provided to LLM
        with st.expander("🔎 Prompt Provided to LLM"):
            st.code(sem_data.get("prompt_sent", ""), language="markdown")

        # 2. Retrieved Chunk Cards Grid (4-5 cards per row, table completely removed)
        st.markdown("#### Retrieved Chunks")

        if top_chunks:
            cards_per_row = 5 if len(top_chunks) >= 5 else max(len(top_chunks), 1)

            # Render chunks in rows of 4-5 cards
            for row_start in range(0, len(top_chunks), cards_per_row):
                row_chunks = top_chunks[row_start : row_start + cards_per_row]
                cols = st.columns(len(row_chunks))

                for idx, c in enumerate(row_chunks):
                    rank = row_start + idx + 1
                    with cols[idx]:
                        st.markdown(
                            f"""
                            <div class="chunk-card">
                                <div>
                                    <div class="chunk-header">
                                        <span class="chunk-rank">#{rank} Chunk</span>
                                        <span class="chunk-score">Sim: {c['score']:.4f}</span>
                                    </div>
                                    <div class="chunk-meta">📄 {c['source']} · p.{c['page']}</div>
                                </div>
                                <div class="chunk-text-body">{c['text']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.popover("Full text", use_container_width=True):
                            st.caption(f"Source: {c['source']} (Page {c['page']}) | Cosine Sim: {c['score']:.4f}")
                            st.write(c["text"])
        else:
            st.warning("No chunks were retrieved.")

        # Raw Context Expander
        with st.expander("View Flattened Semantic Context"):
            st.text_area("Context String", value=sem_data.get("context", ""), height=180, disabled=True)


# =============================================================================
# TAB 2: GRAPH RAG
# =============================================================================
with tab_graph:
    graph_data = st.session_state.get("graph_result")

    if graph_data is None:
        st.info("Enter your question above and click **Search** to run the Graph RAG pipeline.")
    else:
        seeds_df = graph_data.get("seeds_df", pd.DataFrame())
        subgraph_df = graph_data.get("subgraph_df", pd.DataFrame())
        transform_stats = graph_data.get("transform_stats", {})
        exec_time = graph_data.get("execution_time", 0)

        # Compact stats summary
        st.markdown(
            f"""
            <div style="margin-bottom: 12px; display: flex; gap: 10px; flex-wrap: wrap;">
                <div class="stat-pill">Seed Entities: <b>{len(seeds_df)}</b></div>
                <div class="stat-pill">Traversed Triples: <b>{len(subgraph_df)}</b></div>
                <div class="stat-pill">Deduplicated: <b>{transform_stats.get('total_deduped', 0)}</b></div>
                <div class="stat-pill">Latency: <b>{exec_time:.2f}s</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Generated Answer
        st.markdown("#### Generated Answer")
        st.markdown(
            f'<div class="clean-answer-box-kg">{graph_data.get("llm_answer", "")}</div>',
            unsafe_allow_html=True,
        )

        # Prompt Provided to LLM
        with st.expander("🔎 Prompt Provided to LLM"):
            st.code(graph_data.get("prompt_sent", ""), language="markdown")

        # 2. Seed Nodes Inspection
        st.markdown("#### Seed Entities")
        if not seeds_df.empty:
            chips_html = "".join([f'<span class="seed-badge">★ {name}</span>' for name in seeds_df["name"]])
            st.markdown(f'<div style="margin-bottom: 12px;">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.warning("No seed entities matched.")

        # 3. Retrieved Subgraph Inspection
        st.markdown(f"#### Subgraph ({graph_hops}-Hop Traversal)")
        if not subgraph_df.empty:
            col_viz, col_table = st.columns([1, 1])

            with col_viz:
                seed_names = seeds_df["name"].tolist() if not seeds_df.empty else []
                html_net = create_subgraph_network_html(subgraph_df, seed_names=seed_names, max_triples=50)
                st.components.v1.html(html_net, height=450, scrolling=False)

            with col_table:
                st.dataframe(
                    subgraph_df.rename(columns={
                        "head": "Head",
                        "relation": "Relation",
                        "tail": "Tail",
                        "source": "Source",
                        "page": "Page",
                    }),
                    use_container_width=True,
                    height=450,
                    hide_index=True,
                )
        else:
            st.warning("No graph triples found for the seed entities.")

        # 4. Graph Flattening / Context Transformation
        with st.expander("View Flattened Knowledge Graph Context"):
            st.caption(
                f"{transform_stats.get('total_raw', 0)} raw triples ➔ "
                f"{transform_stats.get('total_deduped', 0)} deduplicated triples."
            )
            st.code(graph_data.get("flattened_context", ""), language="markdown")


# =============================================================================
# PERSISTENT BOTTOM SECTION: DYNAMIC DB INGESTION ENGINE
# =============================================================================
st.markdown("---")
with st.expander("👉 Try it on your own files", expanded=False):
    st.markdown("##### Ingest Custom Documents (.pdf, .txt, .json)")
    st.caption("Upload documents to build both `<prefix>_rag.kuzu` (Vector DB) and `<prefix>_kg.kuzu` (Knowledge Graph DB).")

    col_files, col_prefix = st.columns([3, 2])
    with col_files:
        uploaded_files = st.file_uploader(
            "Select files",
            accept_multiple_files=True,
            type=["pdf", "txt", "json"],
            label_visibility="collapsed",
        )
    with col_prefix:
        custom_db_prefix = st.text_input(
            "Database prefix name",
            value="user_project",
            placeholder="e.g. user_project",
        )

    if st.button("Build Dual Databases", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one `.pdf`, `.txt`, or `.json` file.")
        elif not custom_db_prefix.strip():
            st.error("Please specify a database prefix name.")
        else:
            clean_prefix = "".join(c for c in custom_db_prefix.strip() if c.isalnum() or c in ("_", "-"))
            prog_bar = st.progress(0.0)
            status_box = st.empty()

            try:
                embed_model = get_embed_model(EMBED_MODEL_NAME)

                def update_cb(pct, msg):
                    prog_bar.progress(pct)
                    status_box.caption(msg)

                new_rag_path, new_kg_path, stats = build_databases(
                    uploaded_files=uploaded_files,
                    db_prefix=clean_prefix,
                    embed_model=embed_model,
                    progress_callback=update_cb,
                )

                clear_chunk_cache(new_rag_path)
                clear_kg_cache(new_kg_path)
                st.success(f"Created `{new_rag_path}` and `{new_kg_path}` ({stats['total_chunks']} chunks, {stats['total_entities']} entities, {stats['total_edges']} edges).")
                st.toast("Databases created successfully!", icon="✅")
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")
