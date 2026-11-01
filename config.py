import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Default Database Paths
DEFAULT_RAG_DB = str(BASE_DIR / "cleaned_dataset_rag.kuzu")
DEFAULT_KG_DB = str(BASE_DIR / "complete_dataset_final_kg.kuzu")

# Embedding Configuration
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# OpenRouter Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b"

# Primary models including user's gpt-oss models + free openrouter models
FEATURED_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openrouter/free",
    "deepseek/deepseek-chat:free",
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-2.0-flash-thinking-exp:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "liquid/lfm-2.5-2.6b:free",
    "cohere/north-mini-code:free",
]

# Retrieval Defaults
DEFAULT_TOP_K_CHUNKS = 5
DEFAULT_GRAPH_TOP_K = 5
DEFAULT_GRAPH_HOPS = 2

# Text Chunking Defaults
CHUNK_SIZE = 400
CHUNK_STRIDE = 50
