"""LlamaIndex wiring: llama.cpp (OpenAI-like) + Milvus."""

from __future__ import annotations

import httpx
from llama_index.core import Document, PromptTemplate, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore

from src.config import Settings as AppSettings

BIBLE_QA_PROMPT = PromptTemplate(
    "You are a Bible study assistant. Use ONLY the retrieved verses below as your "
    "scripture source. You may briefly explain meaning when asked, but every claim "
    "must be grounded in those verses. Cite Book Chapter:Verse (e.g. Genesis 1:1). "
    "Do not invent or quote verses that are not in the retrieved set. If the verses "
    "are insufficient, say so.\n\n"
    "Retrieved Bible verses:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)

BIBLE_ANALYZE_PROMPT = PromptTemplate(
    "You are a Bible study assistant. The user wants ANALYSIS, not a dump of "
    "scripture. Using ONLY the retrieved verses below, write a structured study.\n\n"
    "Required sections (use these headings):\n"
    "1. Themes — main ideas in the retrieved verses\n"
    "2. Imagery & language — metaphors, symbols, repeated words/phrases\n"
    "3. Structure & tone — how the passage moves; mood\n"
    "4. What the text emphasizes — central message from these verses\n"
    "5. Brief takeaway — 2–4 sentences for today, still tied to the verses\n\n"
    "Rules:\n"
    "- Do NOT only list or paste verses. Explain them under the headings above.\n"
    "- Support each point with short citations like (John 3:16), not long quotes.\n"
    "- Quote at most one short phrase per point when needed.\n"
    "- Do not invent verses outside the retrieved set.\n\n"
    "Retrieved Bible verses:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Analysis request: {query_str}\n\n"
    "Write the structured analysis now:\n"
)


def configure_settings(app: AppSettings) -> None:
    """Set global LlamaIndex LLM + embedding models (llama.cpp server)."""
    # Connect fails fast; read waits as long as LLAMA_TIMEOUT (or forever if <= 0)
    read_timeout = None if app.llama_timeout <= 0 else app.llama_timeout
    http_timeout = httpx.Timeout(
        connect=60.0,
        read=read_timeout,
        write=read_timeout,
        pool=60.0,
    )
    llm_http = httpx.Client(timeout=http_timeout)
    embed_http = httpx.Client(timeout=http_timeout)

    Settings.llm = OpenAILike(
        model=app.llama_model,
        api_base=app.llama_base_url,
        api_key=app.llama_api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=0.2,
        timeout=app.llama_timeout if app.llama_timeout > 0 else 86400 * 7,
        max_tokens=app.llama_max_tokens,
        max_retries=0,
        context_window=8192,
        reuse_client=True,
        http_client=llm_http,
        system_prompt=(
            "You help with Bible questions and textual analysis using only provided "
            "verse context. When asked to analyze, use clear headings (Themes, "
            "Imagery, Structure, Emphasis, Takeaway) and explain — do not only "
            "reprint scripture. Cite Book Chapter:Verse. Do not invent scripture."
        ),
    )
    Settings.embed_model = OpenAILikeEmbedding(
        model_name=app.llama_embed_model,
        api_base=app.llama_embed_base_url,
        api_key=app.llama_api_key,
        timeout=app.llama_timeout if app.llama_timeout > 0 else 86400 * 7,
        max_retries=0,
        reuse_client=True,
        http_client=embed_http,
    )
    # Keep each CSV row (verse) as a single node unless a cell is huge
    Settings.transformations = [
        SentenceSplitter(chunk_size=4096, chunk_overlap=0),
    ]


def build_vector_store(
    app: AppSettings,
    *,
    overwrite: bool = False,
) -> MilvusVectorStore:
    kwargs: dict = {
        "uri": app.milvus_uri,
        "collection_name": app.collection_name,
        "dim": app.embed_dim,
        "overwrite": overwrite,
    }
    if app.milvus_token:
        kwargs["token"] = app.milvus_token
    if app.milvus_db_name and app.milvus_db_name != "default":
        kwargs["db_name"] = app.milvus_db_name
    return MilvusVectorStore(**kwargs)


def ingest_documents(
    app: AppSettings,
    documents: list[Document],
    *,
    overwrite: bool = True,
    show_progress: bool = True,
) -> VectorStoreIndex:
    configure_settings(app)
    vector_store = build_vector_store(app, overwrite=overwrite)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=show_progress,
    )


def make_query_engine(
    app: AppSettings,
    *,
    top_k: int = 5,
    analyze: bool = False,
):
    configure_settings(app)
    vector_store = build_vector_store(app, overwrite=False)
    index = VectorStoreIndex.from_vector_store(vector_store)
    template = BIBLE_ANALYZE_PROMPT if analyze else BIBLE_QA_PROMPT
    return index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=template,
    )
