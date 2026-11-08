"""LlamaIndex wiring: llama.cpp (OpenAI-like) + Milvus."""

from __future__ import annotations

import httpx
from llama_index.core import Document, PromptTemplate, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import DataType

from src.config import Settings as AppSettings

SCRIPTURE_QA_PROMPT = PromptTemplate(
    "You are a scripture study assistant. Use ONLY the retrieved passages below as "
    "your source. You may briefly explain meaning when asked, but every claim must "
    "be grounded in those passages. Cite with category/subcategory and reference "
    "(e.g. Christianity/KJV Genesis 1:1). Do not invent passages. If insufficient, "
    "say so.\n\n"
    "Retrieved passages:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)

SCRIPTURE_ANALYZE_PROMPT = PromptTemplate(
    "You are a scripture study assistant. The user wants ANALYSIS, not a dump of "
    "text. Using ONLY the retrieved passages below, write a structured study.\n\n"
    "Required sections (use these headings):\n"
    "1. Themes — main ideas in the retrieved passages\n"
    "2. Imagery & language — metaphors, symbols, repeated words/phrases\n"
    "3. Structure & tone — how the passage moves; mood\n"
    "4. What the text emphasizes — central message from these passages\n"
    "5. Brief takeaway — 2–4 sentences for today, still tied to the text\n\n"
    "Rules:\n"
    "- Do NOT only list or paste scripture. Explain under the headings above.\n"
    "- Cite as Category/Subcategory Ref (e.g. Christianity/KJV John 3:16).\n"
    "- Quote at most one short phrase per point when needed.\n"
    "- Do not invent passages outside the retrieved set.\n"
    "- If multiple subcategories (translations) appear, note differences briefly.\n\n"
    "Retrieved passages:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Analysis request: {query_str}\n\n"
    "Write the structured analysis now:\n"
)

SCRIPTURE_COMPARE_PROMPT = PromptTemplate(
    "You are a scripture comparison assistant. The retrieved passages may come from "
    "different categories (religions/scriptures) and/or subcategories "
    "(translations/interpretations such as KJV, NIV, NKJV).\n\n"
    "Compare them carefully using ONLY the retrieved text:\n"
    "1. Shared themes\n"
    "2. Wording / imagery differences by subcategory\n"
    "3. Emphasis differences\n"
    "4. Where they agree or diverge on the question\n"
    "5. Short balanced takeaway\n\n"
    "Always label citations as Category/Subcategory Ref. Do not invent text.\n\n"
    "Retrieved passages:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Comparison question: {query_str}\n\n"
    "Write the comparison now:\n"
)

SCRIPTURE_DEBATE_CHRISTIAN_PROMPT = PromptTemplate(
    "You are a Christian apologetics debater. Argue in favor of the Christian "
    "position on the user's topic. Use ONLY the retrieved Bible and Quran "
    "passages below — do not invent verses or ayahs.\n\n"
    "Goals:\n"
    "- Present the Christian claim clearly from the Bible passages\n"
    "- Identify tensions, weaknesses, or contradictions in the Quran passages "
    "relative to that claim (only where the retrieved Quran text actually supports "
    "the critique)\n"
    "- Prove / support your Christian case by citing Bible passages\n"
    "- When criticizing the Quran side, quote or cite the specific retrieved "
    "Quran passages you are answering\n"
    "- Judge which portrayal is more moral, more just, and more consistent with "
    "the character of a holy, loving, righteous God — and explain why, from the "
    "retrieved texts\n"
    "- Note where Quran material parallels earlier Bible content that Muhammad "
    "(and his milieu) could have heard or known — stories, laws, titles, or "
    "themes that appear first in the retrieved Bible passages\n"
    "- Stay civil but firm; this is debate, not insult\n\n"
    "Required sections:\n"
    "1. Christian position (Bible)\n"
    "2. What the retrieved Quran passages say on this topic\n"
    "3. Faults / tensions in the Quran passages (vs Christian claim)\n"
    "4. Proof from scripture (Bible citations answering the issue)\n"
    "5. Morality & justice — which side's teaching (from the retrieved texts) is "
    "more moral and more just? Give concrete reasons tied to the citations\n"
    "6. More like God — which portrayal better fits a God who is holy, just, "
    "merciful, and truthful? Explain why the Christian reading is probably closer "
    "to God's character, using the retrieved Bible (and contrasting Quran) text\n"
    "7. Possible biblical sources Muhammad may have known — where the retrieved "
    "Quran passages echo older Bible material (people, events, commands, themes). "
    "Cite the Bible passage that looks earlier/source-like and the Quran passage "
    "that parallels it. Mark clearly what is textual parallel vs. historical "
    "inference; do not invent books or verses not in the retrieved set. If "
    "parallels are weak, say so.\n"
    "8. Closing: why the Christian reading is stronger overall on this question\n\n"
    "Citation format: Christianity/KJV John 3:16 or Islam/... Al-Fatihah 1:5.\n"
    "If retrieved text is too thin to support a critique, say so instead of inventing.\n\n"
    "Retrieved passages (Bible + Quran):\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Debate topic: {query_str}\n\n"
    "Write the Christian apologetics debate response now:\n"
)


def configure_settings(app: AppSettings) -> None:
    """Set global LlamaIndex LLM + embedding models (llama.cpp server)."""
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
            "You help with scripture questions, analysis, comparison, and "
            "Christian apologetics debate using only provided passage context. "
            "Cite Category/Subcategory and reference. Do not invent scripture."
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
        # Filterable taxonomy fields
        "scalar_field_names": ["category", "subcategory"],
        "scalar_field_types": [DataType.VARCHAR, DataType.VARCHAR],
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


def build_metadata_filters(
    *,
    category: str | None = None,
    subcategory: str | None = None,
    subcategories: list[str] | None = None,
    categories: list[str] | None = None,
) -> MetadataFilters | None:
    """Build Milvus metadata filters for category / subcategory."""
    filters: list[MetadataFilter] = []

    cats = [c for c in (categories or []) if c]
    if category:
        cats.append(category)
    cats = [_norm(c) for c in cats]
    if len(cats) == 1:
        filters.append(MetadataFilter(key="category", value=cats[0]))
    elif len(cats) > 1:
        filters.append(
            MetadataFilter(
                key="category", value=cats, operator=FilterOperator.IN
            )
        )

    subs = [s for s in (subcategories or []) if s]
    if subcategory:
        subs.append(subcategory)
    subs = [_norm(s) for s in subs]
    if len(subs) == 1:
        filters.append(MetadataFilter(key="subcategory", value=subs[0]))
    elif len(subs) > 1:
        filters.append(
            MetadataFilter(
                key="subcategory", value=subs, operator=FilterOperator.IN
            )
        )

    if not filters:
        return None
    return MetadataFilters(filters=filters, condition=FilterCondition.AND)


def make_query_engine(
    app: AppSettings,
    *,
    top_k: int = 5,
    analyze: bool = False,
    compare: bool = False,
    debate: bool = False,
    category: str | None = None,
    subcategory: str | None = None,
    categories: list[str] | None = None,
    subcategories: list[str] | None = None,
):
    configure_settings(app)
    vector_store = build_vector_store(app, overwrite=False)
    index = VectorStoreIndex.from_vector_store(vector_store)

    if debate:
        # Dual retrieval is handled by debate_query(); engine is a fallback
        template = SCRIPTURE_DEBATE_CHRISTIAN_PROMPT
        categories = categories or ["Christianity", "Islam"]
    elif compare:
        template = SCRIPTURE_COMPARE_PROMPT
    elif analyze:
        template = SCRIPTURE_ANALYZE_PROMPT
    else:
        template = SCRIPTURE_QA_PROMPT

    filters = build_metadata_filters(
        category=category,
        subcategory=subcategory,
        categories=categories,
        subcategories=subcategories,
    )
    kwargs: dict = {
        "similarity_top_k": top_k,
        "text_qa_template": template,
    }
    if filters is not None:
        kwargs["filters"] = filters
    return index.as_query_engine(**kwargs)


def debate_query(
    app: AppSettings,
    question: str,
    *,
    top_k_each: int = 8,
    subcategories: list[str] | None = None,
):
    """Retrieve balanced Bible + Quran hits, then run Christian debate synthesis.

    Returns a simple namespace with .response_text and .source_nodes.
    """
    from types import SimpleNamespace

    from llama_index.core.schema import MetadataMode

    configure_settings(app)
    vector_store = build_vector_store(app, overwrite=False)
    index = VectorStoreIndex.from_vector_store(vector_store)

    per = max(2, top_k_each)
    bible_filters = build_metadata_filters(
        category="Christianity", subcategories=subcategories
    )
    quran_filters = build_metadata_filters(
        category="Islam", subcategories=subcategories
    )

    bible_nodes = index.as_retriever(
        similarity_top_k=per, filters=bible_filters
    ).retrieve(question)
    quran_nodes = index.as_retriever(
        similarity_top_k=per, filters=quran_filters
    ).retrieve(question)
    source_nodes = list(bible_nodes) + list(quran_nodes)

    def _fmt(nodes, label: str) -> str:
        blocks = []
        for n in nodes:
            meta = n.metadata or {}
            ref = _format_ref(meta)
            text = n.get_content(metadata_mode=MetadataMode.NONE)
            blocks.append(f"[{label} | {ref} | score={n.score}]\n{text}")
        return "\n\n".join(blocks) if blocks else f"({label}: no passages retrieved)"

    context = (
        "=== BIBLE (Christianity) ===\n"
        + _fmt(bible_nodes, "Bible")
        + "\n\n=== QURAN (Islam) ===\n"
        + _fmt(quran_nodes, "Quran")
    )
    prompt = SCRIPTURE_DEBATE_CHRISTIAN_PROMPT.format(
        context_str=context, query_str=question
    )
    response_text = str(Settings.llm.complete(prompt))
    return SimpleNamespace(
        response_text=response_text,
        source_nodes=source_nodes,
        bible_nodes=bible_nodes,
        quran_nodes=quran_nodes,
    )


def _format_ref(meta: dict) -> str:
    cat = meta.get("category", "?")
    sub = meta.get("subcategory", "?")
    book = meta.get("Book") or meta.get("SurahName")
    chapter = meta.get("Chapter") if meta.get("Chapter") is not None else meta.get("Surah")
    verse = meta.get("Verse") if meta.get("Verse") is not None else meta.get("Ayah")
    if book is not None and chapter is not None and verse is not None:
        return f"{cat}/{sub}: {book} {chapter}:{verse}"
    return f"{cat}/{sub}"


def _norm(value: str) -> str:
    cleaned = value.strip()
    lower = cleaned.lower()
    if lower == "islam":
        return "Islam"
    if lower == "christianity":
        return "Christianity"
    if lower == "judaism":
        return "Judaism"
    if lower in {"kjv", "niv", "nkjv", "esv", "nasb", "nlt", "asv"}:
        return cleaned.upper()
    if lower in {"geneva", "tyndale"}:
        return cleaned[:1].upper() + cleaned[1:].lower()
    return cleaned

