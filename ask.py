#!/usr/bin/env python3
"""CLI: ask / analyze / compare / debate scripture from Milvus + llama.cpp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.llamaindex_app import debate_query, make_query_engine


def _check_milvus_reachable(uri: str) -> str | None:
    """Return an error message if Milvus TCP port is not open."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(uri if "://" in uri else f"http://{uri}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 19530
    sock = socket.socket()
    sock.settimeout(3)
    try:
        sock.connect((host, port))
    except OSError as exc:
        return (
            f"Cannot reach Milvus at {host}:{port} ({exc}).\n"
            "Ask retrieves passages from Milvus first; if it is down you get empty results.\n"
            "Fix: start port-forward, e.g.\n"
            "  kubectl -n <ns> port-forward svc/<milvus-proxy> 19530:19530\n"
            "Or set MILVUS_URL to your MetalLB IP in .env"
        )
    finally:
        sock.close()
    return None


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ask questions about ingested scripture CSVs in Milvus. "
            "Filter by --category / --translation. "
            "Use --compare for neutral contrast, or --debate for Christian "
            "apologetics vs Quran (cites both)."
        )
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question (omit for interactive mode)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=8,
        help="Passages to retrieve (debate: this many from EACH side; default 8)",
    )
    parser.add_argument(
        "--show-hits",
        action="store_true",
        help="Print retrieved passages before each answer",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Structured analysis (themes, imagery, …)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Neutral compare across categories/subcategories",
    )
    parser.add_argument(
        "--debate",
        action="store_true",
        help=(
            "Christian apologetics debate: retrieve Bible + Quran, argue for "
            "Christianity, critique Quran passages, prove with citations from both"
        ),
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to one category (e.g. Christianity). Comma-list OK.",
    )
    parser.add_argument(
        "--subcategory",
        "--translation",
        dest="subcategory",
        type=str,
        default=None,
        help="Filter to translation(s), e.g. KJV or KJV,NIV",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    categories = _split_csv(args.category)
    subcategories = _split_csv(args.subcategory)
    debate = args.debate
    compare = (args.compare or len(subcategories) > 1 or len(categories) > 1) and not debate
    analyze = args.analyze and not compare and not debate

    if debate:
        mode = "debate-christian"
        top_k = args.top_k
        if not categories:
            categories = ["Christianity", "Islam"]
    elif compare:
        mode = "compare"
        top_k = 16 if args.top_k == 8 else args.top_k
    elif analyze:
        mode = "analyze"
        top_k = 12 if args.top_k == 8 else args.top_k
    else:
        mode = "ask"
        top_k = args.top_k

    timeout_label = (
        "forever" if settings.llama_timeout <= 0 else f"{settings.llama_timeout:g}s"
    )
    filter_bits = []
    if categories:
        filter_bits.append(f"category={','.join(categories)}")
    if subcategories:
        filter_bits.append(f"subcategory={','.join(subcategories)}")
    filter_label = " ".join(filter_bits) if filter_bits else "filters=none"

    print(
        f"Scripture {mode}  collection={settings.collection_name}  "
        f"milvus={settings.milvus_uri}  {filter_label}  "
        f"chat={settings.llama_base_url}  embed={settings.llama_embed_base_url}  "
        f"timeout={timeout_label}  max_tokens={settings.llama_max_tokens}"
    )
    milvus_err = _check_milvus_reachable(settings.milvus_uri)
    if milvus_err:
        print(milvus_err, file=sys.stderr)
        return 1

    engine = None
    if not debate:
        print("Building query engine ...")
        try:
            engine = make_query_engine(
                settings,
                top_k=top_k,
                analyze=analyze,
                compare=compare,
                debate=False,
                categories=categories or None,
                subcategories=subcategories or None,
            )
        except Exception as exc:
            print(
                f"Error connecting to Milvus/LLM: {exc}\n"
                "Ingest Bible + Quran into the same collection first.",
                file=sys.stderr,
            )
            return 1
    else:
        print("Debate mode: retrieving from Christianity and Islam separately ...")

    if args.question:
        if debate:
            return _debate_one(
                settings,
                args.question,
                top_k_each=top_k,
                show_hits=args.show_hits,
                subcategories=subcategories or None,
            )
        return _ask_one(engine, args.question, show_hits=args.show_hits)

    print(
        "Interactive mode.\n"
        "  ask:     python ask.py --category Christianity \"...\"\n"
        "  compare: python ask.py --compare --category Christianity,Islam \"...\"\n"
        "  debate:  python ask.py --debate \"Is Jesus the Son of God?\"\n"
        "Type quit or empty line to exit.\n"
    )
    while True:
        try:
            question = input("Scripture> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question or question.lower() in {"quit", "exit", "q"}:
            return 0
        if debate:
            _debate_one(
                settings,
                question,
                top_k_each=top_k,
                show_hits=args.show_hits,
                subcategories=subcategories or None,
            )
        else:
            _ask_one(engine, question, show_hits=args.show_hits)
        print()


def _debate_one(
    settings,
    question: str,
    *,
    top_k_each: int,
    show_hits: bool,
    subcategories: list[str] | None,
) -> int:
    print("Retrieving Bible + Quran passages and debating ...")
    try:
        result = debate_query(
            settings,
            question,
            top_k_each=top_k_each,
            subcategories=subcategories,
        )
    except Exception as exc:
        print(f"Debate query failed: {exc}", file=sys.stderr)
        return 1

    source_nodes = result.source_nodes or []
    if show_hits or not source_nodes:
        print("\n--- Retrieved Bible ---")
        for node in result.bible_nodes or []:
            print(f"{_citation(node.metadata or {})}  score={node.score}")
            print(_passage_text(node))
            print("---")
        print("\n--- Retrieved Quran ---")
        for node in result.quran_nodes or []:
            print(f"{_citation(node.metadata or {})}  score={node.score}")
            print(_passage_text(node))
            print("---")

    if not result.bible_nodes:
        print(
            "\nNo Bible passages retrieved. Ingest a Bible CSV with "
            "--preset bible/kjv first.",
            file=sys.stderr,
        )
        return 1
    if not result.quran_nodes:
        print(
            "\nNo Quran passages retrieved. Ingest quran_dataset.csv with "
            "--preset quran --append first.",
            file=sys.stderr,
        )
        return 1

    print("\n=== Debate (Christian apologetics) ===")
    print(result.response_text.strip())
    print("\n=== Citations ===")
    for node in source_nodes:
        meta = node.metadata or {}
        print(f"- {_citation(meta)} (score={node.score})")
    return 0


def _ask_one(engine, question: str, *, show_hits: bool) -> int:
    print("Retrieving passages and asking the model ...")
    try:
        response = engine.query(question)
    except Exception as exc:
        print(f"Query failed: {exc}", file=sys.stderr)
        return 1

    source_nodes = getattr(response, "source_nodes", None) or []
    if show_hits or not source_nodes:
        print("\n--- Retrieved passages ---")
        if not source_nodes:
            print("(none)")
        for node in source_nodes:
            meta = node.metadata or {}
            score = getattr(node, "score", None)
            print(f"{_citation(meta)}  score={score}")
            print(_passage_text(node))
            print("---")

    if not source_nodes:
        print(
            "\nNo matching passages. Check filters, or re-ingest with "
            "--category / --translation so metadata exists.",
            file=sys.stderr,
        )
        return 1

    print("\n=== Answer ===")
    print(str(response).strip())
    print("\n=== Citations ===")
    for node in source_nodes:
        meta = node.metadata or {}
        score = getattr(node, "score", None)
        print(f"- {_citation(meta)} (score={score})")
    return 0


def _citation(meta: dict) -> str:
    cat = meta.get("category")
    sub = meta.get("subcategory")
    prefix = ""
    if cat or sub:
        prefix = f"{cat or '?'}/{sub or '?'}: "

    surah_name = meta.get("SurahName") or meta.get("Book")
    surah = meta.get("Surah") if meta.get("Surah") is not None else meta.get("Chapter")
    ayah = meta.get("Ayah") if meta.get("Ayah") is not None else meta.get("Verse")
    kind = (meta.get("scripture_kind") or "").lower()
    if kind == "quran" and surah is not None and ayah is not None:
        label = surah_name or f"Surah {surah}"
        return f"{prefix}{label} {surah}:{ayah}"

    book = meta.get("Book")
    chapter = meta.get("Chapter")
    verse = meta.get("Verse")
    if book is not None and chapter is not None and verse is not None:
        return f"{prefix}{book} {chapter}:{verse}"
    if surah is not None and ayah is not None:
        label = surah_name or f"Surah {surah}"
        return f"{prefix}{label} {surah}:{ayah}"
    if "source_row" in meta:
        return f"{prefix}source_row={meta['source_row']}"
    return f"{prefix}passage"


def _passage_text(node) -> str:
    meta = node.metadata or {}
    for key in ("Text", "ayah_en", "verse_text", "ayah_text", "content"):
        if meta.get(key):
            return str(meta[key])
    content = node.get_content()
    for line in content.splitlines():
        if line.startswith("Text:") or line.startswith("ayah_en:"):
            return line.split(":", 1)[1].strip()
    return content


if __name__ == "__main__":
    raise SystemExit(main())
