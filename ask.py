#!/usr/bin/env python3
"""CLI: ask questions about ingested bible.csv (Milvus retrieve + llama.cpp)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.llamaindex_app import make_query_engine


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
            "Ask retrieves verses from Milvus first; if it is down you get empty results.\n"
            "Fix: start port-forward, e.g.\n"
            "  kubectl -n <ns> port-forward svc/<milvus-proxy> 19530:19530\n"
            "Or set MILVUS_URL to your MetalLB IP in .env"
        )
    finally:
        sock.close()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ask questions about bible.csv verses stored in Milvus "
            "(retrieve relevant verses, then answer with llama.cpp)."
        )
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question about the Bible (omit for interactive mode)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=8,
        help="Number of verses to retrieve (default: 8)",
    )
    parser.add_argument(
        "--show-hits",
        action="store_true",
        help="Print retrieved verses before each answer",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "Analyze retrieved verses (themes, imagery, structure) "
            "instead of a short factual answer"
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    mode = "analyze" if args.analyze else "ask"
    # Analysis benefits from a few more verses when using the default -k
    top_k = 12 if args.analyze and args.top_k == 8 else args.top_k

    timeout_label = (
        "forever" if settings.llama_timeout <= 0 else f"{settings.llama_timeout:g}s"
    )
    print(
        f"Bible {mode}  collection={settings.collection_name}  "
        f"milvus={settings.milvus_uri}  "
        f"chat={settings.llama_base_url}  embed={settings.llama_embed_base_url}  "
        f"timeout={timeout_label}  max_tokens={settings.llama_max_tokens}"
    )
    milvus_err = _check_milvus_reachable(settings.milvus_uri)
    if milvus_err:
        print(milvus_err, file=sys.stderr)
        return 1

    print("Building query engine ...")
    try:
        engine = make_query_engine(
            settings, top_k=top_k, analyze=args.analyze
        )
    except Exception as exc:
        print(
            f"Error connecting to Milvus/LLM: {exc}\n"
            "Ingest first: python ingest.py bible.csv --text-cols Book,Chapter,Verse,Text",
            file=sys.stderr,
        )
        return 1

    if args.question:
        return _ask_one(engine, args.question, show_hits=args.show_hits)

    print(
        "Interactive mode — ask about the Bible (from your ingested bible.csv).\n"
        "Examples:  What does Genesis 1:1 say?\n"
        "           Which verses mention light in creation?\n"
        "With --analyze: Analyze the themes in Genesis 1:1-5\n"
        "Type quit or press Enter on an empty line to exit.\n"
    )
    while True:
        try:
            question = input("Bible> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question or question.lower() in {"quit", "exit", "q"}:
            return 0
        _ask_one(engine, question, show_hits=args.show_hits)
        print()


def _ask_one(engine, question: str, *, show_hits: bool) -> int:
    print("Retrieving verses and asking the model ...")
    try:
        response = engine.query(question)
    except Exception as exc:
        print(f"Query failed: {exc}", file=sys.stderr)
        return 1

    source_nodes = getattr(response, "source_nodes", None) or []
    if show_hits or not source_nodes:
        print("\n--- Retrieved verses ---")
        if not source_nodes:
            print("(none)")
        for node in source_nodes:
            meta = node.metadata or {}
            score = getattr(node, "score", None)
            print(f"{_citation(meta)}  score={score}")
            print(_verse_text(node))
            print("---")

    if not source_nodes:
        print(
            "\nNo matching verses found. Try rephrasing, or confirm ingest completed.",
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
    book = meta.get("Book")
    chapter = meta.get("Chapter")
    verse = meta.get("Verse")
    if book is not None and chapter is not None and verse is not None:
        return f"{book} {chapter}:{verse}"
    if "source_row" in meta:
        return f"source_row={meta['source_row']}"
    return "verse"


def _verse_text(node) -> str:
    meta = node.metadata or {}
    text = meta.get("Text")
    if text:
        return str(text)
    content = node.get_content()
    # Document text often looks like "Book: ...\nText: ..."
    for line in content.splitlines():
        if line.startswith("Text:"):
            return line[len("Text:") :].strip()
    return content


if __name__ == "__main__":
    raise SystemExit(main())
