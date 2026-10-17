#!/usr/bin/env python3
"""CLI: ingest a CSV into Milvus via LlamaIndex + llama.cpp embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.csv_ingest import load_csv_documents
from src.llamaindex_app import ingest_documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest a CSV into Milvus (LlamaIndex + llama.cpp)."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file")
    parser.add_argument(
        "--text-cols",
        type=str,
        default=None,
        help="Comma-separated columns to embed (default: TEXT_COLS env or auto)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing collection instead of recreating it",
    )
    args = parser.parse_args(argv)

    if not args.csv_path.is_file():
        print(f"Error: CSV not found: {args.csv_path}", file=sys.stderr)
        return 1

    settings = get_settings()
    text_cols = (
        [c.strip() for c in args.text_cols.split(",") if c.strip()]
        if args.text_cols
        else settings.text_cols
    )

    print(f"Loading {args.csv_path} ...")
    documents = load_csv_documents(args.csv_path, text_cols=text_cols)
    if not documents:
        print("No rows to ingest.", file=sys.stderr)
        return 1
    print(f"Prepared {len(documents)} document(s) for embedding.")
    print(
        f"Writing to collection '{settings.collection_name}' at {settings.milvus_uri}"
        f"{' (remote)' if settings.is_remote_milvus else ' (Milvus Lite)'} ..."
    )

    ingest_documents(
        settings,
        documents,
        overwrite=not args.append,
        show_progress=True,
    )
    print(f"Ingested {len(documents)} row(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
