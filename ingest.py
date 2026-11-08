#!/usr/bin/env python3
"""CLI: ingest Bible / Quran (or other) CSVs into Milvus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.csv_ingest import detect_scripture_kind, load_csv_documents
from src.llamaindex_app import ingest_documents

# Convenience presets: category + typical subcategory label
_PRESETS = {
    "bible": {"category": "Christianity", "subcategory": "KJV"},
    "kjv": {"category": "Christianity", "subcategory": "KJV"},
    "niv": {"category": "Christianity", "subcategory": "NIV"},
    "nkjv": {"category": "Christianity", "subcategory": "NKJV"},
    "geneva": {"category": "Christianity", "subcategory": "Geneva"},
    "tyndale": {"category": "Christianity", "subcategory": "Tyndale"},
    "asv": {"category": "Christianity", "subcategory": "ASV"},
    "quran": {"category": "Islam", "subcategory": "SahihInternational"},
    "islam": {"category": "Islam", "subcategory": "SahihInternational"},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest scripture CSVs (Bible, Quran, …) into one Milvus collection.\n"
            "Use --preset bible|quran or --category / --translation."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file")
    parser.add_argument(
        "--preset",
        choices=sorted(_PRESETS.keys()),
        default=None,
        help="Shortcut: bible/kjv/niv/nkjv → Christianity; quran/islam → Islam",
    )
    parser.add_argument(
        "--text-cols",
        type=str,
        default=None,
        help="Comma-separated columns to embed (default: auto-detect Bible/Quran cols)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Scripture/religion (e.g. Christianity, Islam). Overrides preset/CSV.",
    )
    parser.add_argument(
        "--subcategory",
        "--translation",
        dest="subcategory",
        type=str,
        default=None,
        help="Translation/interpretation (e.g. KJV, NIV, SahihInternational)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append without dropping the collection (use when adding Quran after Bible)",
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

    category = args.category
    subcategory = args.subcategory
    if args.preset:
        preset = _PRESETS[args.preset]
        category = category or preset["category"]
        subcategory = subcategory or preset["subcategory"]

    # Peek at columns for messaging / auto category if still unset
    import pandas as pd

    from src.csv_ingest import _detect_header_skip

    header_cols = list(
        pd.read_csv(
            args.csv_path, nrows=0, skiprows=_detect_header_skip(args.csv_path)
        ).columns
    )
    kind = detect_scripture_kind(header_cols)
    if category is None and kind in {"bible", "quran"}:
        category = "Christianity" if kind == "bible" else "Islam"

    # .env TEXT_COLS is often Bible-specific; fall back to auto-detect for Quran/etc.
    if text_cols:
        missing = [c for c in text_cols if c not in header_cols]
        if missing:
            print(
                f"Note: TEXT_COLS {text_cols} not in this CSV (missing {missing}); "
                "using auto-detect for embed columns."
            )
            text_cols = None

    print(f"Loading {args.csv_path} (detected={kind}) ...")
    documents = load_csv_documents(
        args.csv_path,
        text_cols=text_cols,
        category=category,
        subcategory=subcategory,
    )
    if not documents:
        print("No rows to ingest.", file=sys.stderr)
        return 1

    sample = documents[0].metadata
    print(
        f"Prepared {len(documents)} passage(s)  "
        f"category={sample.get('category')}  "
        f"subcategory={sample.get('subcategory')}  "
        f"kind={sample.get('scripture_kind')}"
    )
    if not args.append:
        print(
            "Recreating collection. Next corpus (e.g. Quran after Bible) must use --append."
        )
    print(
        f"Writing to '{settings.collection_name}' at {settings.milvus_uri}"
        f"{' (remote)' if settings.is_remote_milvus else ' (Milvus Lite)'} ..."
    )

    ingest_documents(
        settings,
        documents,
        overwrite=not args.append,
        show_progress=True,
    )
    print(f"Ingested {len(documents)} row(s).")
    print(
        "Query examples:\n"
        f"  python ask.py --category {sample.get('category')} "
        f"--translation {sample.get('subcategory')} \"your question\"\n"
        '  python ask.py --compare --category Christianity,Islam "shared themes?"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
