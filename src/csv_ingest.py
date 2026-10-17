"""Load a CSV into LlamaIndex Documents (one row per document)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from llama_index.core import Document


def load_csv_documents(
    path: str | Path,
    text_cols: list[str] | None = None,
) -> list[Document]:
    """Return one Document per CSV row with text + metadata."""
    df = pd.read_csv(path)
    if df.empty:
        return []

    cols = list(df.columns)
    if text_cols:
        missing = [c for c in text_cols if c not in cols]
        if missing:
            raise ValueError(
                f"TEXT_COLS not in CSV: {missing}. Available columns: {cols}"
            )
        embed_cols = text_cols
    else:
        embed_cols = _default_text_columns(df)

    if not embed_cols:
        raise ValueError(
            "No text columns found to embed. Pass --text-cols or set TEXT_COLS."
        )

    documents: list[Document] = []
    for row_idx, row in df.iterrows():
        parts: list[str] = []
        for col in embed_cols:
            val = row[col]
            if pd.isna(val):
                continue
            parts.append(f"{col}: {val}")
        text = "\n".join(parts).strip()
        if not text:
            continue

        metadata: dict[str, Any] = {
            "source_row": int(row_idx) if isinstance(row_idx, (int, float)) else row_idx,
            "source_file": str(path),
        }
        for col in cols:
            val = row[col]
            if pd.isna(val):
                continue
            if hasattr(val, "item"):
                val = val.item()
            # Milvus / LlamaIndex metadata values should be simple JSON types
            if isinstance(val, (str, int, float, bool)):
                metadata[col] = val
            else:
                metadata[col] = str(val)

        # Stable id for Bible refs when columns exist
        doc_id = _doc_id(metadata, row_idx)
        documents.append(
            Document(
                text=text,
                metadata=metadata,
                id_=doc_id,
            )
        )

    return documents


def _doc_id(metadata: dict[str, Any], row_idx: Any) -> str:
    book = metadata.get("Book")
    chapter = metadata.get("Chapter")
    verse = metadata.get("Verse")
    if book is not None and chapter is not None and verse is not None:
        return f"{book}:{chapter}:{verse}"
    return f"row-{row_idx}"


def _default_text_columns(df: pd.DataFrame) -> list[str]:
    text_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            text_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # Include numeric cols like Chapter/Verse when useful for refs
            if col.lower() in {"chapter", "verse", "chap", "v"}:
                text_cols.append(col)
    return text_cols
