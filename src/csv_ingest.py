"""Load a CSV into LlamaIndex Documents (one row per document)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from llama_index.core import Document

# CSV column names that may carry taxonomy (case-insensitive match)
_CATEGORY_ALIASES = ("category", "religion", "tradition", "scripture")
_SUBCATEGORY_ALIASES = (
    "subcategory",
    "sub_category",
    "translation",
    "version",
    "interpretation",
)

# Reference field aliases (Bible vs Quran-style CSVs)
_BOOK_ALIASES = (
    "book",
    "book name",
    "book_name",
    "surahname",
    "surah_name",
    "sura_name",
    "surah_name_roman",
    "surah_name_en",
    "name",
)
_CHAPTER_ALIASES = (
    "chapter",
    "surah",
    "sura",
    "surah_number",
    "sura_number",
    "surah_no",
)
_VERSE_ALIASES = (
    "verse",
    "ayah",
    "aya",
    "ayat",
    "verse_number",
    "ayah_no_surah",
    "ayah_no",
)
_TEXT_ALIASES = (
    "text",
    "verse_text",
    "content",
    "translation_text",
    "ayah_text",
    "ayah_en",
    "ayah_ar",
)


def load_csv_documents(
    path: str | Path,
    text_cols: list[str] | None = None,
    *,
    category: str | None = None,
    subcategory: str | None = None,
) -> list[Document]:
    """Return one Document per CSV row with text + metadata.

    ``category`` = scripture/religion (e.g. Christianity, Islam).
    ``subcategory`` = translation/interpretation (e.g. KJV, SahihInternational).

    Supports Bible columns (Book/Chapter/Verse/Text) and Quran columns
    (Surah/Ayah/SurahName/Text). Values from CLI override CSV taxonomy columns.
    """
    df = pd.read_csv(path, skiprows=_detect_header_skip(path))
    if df.empty:
        return []

    # Normalize header names (e.g. "Book Name" → usable aliases)
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    col_map = {c.lower(): c for c in cols}
    kind = detect_scripture_kind(cols)

    if text_cols:
        missing = [c for c in text_cols if c not in cols]
        if missing:
            raise ValueError(
                f"TEXT_COLS not in CSV: {missing}. Available columns: {cols}"
            )
        embed_cols = text_cols
    else:
        embed_cols = default_text_columns_for(df, kind)

    if not embed_cols:
        raise ValueError(
            "No text columns found to embed. Pass --text-cols or set TEXT_COLS."
        )

    cat_col = _find_col(col_map, _CATEGORY_ALIASES)
    sub_col = _find_col(col_map, _SUBCATEGORY_ALIASES)

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

        row_category = category or _cell(row, cat_col) or _default_category(kind)
        row_subcategory = subcategory or _cell(row, sub_col) or "unspecified"
        row_category = _normalize_label(row_category)
        row_subcategory = _normalize_label(row_subcategory)

        metadata: dict[str, Any] = {
            "source_row": int(row_idx) if isinstance(row_idx, (int, float)) else row_idx,
            "source_file": str(path),
            "scripture_kind": kind,
        }
        for col in cols:
            if col.lower() in {"category", "subcategory"}:
                continue
            val = row[col]
            if pd.isna(val):
                continue
            if hasattr(val, "item"):
                val = val.item()
            if isinstance(val, (str, int, float, bool)):
                metadata[col] = val
            else:
                metadata[col] = str(val)

        # Normalized reference fields for citations (Bible + Quran)
        _attach_normalized_refs(metadata, col_map, row)

        metadata["category"] = row_category
        metadata["subcategory"] = row_subcategory

        doc_id = _doc_id(metadata, row_idx)
        documents.append(
            Document(
                text=text,
                metadata=metadata,
                id_=doc_id,
            )
        )

    return documents


def _detect_header_skip(path: str | Path) -> int:
    """Skip preamble lines (title/license) until a real CSV header appears."""
    path = Path(path)
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            lower = line.lower()
            has_text = "text" in lower
            has_book = "book" in lower or "surah" in lower
            has_verse = "verse" in lower or "ayah" in lower
            if has_text and has_book and has_verse:
                return i
            # Stop scanning after a reasonable preamble
            if i >= 40:
                break
    return 0


def detect_scripture_kind(columns: list[str]) -> str:
    """Return 'quran', 'bible', or 'generic' from column names."""
    lower = {c.lower() for c in columns}
    quran_hits = {
        "surah",
        "sura",
        "ayah",
        "aya",
        "ayat",
        "surahname",
        "surah_name",
        "surah_no",
        "ayah_no_surah",
        "ayah_en",
        "ayah_ar",
        "surah_name_roman",
        "surah_name_en",
    }
    bible_hits = {"book", "book name", "book_name", "chapter", "verse"}
    # Headers may arrive as 'Book Name' → lower 'book name'
    if lower & quran_hits or any("surah" in c or "ayah" in c for c in lower):
        return "quran"
    if lower & bible_hits or any("book" in c for c in lower):
        if any("verse" in c or "chapter" in c for c in lower):
            return "bible"
    return "generic"


def default_text_columns_for(df: pd.DataFrame, kind: str) -> list[str]:
    col_map = {c.lower(): c for c in df.columns}
    if kind == "quran":
        preferred = []
        for aliases in (
            ("surah_name_roman", "surahname", "surah_name", "sura_name", "surah_name_en"),
            ("surah_no", "surah", "sura", "chapter"),
            ("ayah_no_surah", "ayah", "aya", "ayat", "verse"),
            # Prefer English translation for embedding; Arabic kept in metadata
            ("ayah_en", "text", "verse_text", "content", "ayah_text", "ayah_ar"),
        ):
            found = _find_col(col_map, aliases)
            if found:
                preferred.append(found)
        if preferred:
            return preferred
    if kind == "bible":
        preferred = []
        for aliases in (
            ("book name", "book_name", "book"),
            ("chapter",),
            ("verse",),
            ("text", "verse_text", "content"),
        ):
            found = _find_col(col_map, aliases)
            if found:
                preferred.append(found)
        if preferred:
            return preferred
    return _default_text_columns(df)


def _default_category(kind: str) -> str:
    if kind == "quran":
        return "Islam"
    if kind == "bible":
        return "Christianity"
    return "unspecified"


def _attach_normalized_refs(
    metadata: dict[str, Any],
    col_map: dict[str, str],
    row: pd.Series,
) -> None:
    """Set Book/Chapter/Verse (and Surah/Ayah) consistently for citations."""
    book = metadata.get("Book") or _cell(row, _find_col(col_map, _BOOK_ALIASES))
    chapter = metadata.get("Chapter") or _cell(
        row, _find_col(col_map, _CHAPTER_ALIASES)
    )
    verse = metadata.get("Verse") or _cell(row, _find_col(col_map, _VERSE_ALIASES))
    text = metadata.get("Text") or _cell(row, _find_col(col_map, _TEXT_ALIASES))

    # Quran: prefer Surah name as Book label when present
    surah_name = _cell(
        row,
        _find_col(
            col_map,
            ("surah_name_roman", "surahname", "surah_name", "sura_name", "surah_name_en"),
        ),
    )
    surah_num = _cell(row, _find_col(col_map, _CHAPTER_ALIASES))
    ayah = _cell(row, _find_col(col_map, _VERSE_ALIASES))

    if surah_num is not None:
        metadata["Surah"] = _maybe_num(surah_num)
    if ayah is not None:
        metadata["Ayah"] = _maybe_num(ayah)
    if surah_name:
        metadata["SurahName"] = surah_name
        book = book or surah_name
    if chapter is None and surah_num is not None:
        chapter = surah_num
    if verse is None and ayah is not None:
        verse = ayah

    if book is not None:
        metadata["Book"] = book
    if chapter is not None:
        metadata["Chapter"] = _maybe_num(chapter)
    if verse is not None:
        metadata["Verse"] = _maybe_num(verse)
    if text is not None:
        metadata["Text"] = text


def _maybe_num(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else value
    try:
        f = float(str(value))
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return value


def _find_col(col_map: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in col_map:
            return col_map[alias]
    return None


def _cell(row: pd.Series, col: str | None) -> str | None:
    if not col or col not in row.index or pd.isna(row[col]):
        return None
    return str(row[col]).strip() or None


_VERSION_CODES = {
    "kjv",
    "niv",
    "nkjv",
    "esv",
    "nasb",
    "nlt",
    "rsv",
    "asv",
    "geneva",
    "tyndale",
}
_TRADITIONS = {
    "islam": "Islam",
    "christianity": "Christianity",
    "judaism": "Judaism",
    "bible": "Christianity",
    "quran": "Islam",
    "unspecified": "unspecified",
}


def _normalize_label(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return "unspecified"
    lower = cleaned.lower()
    if lower in _TRADITIONS:
        return _TRADITIONS[lower]
    if lower in _VERSION_CODES:
        return cleaned.upper()
    compact = cleaned.replace(" ", "").replace("_", "")
    if compact.lower() in {"sahihinternational", "sahih", "pickthall", "yusufali"}:
        return "SahihInternational" if "sahih" in compact.lower() else compact
    # Short all-caps-ish codes (e.g. KJV) — not tradition names
    if len(cleaned) <= 5 and cleaned.replace("-", "").isalpha() and cleaned.isupper():
        return cleaned
    return cleaned


def _doc_id(metadata: dict[str, Any], row_idx: Any) -> str:
    cat = metadata.get("category", "unspecified")
    sub = metadata.get("subcategory", "unspecified")
    book = metadata.get("Book")
    chapter = metadata.get("Chapter")
    verse = metadata.get("Verse")
    if book is not None and chapter is not None and verse is not None:
        return f"{cat}:{sub}:{book}:{chapter}:{verse}"
    surah = metadata.get("Surah")
    ayah = metadata.get("Ayah")
    if surah is not None and ayah is not None:
        return f"{cat}:{sub}:Surah{surah}:{ayah}"
    return f"{cat}:{sub}:row-{row_idx}"


def _default_text_columns(df: pd.DataFrame) -> list[str]:
    skip = {a.lower() for a in _CATEGORY_ALIASES + _SUBCATEGORY_ALIASES}
    text_cols: list[str] = []
    for col in df.columns:
        if col.lower() in skip:
            continue
        series = df[col]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            text_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            if col.lower() in {
                "chapter",
                "verse",
                "chap",
                "v",
                "surah",
                "sura",
                "ayah",
                "aya",
            }:
                text_cols.append(col)
    return text_cols
