# Scripture CSV + Milvus + llama.cpp RAG (LlamaIndex)

Ingest scripture CSVs into **Milvus**, retrieve by embedding similarity, and answer with **llama.cpp** via **LlamaIndex**.

Supports:

- **category** — religion / scripture corpus (e.g. `Christianity`, `Islam`)
- **subcategory** (translation / interpretation) — e.g. `KJV`, `NIV`, `NKJV`

## Setup

```bash
cd milvusproto
python3 -m venv .venv   # or: source virtual/bin/activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `EMBED_DIM`, `LLAMA_BASE_URL`, `LLAMA_EMBED_BASE_URL`, and `MILVUS_URL` in `.env`.

## Taxonomy

| Field | Meaning | Examples |
|-------|---------|----------|
| `category` | Which scripture / tradition | `Christianity`, `Islam` |
| `subcategory` | Translation or interpretation | `KJV`, `NIV`, `NKJV` |

Set via CLI flags, or CSV columns named `Category` / `Subcategory` / `Translation` / `Version`.

Use **one** Milvus collection (e.g. `scriptures`) and **`--append`** when adding another translation so they can be compared.

## Ingest

Put Bible and Quran in **one** collection (`COLLECTION_NAME=scriptures`) so ask can retrieve both.

```bash
# 1) Bible first (creates collection)
python ingest.py data/bible_sample.csv --preset bible
# full file examples:
# python ingest.py bible.csv --preset kjv
# python ingest.py niv.csv --preset niv --append

# 2) Quran into the SAME collection
python ingest.py data/quran_sample.csv --preset quran --append
# python ingest.py quran.csv --preset quran --append
```

| `--preset` | category | subcategory |
|------------|----------|-------------|
| `bible` / `kjv` | Christianity | KJV |
| `niv` / `nkjv` | Christianity | NIV / NKJV |
| `quran` / `islam` | Islam | SahihInternational |

CSV auto-detect: Bible `Book,Chapter,Verse,Text` · Quran `Surah,Ayah,SurahName,Text`.

**Important:** Recreate once without `--append`, then `--append` every additional corpus.

## Ask / analyze / compare

```bash
# Search everything in the collection (Bible + Quran if both loaded)
python ask.py "What do the scriptures say about mercy?" --show-hits

# Bible only
python ask.py --category Christianity --translation KJV \
  "What does Genesis 1:1 say?"

# Quran only
python ask.py --category Islam --translation SahihInternational \
  "What does Al-Fatiha say about the straight path?"

# Compare Bible vs Quran (both must be ingested with --append)
python ask.py --compare --category Christianity,Islam \
  "What do these texts say about mercy and guidance?" --show-hits

# Compare Bible translations
python ask.py --compare --translation KJV,NIV,NKJV \
  "How do these translations describe salvation?"
```

`--show-hits` prints retrieved passages. Citations look like `Christianity/KJV: John 3:16` or `Islam/SahihInternational: Al-Fatiha 1:5`.

## Layout

| Path | Role |
|------|------|
| `ingest.py` | CSV → Documents (+ category/subcategory) → Milvus |
| `ask.py` | Filter / analyze / compare query CLI |
| `src/csv_ingest.py` | CSV → `Document` list |
| `src/llamaindex_app.py` | LlamaIndex + Milvus filters + prompts |


Use llama.cpp

  823  podman run -d   --name llama-server-embedding   --network host   --device /dev/dri   --group-add keep-groups   --security-opt label=disable   -v ~/models:/models:Z   llama-vulkan-fixed   -m /models/Qwen3-Embedding-0.6B-Q8_0.gguf    -ngl 99 -fa 1 -c 16384   --host 0.0.0.0 --port 8081 --embedding --pooling last
  840  podman run -d   --name llama-server   --network host   --device /dev/dri   --group-add keep-groups   --security-opt label=disable   -v ~/models:/models:Z   llama-vulkan-fixed   -m /models/Qwen3-14B-Q4_K_M.gguf   -ngl 99 -fa 1 -c 16384   --host 0.0.0.0 --port 8080   --embeddings --pooling mean

Use the embedding model for importing milvis bible
use the the regular qeb3-14b for the real AI stuff language

Stack: **LlamaIndex** → llama.cpp (chat + embeddings) + **Milvus**.
