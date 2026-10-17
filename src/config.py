"""Load settings from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llama_base_url: str
    llama_embed_base_url: str
    llama_model: str
    llama_embed_model: str
    llama_api_key: str
    llama_timeout: float
    llama_max_tokens: int
    embed_dim: int
    # Remote Milvus (K8s) takes precedence over local Lite path when set
    milvus_url: str | None
    milvus_token: str | None
    milvus_user: str | None
    milvus_password: str | None
    milvus_db_name: str
    milvus_db_path: str
    collection_name: str
    text_cols: list[str] | None

    @property
    def milvus_uri(self) -> str:
        """URI passed to MilvusClient (remote URL or local .db path)."""
        return self.milvus_url or self.milvus_db_path

    @property
    def is_remote_milvus(self) -> bool:
        return bool(self.milvus_url)


def get_settings() -> Settings:
    text_cols_raw = os.getenv("TEXT_COLS", "").strip()
    text_cols = (
        [c.strip() for c in text_cols_raw.split(",") if c.strip()]
        if text_cols_raw
        else None
    )

    # Remote server: use MILVUS_URL (http[s]://host:19530). Do not put a .db path
    # in MILVUS_URI — that env name is reserved by pymilvus and breaks imports.
    milvus_url = (
        os.getenv("MILVUS_URL", "").strip()
        or os.getenv("MILVUS_ADDRESS", "").strip()
        or None
    )
    if milvus_url:
        milvus_url = milvus_url.rstrip("/")
        if not milvus_url.startswith(("http://", "https://", "tcp://", "unix://")):
            # host:port -> http://host:port (MilvusClient default)
            milvus_url = f"http://{milvus_url}"

    milvus_db_path = os.getenv("MILVUS_DB_PATH") or os.getenv(
        "MILVUS_LITE_URI", "./milvus_demo.db"
    )
    milvus_path = Path(milvus_db_path)
    if not milvus_path.is_absolute():
        milvus_db_path = str((_PROJECT_ROOT / milvus_path).resolve())

    token = os.getenv("MILVUS_TOKEN", "").strip() or None
    user = os.getenv("MILVUS_USER", "").strip() or None
    password = os.getenv("MILVUS_PASSWORD", "").strip() or None
    # Convenience: USER+PASSWORD without TOKEN -> token="user:password"
    if not token and user and password is not None:
        token = f"{user}:{password}"

    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1").rstrip(
        "/"
    )
    llama_embed_base_url = (
        os.getenv("LLAMA_EMBED_BASE_URL", "").strip().rstrip("/") or llama_base_url
    )
    llama_model = os.getenv("LLAMA_MODEL", "local")
    llama_embed_model = os.getenv("LLAMA_EMBED_MODEL", "").strip() or llama_model

    return Settings(
        llama_base_url=llama_base_url,
        llama_embed_base_url=llama_embed_base_url,
        llama_model=llama_model,
        llama_embed_model=llama_embed_model,
        llama_api_key=os.getenv("LLAMA_API_KEY", "not-needed"),
        # 14B analyze prompts can take many minutes on CPU/slow GPU
        llama_timeout=float(os.getenv("LLAMA_TIMEOUT", "86400")),
        llama_max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "2048")),
        embed_dim=int(os.getenv("EMBED_DIM", "1024")),
        milvus_url=milvus_url,
        milvus_token=token,
        milvus_user=user,
        milvus_password=password,
        milvus_db_name=os.getenv("MILVUS_DB_NAME", "default").strip() or "default",
        milvus_db_path=milvus_db_path,
        collection_name=os.getenv("COLLECTION_NAME", "csv_rows"),
        text_cols=text_cols,
    )
