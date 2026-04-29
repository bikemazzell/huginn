from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key: str
    model: str


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chat: ModelEndpointConfig
    embedding: ModelEndpointConfig


class IndexingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "IndexingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ocr_fallback: bool
    query_rewrite: bool
    rerank: bool
    answer_validation: bool


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_path: Path
    local_only: bool
    models: ModelsConfig
    indexing: IndexingConfig
    features: FeaturesConfig

    @model_validator(mode="after")
    def validate_local_only_endpoints(self) -> "RuntimeConfig":
        if not self.local_only:
            return self

        for endpoint in (self.models.chat, self.models.embedding):
            host = urlparse(endpoint.base_url).hostname
            if host not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("local_only requires localhost model endpoints")
        return self


class ExtractedPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(gt=0)
    text: str


class ExtractedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    title: str
    pages: list[ExtractedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_index: int = Field(ge=0)
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    text: str
    token_count: int = Field(ge=0)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: int
    source_path: str
    page_start: int
    page_end: int
    text: str
    score: float


class QueryAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_text: str
    citations: list[str]
    evidence_note: str | None = None


class IngestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discovered_count: int = 0
    indexed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
