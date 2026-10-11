"""Strict, bounded contracts shared by the research server and host runner."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MARKDOWN_BYTES = 6000
MAX_RESULT_BYTES = 16_384
Query = Annotated[str, Field(min_length=1, max_length=200)]
Limit = Annotated[int, Field(strict=True, ge=1, le=3)]


def normalize_query(query: str) -> str:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
        raise ValueError('Укажите поисковый запрос длиной от 1 до 200 символов.')
    return query.strip()


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')


class SearchArguments(StrictModel):
    query: Query
    limit: Limit = 3

    @field_validator('query', mode='before')
    @classmethod
    def clean_query(cls, value):
        return normalize_query(value)


class RepositoryItem(StrictModel):
    full_name: Annotated[str, Field(min_length=1, max_length=256)]
    description: Annotated[str, Field(max_length=300)] | None
    language: Annotated[str, Field(min_length=1, max_length=100)] | None
    stars: Annotated[int, Field(ge=0, le=10**12)]
    url: Annotated[str, Field(max_length=512, pattern=r'^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')]


class SearchResult(StrictModel):
    query: Query
    fetched_at: Annotated[str, Field(min_length=1, max_length=40)]
    total_count: Annotated[int, Field(ge=0, le=10**12)]
    incomplete_results: bool
    items: Annotated[list[RepositoryItem], Field(max_length=3)]

    @field_validator('fetched_at')
    @classmethod
    def timestamp_has_timezone(cls, value):
        timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            raise ValueError('fetched_at must include a timezone')
        return value


class SummaryResult(StrictModel):
    query: Query
    source_count: Annotated[int, Field(ge=0, le=3)]
    markdown: Annotated[str, Field(min_length=1, max_length=MAX_MARKDOWN_BYTES)]

    @field_validator('markdown')
    @classmethod
    def bounded_utf8(cls, value):
        if len(value.encode('utf-8')) > MAX_MARKDOWN_BYTES:
            raise ValueError('Markdown превышает 6000 байт UTF-8.')
        return value


class SavedFile(StrictModel):
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    bytes_written: Annotated[int, Field(ge=1, le=MAX_MARKDOWN_BYTES)]
    sha256: Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
