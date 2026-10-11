"""Deterministic research processing and exclusive local report creation."""

import hashlib
import html
import re
from pathlib import Path
from uuid import uuid4

from deepseek_app.pipeline.pipeline_models import SavedFile, SearchResult, SummaryResult

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / 'data' / 'reports'


def _markdown_text(value: str) -> str:
    """Keep external metadata as a single line of literal Markdown text."""
    value = html.escape(' '.join(value.split()), quote=False)
    return re.sub(r'([\\`*_{}\[\]()#+.!|>~-])', r'\\\1', value)


def summarize(search_result: dict) -> dict:
    result = SearchResult.model_validate(search_result)
    lines = ['# Обзор публичных репозиториев GitHub', '',
             f'Запрос: {_markdown_text(result.query)}',
             f'Получено репозиториев: {len(result.items)}. Всего найдено: {result.total_count}.',
             f'Данные получены: {_markdown_text(result.fetched_at)}', '']
    if result.incomplete_results:
        lines.extend(['GitHub вернул неполные результаты поиска.', ''])
    if not result.items:
        lines.extend(['Ничего не найдено.', ''])
    for item in result.items:
        lines.extend([
            f'## {_markdown_text(item.full_name)}', '',
            f'Язык: {_markdown_text(item.language) if item.language else "не указан"}. Звёзды: {item.stars}.',
            _markdown_text(item.description) if item.description else 'Описание не указано.',
            f'Источник: [{_markdown_text(item.full_name)}]({item.url})', '',
        ])
    return SummaryResult(query=result.query, source_count=len(result.items), markdown='\n'.join(lines)).model_dump()


def save_to_file(report: dict, *, reports_dir: Path | None = None) -> dict:
    validated = SummaryResult.model_validate(report)
    content = validated.markdown.encode('utf-8')
    directory = Path(reports_dir) if reports_dir is not None else DEFAULT_REPORTS_DIR
    directory = directory.resolve()
    path = directory / f'{uuid4()}.md'
    saved = SavedFile(path=str(path), bytes_written=len(content),
                      sha256=hashlib.sha256(content).hexdigest()).model_dump()
    directory.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open('xb') as output:
            created = True
            written = output.write(content)
            if written != len(content):
                raise OSError('Неполная запись отчёта.')
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    return saved
