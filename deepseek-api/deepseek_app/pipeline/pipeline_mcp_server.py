"""Day 19: python pipeline_mcp_server.py; Streamable HTTP on loopback port 8003."""

import json
from pathlib import Path
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, BeforeValidator, ValidationError

from deepseek_app.mcp import github_api
from deepseek_app.pipeline import pipeline_tools
from deepseek_app.pipeline.pipeline_models import (
    Limit, MAX_RESULT_BYTES, Query, SavedFile, SearchResult, SummaryResult, normalize_query,
)


def bounded_result(model: BaseModel) -> CallToolResult:
    data = model.model_dump(mode='json')
    result = CallToolResult(
        content=[TextContent(type='text', text=json.dumps(data, ensure_ascii=False))],
        structuredContent=data,
    )
    if len(json.dumps(result.model_dump(mode='json'), ensure_ascii=False).encode('utf-8')) > MAX_RESULT_BYTES:
        raise ValueError('Результат MCP превышает 16 KiB.')
    return result


class PipelineMCP(FastMCP):
    async def call_tool(self, name, arguments):
        # SDK v1 wraps validation and tool exceptions in ToolError. Return a
        # bounded message without reflecting arbitrary arguments or OS errors.
        try:
            tool = self._tool_manager.get_tool(name)
            if tool is None:
                raise ValueError('Неизвестный MCP-инструмент.')
            # SDK v1 otherwise turns JSON strings into nested objects before
            # validation, violating the published strict object contract.
            tool.fn_metadata.arg_model.model_validate(arguments, strict=True)
            return await super().call_tool(name, arguments)
        except Exception as error:
            cause = error
            while cause.__cause__ is not None:
                cause = cause.__cause__
            if isinstance(cause, github_api.GitHubAPIError):
                message = str(cause)
            elif isinstance(cause, OSError):
                message = 'Не удалось сохранить отчёт. Проверьте доступность каталога data/reports.'
            elif isinstance(cause, ValueError) and not isinstance(cause, ValidationError):
                message = str(cause)[:300]
            else:
                message = 'Некорректные аргументы или результат MCP-инструмента.'
            return CallToolResult(isError=True, content=[TextContent(type='text', text=message)])


def create_server(*, client: httpx.AsyncClient | None = None,
                  reports_dir: Path | None = None) -> FastMCP:
    mcp = PipelineMCP('Public GitHub research', host='127.0.0.1', port=8003,
                      streamable_http_path='/mcp',
                      instructions='Последовательно вызовите search, summarize, saveToFile; передавайте structuredContent целиком.')

    @mcp.tool(description='Найти до трёх публичных репозиториев GitHub по поисковому запросу.',
              annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True))
    async def search(query: Annotated[Query, BeforeValidator(normalize_query)], limit: Limit = 3) -> SearchResult:
        result = await github_api.search_repositories(query, limit=limit, client=client)
        return bounded_result(SearchResult.model_validate(result))

    @mcp.tool(description='Сформировать Markdown-обзор по результату search, без LLM и новых сетевых запросов.',
              annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    def summarize(search_result: SearchResult) -> SummaryResult:
        return bounded_result(SummaryResult.model_validate(pipeline_tools.summarize(search_result.model_dump())))

    @mcp.tool(description='Создать новый Markdown-файл в data/reports с новым UUID-именем. Каждый вызов создаёт отдельный файл.',
              annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=False, openWorldHint=False))
    def saveToFile(report: SummaryResult) -> SavedFile:
        return bounded_result(SavedFile.model_validate(pipeline_tools.save_to_file(report.model_dump(), reports_dir=reports_dir)))

    # Keep SDK v1's argument validator and advertised schemas aligned.
    for name in ('search', 'summarize', 'saveToFile'):
        tool = mcp._tool_manager.get_tool(name)
        tool.fn_metadata.arg_model.model_config.update(extra='forbid', strict=True)
        tool.fn_metadata.arg_model.model_rebuild(force=True)
        tool.parameters = tool.fn_metadata.arg_model.model_json_schema()
    return mcp


mcp = create_server()

if __name__ == '__main__':
    mcp.run(transport='streamable-http')
