"""Local GitHub MCP server. Run with: python github_mcp_server.py."""

import json
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

import github_api


def create_server(*, client: httpx.AsyncClient | None = None) -> FastMCP:
    mcp = FastMCP(
        'GitHub repositories',
        instructions='Получение актуальных метаданных публичного репозитория GitHub.',
        host='127.0.0.1', port=8001, streamable_http_path='/mcp',
    )

    @mcp.tool(
        description='Получить актуальные метаданные публичного репозитория GitHub по владельцу и имени.',
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True),
    )
    async def get_repository(
        owner: Annotated[str, Field(description='Владелец репозитория: пользователь или организация GitHub.')],
        repo: Annotated[str, Field(description='Имя публичного репозитория GitHub без владельца и URL.')],
    ) -> CallToolResult:
        try:
            data = await github_api.get_repository(owner, repo, client=client)
        except github_api.GitHubAPIError as error:
            return CallToolResult(isError=True, content=[TextContent(type='text', text=str(error))])
        return CallToolResult(
            content=[TextContent(type='text', text=json.dumps(data, ensure_ascii=False))],
            structuredContent=data,
        )

    # FastMCP v1 does not expose extra=forbid through @tool. Configure its
    # generated argument model and published schema together so they agree.
    tool = mcp._tool_manager.get_tool('get_repository')
    tool.fn_metadata.arg_model.model_config['extra'] = 'forbid'
    tool.fn_metadata.arg_model.model_rebuild(force=True)
    tool.parameters = tool.fn_metadata.arg_model.model_json_schema()
    return mcp


mcp = create_server()


if __name__ == '__main__':
    mcp.run(transport='streamable-http')
