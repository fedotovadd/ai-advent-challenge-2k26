"""Read public repository metadata from the GitHub REST API."""

import os
import unicodedata
from datetime import datetime, timezone
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

GITHUB_API_URL = 'https://api.github.com'
REQUEST_TIMEOUT = 10.0


class GitHubAPIError(Exception):
    """An error whose message is safe to display to the user."""


class _Repository(BaseModel):
    model_config = ConfigDict(strict=True)

    full_name: str
    description: str | None
    language: str | None
    stargazers_count: int
    forks_count: int
    default_branch: str
    html_url: str
    archived: bool
    pushed_at: str | None
    private: bool


def _validate_identifier(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {'.', '..'}
        or '/' in value
        or '\\' in value
        or any(unicodedata.category(char) == 'Cc' for char in value)
    ):
        raise GitHubAPIError(f'Некорректное {label}: укажите непустое имя без разделителей пути и управляющих символов.')


def _response_error(response: httpx.Response) -> GitHubAPIError:
    status = response.status_code
    if status == 401:
        message = 'GitHub отклонил токен. Проверьте GITHUB_TOKEN.'
    elif status in {403, 429}:
        rate_limited = status == 429 or response.headers.get('x-ratelimit-remaining') == '0' or 'retry-after' in response.headers
        if status == 403 and not rate_limited:
            try:
                body = response.json()
                detail = body.get('message', '') if isinstance(body, dict) else ''
                rate_limited = isinstance(detail, str) and 'rate limit' in detail.lower()
            except ValueError:
                pass
        message = (
            'Превышен лимит запросов GitHub. Повторите попытку позже.'
            if rate_limited else 'GitHub запретил доступ. Проверьте разрешения токена и публичность репозитория.'
        )
    elif status == 404:
        message = 'Публичный репозиторий не найден. Проверьте владельца и имя репозитория.'
    elif 300 <= status < 400:
        message = 'GitHub вернул перенаправление. Укажите актуальные владельца и имя репозитория.'
    elif status >= 500:
        message = 'GitHub временно недоступен. Повторите попытку позже.'
    else:
        message = 'Не удалось получить данные репозитория из GitHub.'
    return GitHubAPIError(message)


async def get_repository(owner: str, repo: str, *, client: httpx.AsyncClient | None = None) -> dict:
    """Return the fixed public metadata fields; never follow API redirects."""
    _validate_identifier(owner, 'имя владельца')
    _validate_identifier(repo, 'имя репозитория')
    if client is None:
        async with httpx.AsyncClient() as owned_client:
            return await get_repository(owner, repo, client=owned_client)

    headers = {'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
    token = os.environ.get('GITHUB_TOKEN', '').strip()
    if token:
        headers['Authorization'] = f'Bearer {token}'
    url = f'{GITHUB_API_URL}/repos/{quote(owner, safe="")}/{quote(repo, safe="")}'
    try:
        response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT, follow_redirects=False)
    except httpx.TimeoutException:
        raise GitHubAPIError('GitHub не ответил за 10 секунд. Повторите попытку позже.') from None
    except httpx.RequestError:
        raise GitHubAPIError('Не удалось подключиться к GitHub. Проверьте сеть и повторите попытку.') from None
    if response.status_code != 200:
        raise _response_error(response)
    try:
        repository = _Repository.model_validate(response.json())
    except (ValueError, ValidationError):
        raise GitHubAPIError('GitHub вернул некорректные данные репозитория. Повторите попытку позже.') from None
    if repository.private:
        raise GitHubAPIError('Доступны только публичные репозитории GitHub.')
    result = repository.model_dump(exclude={'private'})
    result['fetched_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    return result
