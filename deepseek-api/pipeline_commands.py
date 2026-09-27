"""Explicit chat command; it never asks a model to choose tools."""
import re


def parse_research_command(text):
    if not isinstance(text, str):
        return None
    match = re.match(r'^/research(?:\s+(.*))?$', text.strip(), re.DOTALL)
    if not match:
        return None
    query = (match.group(1) or '').strip()
    if not query:
        raise ValueError('Укажите запрос: /research mcp language:python')
    if len(query) > 2000 or any(ord(c) < 32 for c in query):
        raise ValueError('Запрос должен быть одной строкой от 1 до 2000 символов.')
    return query
