"""Application-wide MCP connections; no agent owns connection state."""
import copy
import hashlib
import json
import re
import threading
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, SchemaError

from deepseek_app.mcp import mcp_client


class McpStorageError(Exception):
    """Saving server definitions failed; distinct from network OSError."""


def _validate_schema(schema):
    def external_ref(value):
        if isinstance(value, dict):
            return any((k in {'$ref', '$dynamicRef'} and isinstance(v, str) and not v.startswith('#'))
                       or external_ref(v) for k, v in value.items())
        return isinstance(value, list) and any(external_ref(v) for v in value)

    if not isinstance(schema, dict) or external_ref(schema) or len(json.dumps(schema)) > 32768:
        raise ValueError('Схема MCP-инструмента не поддерживается.')
    Draft202012Validator.check_schema(schema)


class McpRegistry:
    def __init__(self, state_path=None, discover=None, invoke=None):
        self._lock = threading.RLock()
        self._path = Path(state_path) if state_path else None
        self._discover = discover or mcp_client.list_tools
        self._invoke = invoke or mcp_client.call_tool
        self._servers = {}
        if self._path:
            try:
                definitions = json.loads(self._path.read_text())
                if not isinstance(definitions, list):
                    raise ValueError()
                for item in definitions:
                    url = mcp_client.validate_url(item['url'])
                    self._servers[self._id(url)] = self._entry(url)
            except (OSError, ValueError, KeyError, TypeError):
                self._servers = {}

    @staticmethod
    def _id(url):
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _entry(self, url):
        return {'id': self._id(url), 'url': url, 'status': 'disconnected', 'tools': [], 'revision': 0}

    def _save(self):
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_suffix('.tmp')
        temp.write_text(json.dumps([{'url': s['url']} for s in self._servers.values()]))
        temp.replace(self._path)

    def servers(self):
        with self._lock:
            return [copy.deepcopy({k: v for k,v in s.items() if k != 'revision'}) for s in self._servers.values()]

    def connect(self, url):
        url = mcp_client.validate_url(url)
        server_id = self._id(url)
        with self._lock:
            entry = self._servers.setdefault(server_id, self._entry(url))
            entry['revision'] += 1
            revision = entry['revision']
        tools = self._discover(url)
        if not isinstance(tools, list) or len(tools) > 100:
            raise ValueError('MCP-сервер должен вернуть до 100 инструментов.')
        seen = set()
        for tool in tools:
            if (not isinstance(tool, dict) or not isinstance(tool.get('name'), str)
                    or not 0 < len(tool['name']) <= 200 or tool['name'] in seen
                    or not isinstance(tool.get('inputSchema'), dict)):
                raise ValueError('Некорректное описание MCP-инструмента.')
            seen.add(tool['name'])
            _validate_schema(tool['inputSchema'])
            if 'outputSchema' in tool:
                _validate_schema(tool['outputSchema'])
        with self._lock:
            if entry['revision'] == revision:
                before = copy.deepcopy(entry)
                entry.update(status='connected', tools=copy.deepcopy(tools))
                try:
                    self._save()
                except OSError as error:
                    entry.update(before)
                    raise McpStorageError("Не удалось сохранить настройки MCP.") from error
            return copy.deepcopy({k:v for k,v in entry.items() if k != 'revision'})

    def disconnect(self, server_id):
        with self._lock:
            if server_id not in self._servers:
                raise KeyError(server_id)
            entry = self._servers[server_id]
            entry.update(status='disconnected', tools=[], revision=entry['revision'] + 1)
            return copy.deepcopy({k:v for k,v in entry.items() if k != 'revision'})

    @staticmethod
    def _tool_name(server_id, original_name):
        return 'mcp_' + server_id + '_' + hashlib.sha256(original_name.encode()).hexdigest()[:16]

    def snapshot(self):
        with self._lock:
            return [{'type': 'function', 'function': {
                'name': self._tool_name(s['id'], tool['name']),
                'description': f"{tool['name']} ({s['url']}): {tool.get('description') or ''}"[:1500],
                'parameters': copy.deepcopy(tool['inputSchema'])}}
                for s in self._servers.values() if s['status'] == 'connected' for tool in s['tools']]

    def _resolve_tool(self, name):
        with self._lock:
            for server in self._servers.values():
                if server['status'] == 'connected':
                    for tool in server['tools']:
                        if self._tool_name(server['id'], tool['name']) == name:
                            return server['url'], copy.deepcopy(tool)
        raise ValueError('MCP-сервер отключён или инструмент неизвестен.')

    def resolve(self, name):
        url, tool = self._resolve_tool(name)
        return url, tool['name']

    def call(self, name, arguments, snapshot):
        spec = next((t['function'] for t in snapshot if t['function']['name'] == name), None)
        if spec is None:
            raise ValueError('Инструмент не был предложен модели.')
        if not isinstance(arguments, dict):
            raise ValueError('Аргументы инструмента должны быть JSON-объектом.')
        try:
            Draft202012Validator(spec['parameters']).validate(arguments)
        except (ValidationError, SchemaError):
            raise ValueError('Аргументы не соответствуют схеме MCP-инструмента.') from None
        url, tool = self._resolve_tool(name)
        result = self._invoke(url, tool['name'], arguments)
        if not isinstance(result, dict) or not isinstance(result.get('isError'), bool):
            return {'content': [{'type': 'text', 'text': 'MCP-сервер вернул некорректный ответ.'}],
                    'structuredContent': None, 'isError': True}
        if result['isError']:
            return result
        output_schema = tool.get('outputSchema')
        structured = result.get('structuredContent')
        if output_schema is not None:
            try:
                if not isinstance(structured, dict):
                    raise ValidationError('structuredContent должен быть объектом.')
                Draft202012Validator(output_schema).validate(structured)
            except (ValidationError, SchemaError):
                return {'content': [{'type': 'text', 'text': 'Результат MCP не соответствует выходной схеме.'}],
                        'structuredContent': None, 'isError': True}
        if url == mcp_client.LOCAL_PIPELINE_URL and tool['name'] == 'search':
            if not _valid_search_result(structured):
                return {'content': [{'type': 'text', 'text': 'Результат search не соответствует контракту.'}],
                        'structuredContent': None, 'isError': True}
        if url == mcp_client.LOCAL_PIPELINE_URL and tool['name'] == 'summarize':
            if not _valid_summary_result(structured):
                return {'content': [{'type': 'text', 'text': 'Результат summarize не соответствует контракту.'}],
                        'structuredContent': None, 'isError': True}
        if url == mcp_client.LOCAL_PIPELINE_URL and tool['name'] == 'saveToFile':
            if not _valid_saved_file(structured):
                return {'content': [{'type': 'text', 'text': 'Результат saveToFile не соответствует контракту.'}],
                        'structuredContent': None, 'isError': True}
        if url == mcp_client.LOCAL_SCHEDULER_URL and tool['name'] == 'schedule_repository':
            if not _valid_scheduled_job(structured, arguments):
                return {'content': [{'type': 'text', 'text': 'Результат schedule_repository не соответствует контракту.'}],
                        'structuredContent': None, 'isError': True}
        if url == mcp_client.LOCAL_SCHEDULER_URL and tool['name'] == 'get_schedule_summary':
            if not _valid_schedule_summary(structured, arguments):
                return {'content': [{'type': 'text', 'text': 'Результат get_schedule_summary не соответствует контракту.'}],
                        'structuredContent': None, 'isError': True}
        return result


def _valid_search_result(value):
    if not isinstance(value, dict) or not isinstance(value.get('items'), list) or len(value['items']) > 3:
        return False
    if (not isinstance(value.get('query'), str) or not value['query'].strip()
            or type(value.get('total_count')) is not int or value['total_count'] < 0
            or not isinstance(value.get('incomplete_results'), bool)):
        return False
    fetched_at = value.get('fetched_at')
    if not isinstance(fetched_at, str):
        return False
    try:
        from datetime import datetime
        if datetime.fromisoformat(fetched_at.replace('Z', '+00:00')).tzinfo is None:
            return False
    except ValueError:
        return False
    return all(isinstance(item, dict) and isinstance(item.get('full_name'), str)
               and bool(item['full_name'].strip()) and type(item.get('stars')) is int
               and item['stars'] >= 0 for item in value['items'])


def _valid_summary_result(value):
    if not isinstance(value, dict):
        return False
    markdown = value.get('markdown')
    return (isinstance(value.get('query'), str) and bool(value['query'].strip())
            and type(value.get('source_count')) is int and 0 <= value['source_count'] <= 3
            and isinstance(markdown, str) and bool(markdown.strip())
            and len(markdown.encode('utf-8')) <= 6000)


def _valid_saved_file(value):
    if not isinstance(value, dict):
        return False
    return (isinstance(value.get('path'), str) and 0 < len(value['path'].strip()) <= 4096
            and type(value.get('bytes_written')) is int and 1 <= value['bytes_written'] <= 6000
            and isinstance(value.get('sha256'), str) and re.fullmatch(r'[a-f0-9]{64}', value['sha256']) is not None)


def _valid_scheduled_job(value, arguments):
    job = value.get('job') if isinstance(value, dict) else None
    if not isinstance(job, dict):
        return False
    if (not isinstance(job.get('id'), str) or not job['id'].strip()
            or job.get('owner') != arguments.get('owner') or job.get('repo') != arguments.get('repo')
            or job.get('status') != 'active' or job.get('interval_seconds') is not None):
        return False
    next_run = job.get('next_run_at')
    if not isinstance(next_run, str) or not next_run.strip():
        return False
    try:
        from datetime import datetime
        parsed = datetime.fromisoformat(next_run.replace('Z', '+00:00'))
        return parsed.tzinfo is not None
    except ValueError:
        return False


def _valid_schedule_summary(value, arguments):
    summary = value if isinstance(value, dict) else None
    job = summary.get('job') if summary else None
    return (isinstance(job, dict) and job.get('id') == arguments.get('job_id')
            and isinstance(job.get('owner'), str) and isinstance(job.get('repo'), str)
            and type(summary.get('success_count')) is int and summary['success_count'] >= 0
            and type(summary.get('error_count')) is int and summary['error_count'] >= 0)
