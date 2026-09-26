"""Application-wide MCP connections; no agent owns connection state."""
import copy
import hashlib
import json
import threading
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, SchemaError

import mcp_client


class McpStorageError(Exception):
    """Saving server definitions failed; distinct from network OSError."""


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
            schema = tool['inputSchema']
            # Local schema references are supported; discovery must never fetch remote schemas.
            def external_ref(value):
                if isinstance(value, dict):
                    return any((k in {'$ref', '$dynamicRef'} and isinstance(v, str) and not v.startswith('#'))
                               or external_ref(v) for k,v in value.items())
                return isinstance(value, list) and any(external_ref(v) for v in value)
            if external_ref(schema) or len(json.dumps(schema)) > 32768:
                raise ValueError('Схема MCP-инструмента не поддерживается.')
            Draft202012Validator.check_schema(schema)
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

    def resolve(self, name):
        with self._lock:
            for server in self._servers.values():
                if server['status'] == 'connected':
                    for tool in server['tools']:
                        if self._tool_name(server['id'], tool['name']) == name:
                            return server['url'], tool['name']
        raise ValueError('MCP-сервер отключён или инструмент неизвестен.')

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
        url, original_name = self.resolve(name)
        return self._invoke(url, original_name, arguments)
