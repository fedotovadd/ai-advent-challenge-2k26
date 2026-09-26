import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from mcp_registry import McpRegistry
from web import ChatServer

TOOL = {'name':'repo','description':'repo','inputSchema':{'type':'object','properties':{}}}

class McpWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.mcp = McpRegistry(discover=lambda url:[TOOL],invoke=lambda *args:{'isError':False,'content':[],'structuredContent':{'stars':7}})
        self.calls=[]
        def ask(payload, **options):
            self.calls.append((payload,options))
            if options.get('tools') and payload['messages'][-1]['role'] != 'tool':
                return {'content':None,'tool_calls':[{'id':'c1','type':'function','function':{'name':options['tools'][0]['function']['name'],'arguments':'{}'}}]}
            return '7 stars'
        self.server=ChatServer(('127.0.0.1',0),ask,Path(self.temp.name)/'agents.json',mcp_registry=self.mcp)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.temp.cleanup()
    def request(self,method,path,body=None,headers=None):
        connection=http.client.HTTPConnection(*self.server.server_address)
        connection.request(method,path,body=json.dumps(body) if body is not None else None,headers=headers or {})
        response=connection.getresponse(); data=json.loads(response.read()); connection.close()
        return response.status,data
    def test_global_connect_all_agents_then_disconnect(self):
        status, body=self.request('POST','/api/mcp/servers/connect',{'url':'http://127.0.0.1:8001/mcp'})
        self.assertEqual(status,200)
        sid=body['server']['id']
        self.assertEqual(self.request('GET','/api/mcp/servers')[1]['servers'][0]['status'],'connected')
        new=self.request('POST','/api/agents')[1]['agent']['id']
        for aid in ['agent-1',new]:
            status,result=self.request('POST',f'/api/agents/{aid}/messages',{'text':'get repo'})
            self.assertEqual(status,200,result)
            self.assertEqual(result['agent']['metadata']['toolCalls'][0]['name'],'repo')
        self.assertEqual(self.request('POST',f'/api/mcp/servers/{sid}/disconnect')[0],200)
        self.request('POST','/api/agents/agent-1/messages',{'text':'hello'})
        self.assertNotIn('tools',self.calls[-1][1])
    def test_bad_url_schema_origin_and_oversized_body(self):
        for body in [{'url':'http://example.com'}, {'url':'https://example.com','other':1}, {'url':'x'*5000}]:
            self.assertEqual(self.request('POST','/api/mcp/servers/connect',body)[0],400)
        self.assertEqual(self.request('POST','/api/mcp/servers/connect',{'url':'https://example.com'}, {'Origin':'https://evil.example'})[0],403)

    def test_discovery_timeout_is_connection_error_not_storage_error(self):
        from unittest.mock import patch
        with patch.object(self.mcp,'_discover',side_effect=TimeoutError()):
            status,body=self.request('POST','/api/mcp/servers/connect',{'url':'https://example.com/mcp'})
        self.assertEqual(status,502)
        self.assertIn('подключить',body['error'])
