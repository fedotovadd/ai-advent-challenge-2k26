import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from agent import AgentRegistry
from mcp_registry import McpRegistry

TOOL = {'name':'repo', 'description':'repo', 'inputSchema':{'type':'object','properties':{}}}
USAGE = {'prompt_tokens':10, 'completion_tokens':2, 'total_tokens':12}

class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'agents.json'
        self.mcp = McpRegistry(discover=lambda url:[TOOL], invoke=lambda *args:{'content':[], 'structuredContent':{'stars':7}, 'isError':False})
        self.server = self.mcp.connect('http://127.0.0.1:8001/mcp')
        self.name = self.mcp.snapshot()[0]['function']['name']
        self.ask = Mock()

    def answers(self, last=None):
        self.ask.side_effect = [
            {'content':None, 'tool_calls':[{'id':'c1','type':'function','function':{'name':self.name, 'arguments':'{}'}}], 'usage':USAGE},
            last or {'content':'7 stars','usage':USAGE}]

    def test_global_tools_apply_to_existing_new_and_restored_agents(self):
        registry = AgentRegistry(self.ask, self.path, mcp_registry=self.mcp)
        for agent_id in ['agent-1', registry.create()['id']]:
            self.answers()
            result = registry.respond(agent_id, 'github?')
            self.assertEqual(result['messages'][-1]['content'], '7 stars')
            self.assertEqual(result['metadata']['usage']['totalTokens'],24)
            self.assertEqual(len(result['metadata']['toolCalls']),1)
            self.assertEqual(result['metrics']['totals']['totalTokens'],24)
            self.assertTrue(self.ask.call_args_list[-1].kwargs['tools'])
        restored = AgentRegistry(self.ask, self.path, mcp_registry=self.mcp)
        self.assertEqual(len(restored.agents()),2)
        self.assertEqual(len(restored.agents()[0]['metadata']['toolCalls']),1)
        self.mcp.disconnect(self.server['id'])
        self.ask.side_effect = None
        self.ask.return_value = 'plain'
        restored.respond('agent-1','hello')
        self.assertNotIn('tools',self.ask.call_args.kwargs)

    def test_failed_final_request_preserves_known_usage_once(self):
        registry = AgentRegistry(self.ask,self.path,mcp_registry=self.mcp)
        self.answers(RuntimeError('failed'))
        with self.assertRaises(RuntimeError):
            registry.respond('agent-1','github?')
        restored = AgentRegistry(self.ask,self.path,mcp_registry=self.mcp)
        self.assertEqual(restored.agents()[0]['metrics']['totals']['totalTokens'],12)
        self.assertEqual(len(restored.agents()[0]['metadata']['toolCalls']),1)
        self.answers()
        result = restored.respond('agent-1','retry')
        self.assertEqual(result['metrics']['totals']['totalTokens'],36)

    def test_old_state_metadata_backfilled(self):
        registry = AgentRegistry(lambda *a,**k:'hello',self.path)
        registry.respond('agent-1','hello')
        state = json.loads(self.path.read_text())
        state['version'] = 10
        def strip(value):
            if isinstance(value,dict):
                value.pop('toolCalls',None); value.pop('modelRequests',None)
                for v in value.values(): strip(v)
            elif isinstance(value,list):
                for v in value: strip(v)
        strip(state)
        self.path.write_text(json.dumps(state))
        restored = AgentRegistry(self.ask,self.path,mcp_registry=self.mcp)
        self.assertEqual(restored.agents()[0]['messages'][-1]['content'],'hello')
        self.assertEqual(restored.agents()[0]['metadata']['toolCalls'],[])

    def test_settings_rollback_keeps_shared_registry(self):
        from unittest.mock import patch
        from agent import PersistenceError, default_settings
        registry = AgentRegistry(self.ask,self.path,mcp_registry=self.mcp)
        with patch.object(registry,'_save',side_effect=PersistenceError('disk')):
            with self.assertRaises(PersistenceError):
                registry.update_settings('agent-1',{**default_settings(),'systemPrompt':'changed'})
        self.answers()
        result=registry.respond('agent-1','github?')
        self.assertEqual(len(result['metadata']['toolCalls']),1)

    def test_failed_execute_preserves_usage_but_restores_planning(self):
        registry = AgentRegistry(self.ask,self.path,mcp_registry=self.mcp)
        self.ask.return_value={'content':'[[TASK_PLAN]]\n1. Get repo\n[[/TASK_PLAN]]','usage':USAGE}
        registry.apply_task_command('agent-1',{'action':'task','title':'repo'})
        before=registry.agents()[0]['metrics']['totals']['totalTokens']
        self.answers(RuntimeError('failed'))
        with self.assertRaises(RuntimeError):
            registry.apply_task_command('agent-1',{'action':'execute'})
        result=registry.agents()[0]
        self.assertEqual(result['context']['taskState']['stage'],'PLANNING')
        self.assertEqual(result['metrics']['totals']['totalTokens'],before+12)

    def test_mcp_instruction_is_visible_and_not_counted_as_compression_loss(self):
        registry=AgentRegistry(self.ask,self.path,mcp_registry=self.mcp)
        self.answers()
        result=registry.respond('agent-1','repo?')
        self.assertEqual(result['metadata']['systemPrompt'],result['metadata']['payload']['messages'][0]['content'])
        self.assertEqual(result['metrics']['lastMessage']['compressionGrossSavedTokens'],0)
