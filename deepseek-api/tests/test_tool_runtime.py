import copy
import json
import unittest
from unittest.mock import Mock
from mcp_registry import McpRegistry
from tool_runtime import run_tools

TOOL = {'name':'repo', 'description':'repo', 'inputSchema':{'type':'object','properties':{}}}
USAGE = {'prompt_tokens':10, 'completion_tokens':2, 'total_tokens':12}


class ToolRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.registry = McpRegistry(discover=lambda url:[TOOL], invoke=lambda *a:{'content':[], 'structuredContent':{'stars':7}, 'isError':False})
        self.registry.connect('http://127.0.0.1:8001/mcp')
        self.tools = self.registry.snapshot()
        self.name = self.tools[0]['function']['name']
        self.payload = {'model':'deepseek-v4-flash','messages':[{'role':'user','content':'stars?'}], 'temperature':1}
        self.metadata = {'toolCalls':[], 'modelRequests':[]}

    def tool_response(self, arguments='{}'):
        return {'content':None, 'usage':USAGE, 'tool_calls':[{'id':'c1','type':'function','function':{'name':self.name,'arguments':arguments}}]}

    def test_result_goes_back_to_model_with_matching_id(self):
        ask = Mock(side_effect=[self.tool_response(), {'content':'7 stars','usage':USAGE}])
        answer = run_tools(ask, self.registry, self.payload, {}, self.tools, self.metadata, lambda p:None)
        self.assertEqual(answer, '7 stars')
        messages = ask.call_args_list[1].args[0]['messages']
        self.assertEqual(messages[-2]['tool_calls'][0]['id'], messages[-1]['tool_call_id'])
        self.assertEqual(json.loads(messages[-1]['content'])['structuredContent']['stars'], 7)
        self.assertEqual(len(self.metadata['modelRequests']), 2)
        self.assertEqual(self.metadata['toolCalls'][0]['name'], 'repo')
        self.assertEqual(len(self.payload['messages']), 1)

    def test_bad_arguments_return_tool_error_instead_of_breaking_pair(self):
        ask = Mock(side_effect=[self.tool_response('oops'), {'content':'bad arguments','usage':USAGE}])
        run_tools(ask,self.registry,self.payload,{},self.tools,self.metadata,lambda p:None)
        self.assertTrue(self.metadata['toolCalls'][0]['result']['isError'])
        self.assertEqual(ask.call_args_list[1].args[0]['messages'][-1]['tool_call_id'], 'c1')

    def test_limit_forces_final_answer_and_partial_requests_survive_failure(self):
        ask = Mock(side_effect=[self.tool_response(), self.tool_response(), RuntimeError('failed')])
        with self.assertRaises(RuntimeError):
            run_tools(ask,self.registry,self.payload,{},self.tools,self.metadata,lambda p:None)
        self.assertEqual(ask.call_args.kwargs['tool_choice'], 'none')
        self.assertEqual(len(self.metadata['modelRequests']), 3)
        self.assertEqual(self.metadata['modelRequests'][0]['usage'], USAGE)
        self.assertEqual(len(self.metadata['toolCalls']), 2)

    def test_context_check_runs_before_each_model_request(self):
        guard = Mock(side_effect=[None, ValueError('context')])
        ask = Mock(return_value=self.tool_response())
        with self.assertRaises(ValueError):
            run_tools(ask,self.registry,self.payload,{},self.tools,self.metadata,guard)
        self.assertEqual(ask.call_count,1)

    def test_all_parallel_calls_receive_results_but_only_three_execute(self):
        response=self.tool_response()
        response['tool_calls']=[{**copy.deepcopy(response['tool_calls'][0]),'id':f'c{i}'} for i in range(5)]
        self.registry._invoke=Mock(return_value={'content':[],'isError':False,'structuredContent':{}})
        ask=Mock(side_effect=[response,{'content':'done','usage':USAGE}])
        run_tools(ask,self.registry,self.payload,{},self.tools,self.metadata,lambda p:None)
        self.assertEqual(self.registry._invoke.call_count,3)
        self.assertEqual(len(self.metadata['toolCalls']),5)
        self.assertEqual(ask.call_args.kwargs['tool_choice'],'none')
        results=[m for m in ask.call_args.args[0]['messages'] if m['role']=='tool']
        self.assertEqual([m['tool_call_id'] for m in results],[f'c{i}' for i in range(5)])
        self.assertTrue(json.loads(results[-1]['content'])['isError'])

    def test_plain_response_does_not_call_mcp(self):
        self.registry._invoke=Mock()
        answer=run_tools(Mock(return_value={'content':'hello','usage':USAGE}),self.registry,self.payload,{},self.tools,self.metadata,lambda p:None)
        self.assertEqual(answer,'hello')
        self.registry._invoke.assert_not_called()

    def test_disconnected_server_returns_error_to_model(self):
        def first(payload,**options):
            self.registry.disconnect(self.registry.servers()[0]['id'])
            return self.tool_response()
        answers=iter([first,lambda *a,**k:{'content':'server disconnected','usage':USAGE}])
        run_tools(lambda *a,**k:next(answers)(*a,**k),self.registry,self.payload,{},self.tools,self.metadata,lambda p:None)
        self.assertTrue(self.metadata['toolCalls'][0]['result']['isError'])
