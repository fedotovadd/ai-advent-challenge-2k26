"""Exercise the real async handler with a controlled settings-save boundary."""
import json
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / 'static' / 'context-controls.js'

class ContextUiTests(unittest.TestCase):
    def test_switch_rechecks_pending_message_after_settings_save(self):
        source = SCRIPT.read_text()
        handler = source[source.index('    async function action('):source.index("    el('create-checkpoint').addEventListener")]
        harness = '''
        let resolveSave;
        const save = new Promise(resolve => { resolveSave = resolve; });
        const state = {activeId:'agent-1', pendingAgentIds:new Set(), drafts:{}, chatStatuses:{}, probeAttempts:{}};
        const agent = {id:'agent-1',context:{activeBranch:'main'}};
        const activeAgent = () => agent;
        let busy = false;
        const pending = () => busy || state.pendingAgentIds.has(state.activeId);
        const flushSettingsSave = () => save;
        let apiCalled = false;
        const api = async () => { apiCalled = true; return {response:{ok:true},body:{agent}}; };
        const refresh = () => {}, syncSendButton = () => {}, render = () => {}, replaceAgent = () => {}, setChatStatus = () => {};
        const AgentClientState = {agentExists:()=>true};
        const branchDrafts = {};
        ''' + handler + '''
        (async () => {
          const task = action('switch-branch', {branchId:'branch-1'});
          state.pendingAgentIds.add('agent-1');
          resolveSave(true);
          await task;
          console.log(JSON.stringify({apiCalled}));
        })();
        '''
        result = subprocess.run(['node', '-e', harness], capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)['apiCalled'])
