import json
import subprocess
import unittest
from pathlib import Path


STATE_SCRIPT = Path(__file__).parents[1] / "static" / "agent-state.js"


class AgentClientStateTests(unittest.TestCase):
    def run_state_script(self, expression):
        program = (
            f"const state=require({json.dumps(str(STATE_SCRIPT))});"
            f"console.log(JSON.stringify({expression}));"
        )
        result = subprocess.run(
            ["node", "-e", program], check=True, text=True, capture_output=True,
        )
        return json.loads(result.stdout)

    def test_remove_agent_state_chooses_next_agent_and_clears_its_local_data(self):
        result = self.run_state_script("state.removeAgentState({"
            "agents:[{id:'agent-1'},{id:'agent-2'},{id:'agent-3'}],"
            "activeId:'agent-2',drafts:{'agent-1':'first','agent-2':'remove'},"
            "pendingAgentIds:['agent-2'],chatStatuses:{'agent-2':{text:'wait'}},"
            "pendingSettings:{agentId:'agent-2'},metadata:{agentId:'agent-2'},"
            "metadataAgentId:'agent-2'},'agent-2')")

        self.assertEqual([agent["id"] for agent in result["agents"]], ["agent-1", "agent-3"])
        self.assertEqual(result["activeId"], "agent-3")
        self.assertEqual(result["drafts"], {"agent-1": "first"})
        self.assertEqual(result["pendingAgentIds"], [])
        self.assertEqual(result["chatStatuses"], {})
        self.assertIsNone(result["pendingSettings"])
        self.assertIsNone(result["metadata"])
        self.assertIsNone(result["metadataAgentId"])

    def test_remove_agent_state_chooses_previous_agent_or_empty_state(self):
        selected_last = self.run_state_script("state.removeAgentState({"
            "agents:[{id:'agent-1'},{id:'agent-2'}],activeId:'agent-2',drafts:{},"
            "pendingAgentIds:[],chatStatuses:{}},'agent-2')")
        empty = self.run_state_script("state.removeAgentState({"
            "agents:[{id:'agent-1'}],activeId:'agent-1',drafts:{},"
            "pendingAgentIds:[],chatStatuses:{}},'agent-1')")

        self.assertEqual(selected_last["activeId"], "agent-1")
        self.assertEqual(empty["agents"], [])
        self.assertIsNone(empty["activeId"])

    def test_agent_exists_rejects_late_response_for_deleted_agent(self):
        self.assertTrue(self.run_state_script("state.agentExists({agents:[{id:'agent-1'}]},'agent-1')"))
        self.assertFalse(self.run_state_script("state.agentExists({agents:[{id:'agent-1'}]},'agent-2')"))


if __name__ == "__main__":
    unittest.main()
