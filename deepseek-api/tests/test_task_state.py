import unittest

from task_state import (
    TaskStateError, apply_execution_markers, apply_task_command, default_task_state,
    extract_task_plan, parse_task_command, task_prompt_block, task_status, valid_task_state,
)


class TaskStateTests(unittest.TestCase):
    def test_parses_task_commands_and_rejects_invalid_arguments(self):
        self.assertEqual(parse_task_command(" /task  Лендинг "), {"action": "task", "title": "Лендинг"})
        for command in ("/execute", "/planning", "/pause", "/resume", "/status"):
            self.assertEqual(parse_task_command(command), {"action": command[1:]})
        self.assertIsNone(parse_task_command("Обычный текст"))
        for command in ("/task", "/task ", "/execute сейчас", "/pause позже"):
            with self.subTest(command=command), self.assertRaises(TaskStateError):
                parse_task_command(command)

    def test_transition_flow_pause_and_status(self):
        task, next_id, _ = apply_task_command(None, {"action": "task", "title": "Лендинг"}, 1)
        self.assertEqual((task["id"], next_id, task["stage"]), ("task-1", 2, "PLANNING"))
        task, _, _ = apply_task_command(task, {"action": "pause"}, next_id)
        self.assertTrue(task["paused"])
        with self.assertRaises(TaskStateError):
            apply_task_command(task, {"action": "execute"}, next_id)
        task, _, _ = apply_task_command(task, {"action": "resume"}, next_id)
        planned, _, _ = extract_task_plan("[[TASK_PLAN]]\n# План\n1. Исследовать\n2. Написать\n[[/TASK_PLAN]]", task)
        task = planned["task"]
        task, _, _ = apply_task_command(task, {"action": "execute"}, next_id)
        self.assertEqual((task["stage"], task["step"], task["current"]), ("EXECUTION", 1, "Исследовать"))
        task, _, _ = apply_task_command(task, {"action": "planning"}, next_id)
        self.assertEqual(task["stage"], "PLANNING")
        self.assertIn("Этап: PLANNING", task_status(task))

    def test_markers_advance_steps_and_complete_only_from_validation(self):
        task = default_task_state("task-1", "Лендинг")
        task = extract_task_plan("[[TASK_PLAN]]\n1. Первый\n2. Второй\n[[/TASK_PLAN]]", task)[0]["task"]
        task = apply_task_command(task, {"action": "execute"}, 2)[0]
        result = apply_execution_markers("Сделано\n[[TASK_STEP_DONE]]", task)
        self.assertEqual((result["task"]["stage"], result["task"]["step"], result["visible"]), ("EXECUTION", 2, "Сделано"))
        result = apply_execution_markers("Последний шаг\n[[TASK_STEP_DONE]]", result["task"])
        self.assertEqual(result["task"]["stage"], "VALIDATION")
        self.assertIn("EXECUTION", result["notice"])
        result = apply_execution_markers("Проверено\n[[TASK_TRANSITION:DONE]]", result["task"])
        self.assertEqual(result["task"]["stage"], "DONE")
        self.assertNotIn("[[", result["visible"])

    def test_marker_not_on_last_line_does_not_change_state(self):
        task = default_task_state("task-1", "Лендинг")
        task = extract_task_plan("[[TASK_PLAN]]\n1. Шаг\n[[/TASK_PLAN]]", task)[0]["task"]
        task = apply_task_command(task, {"action": "execute"}, 2)[0]
        result = apply_execution_markers("[[TASK_STEP_DONE]]\nНо работа продолжается", task)
        self.assertEqual(result["task"], task)
        self.assertIn("[[TASK_STEP_DONE]]", result["visible"])

    def test_revised_plan_keeps_completed_prefix_and_prompt_has_current_state(self):
        task = default_task_state("task-1", "Лендинг")
        task = extract_task_plan("[[TASK_PLAN]]\n1. Первый\n2. Второй\n[[/TASK_PLAN]]", task)[0]["task"]
        task = apply_task_command(task, {"action": "execute"}, 2)[0]
        task = apply_execution_markers("ok\n[[TASK_STEP_DONE]]", task)["task"]
        task = apply_task_command(task, {"action": "planning"}, 2)[0]
        task = extract_task_plan("[[TASK_PLAN]]\n1. Первый\n2. Уточнённый\n[[/TASK_PLAN]]", task)[0]["task"]
        self.assertEqual((task["step"], task["current"], task["done"]), (2, "Уточнённый", ["Первый"]))
        self.assertIn("[TASK_STATE]", task_prompt_block(task, "# План"))
        with self.assertRaises(TaskStateError):
            extract_task_plan("[[TASK_PLAN]]\n1. Другой\n[[/TASK_PLAN]]", task)

    def test_validation_rejects_inconsistent_data(self):
        self.assertTrue(valid_task_state(None))
        task = default_task_state("task-1", "Лендинг")
        self.assertTrue(valid_task_state(task))
        task["stage"] = "BAD"
        self.assertFalse(valid_task_state(task))


if __name__ == "__main__":
    unittest.main()
