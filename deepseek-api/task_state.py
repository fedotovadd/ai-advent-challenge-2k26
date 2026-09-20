"""Pure contracts for the persisted task state machine."""
import copy
import re


TASK_STAGES = ("PLANNING", "EXECUTION", "VALIDATION", "DONE")
TASK_TRANSITIONS = {
    "PLANNING": ("EXECUTION",),
    "EXECUTION": ("PLANNING", "VALIDATION"),
    "VALIDATION": ("EXECUTION", "DONE"),
    "DONE": (),
}
TASK_PLAN_OPEN = "[[TASK_PLAN]]"
TASK_PLAN_CLOSE = "[[/TASK_PLAN]]"
TASK_STEP_DONE = "[[TASK_STEP_DONE]]"
TASK_DONE = "[[TASK_TRANSITION:DONE]]"
TASK_VALIDATION_PASSED = "[[TASK_VALIDATION_PASSED]]"
TASK_VALIDATION_FAILED_PREFIX = "[[TASK_VALIDATION_FAILED:"
TASK_MARKER_PATTERN = re.compile(
    r"\[\[TASK_STEP_DONE\]\]|\[\[TASK_TRANSITION:DONE\]\]|"
    r"\[\[TASK_VALIDATION_PASSED\]\]|\[\[TASK_VALIDATION_FAILED:[^\]\r\n]*\]\]"
)
MAX_TITLE_LENGTH = 300
MAX_STEPS = 24
MAX_STEP_LENGTH = 500


class TaskStateError(ValueError):
    """Raised for malformed task input or an invalid state transition."""


def _text(value, maximum):
    return isinstance(value, str) and bool(value.strip()) and len(value.strip()) <= maximum


def default_task_state(task_id, title):
    if not _text(task_id, 80) or not _text(title, MAX_TITLE_LENGTH):
        raise TaskStateError("Название задачи должно быть непустой строкой до 300 символов.")
    return {
        "id": task_id.strip(), "title": title.strip(), "stage": "PLANNING", "paused": False,
        "step": 0, "total": 0, "plan": [], "done": [], "current": "Собрать требования",
        "expectedAction": "Продолжайте отвечать на вопросы агента.", "planPath": None,
    }


def valid_task_state(value):
    if value is None:
        return True
    expected = {"id", "title", "stage", "paused", "step", "total", "plan", "done", "current", "expectedAction", "planPath"}
    if not isinstance(value, dict) or set(value) != expected:
        return False
    if not (_text(value["id"], 80) and _text(value["title"], MAX_TITLE_LENGTH) and value["stage"] in TASK_STAGES
            and isinstance(value["paused"], bool) and isinstance(value["step"], int) and not isinstance(value["step"], bool)
            and isinstance(value["total"], int) and not isinstance(value["total"], bool)
            and isinstance(value["plan"], list) and isinstance(value["done"], list)
            and _text(value["current"], MAX_STEP_LENGTH) and _text(value["expectedAction"], MAX_STEP_LENGTH)
            and (value["planPath"] is None or _text(value["planPath"], 200))):
        return False
    if not (0 <= value["total"] <= MAX_STEPS and 0 <= value["step"] <= value["total"] and len(value["plan"]) == value["total"]
            and all(_text(item, MAX_STEP_LENGTH) for item in value["plan"])
            and all(_text(item, MAX_STEP_LENGTH) for item in value["done"]) and value["done"] == value["plan"][:len(value["done"])]):
        return False
    if value["stage"] == "PLANNING":
        return not value["paused"] or value["stage"] != "DONE"
    if value["stage"] == "EXECUTION":
        return value["total"] > 0 and 1 <= value["step"] <= value["total"] and value["current"] == value["plan"][value["step"] - 1]
    if value["stage"] == "VALIDATION":
        return value["total"] > 0 and value["step"] == value["total"] and value["done"] == value["plan"] and value["current"] == "Проверить результат"
    return value["total"] > 0 and value["done"] == value["plan"] and not value["paused"]


def parse_task_command(text):
    if not isinstance(text, str):
        return None
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return None
    command, argument = parts[0], parts[1].strip() if len(parts) == 2 else ""
    if command == "/task":
        if not _text(argument, MAX_TITLE_LENGTH):
            raise TaskStateError("После /task укажите название задачи до 300 символов.")
        return {"action": "task", "title": argument}
    names = {"/execute": "execute", "/planning": "planning", "/pause": "pause", "/resume": "resume", "/status": "status"}
    if command not in names:
        return None
    if argument:
        raise TaskStateError(f"Команда {command} не принимает дополнительный текст.")
    return {"action": names[command]}


def _require_task(task):
    if task is None:
        raise TaskStateError("Сначала создайте задачу командой /task Название.")


def _transition(task, target):
    if target not in TASK_TRANSITIONS[task["stage"]]:
        if task["stage"] == "DONE":
            raise TaskStateError("Задача уже завершена. Создайте новую через /task Название.")
        raise TaskStateError(f"Переход из {task['stage']} в {target} не разрешён.")
    candidate = copy.deepcopy(task)
    candidate["stage"] = target
    return candidate


def apply_task_command(task, command, next_task_id):
    action = command.get("action") if isinstance(command, dict) else None
    if not isinstance(next_task_id, int) or isinstance(next_task_id, bool) or next_task_id < 1:
        raise TaskStateError("Счётчик задач имеет неверный формат.")
    if action == "task":
        if task is not None and task["stage"] != "DONE":
            raise TaskStateError("Сначала завершите текущую задачу.")
        created = default_task_state(f"task-{next_task_id}", command.get("title"))
        return created, next_task_id + 1, f"Создана задача: {created['title']}."
    _require_task(task)
    if task["paused"] and action not in {"resume", "status"}:
        raise TaskStateError("Задача на паузе. Используйте /resume или /status.")
    candidate = copy.deepcopy(task)
    if action == "status":
        return candidate, next_task_id, task_status(candidate)
    if action == "pause":
        if candidate["stage"] not in {"PLANNING", "EXECUTION", "VALIDATION"}:
            raise TaskStateError("Завершённую задачу нельзя поставить на паузу.")
        candidate["paused"] = True
        return candidate, next_task_id, "Задача поставлена на паузу."
    if action == "resume":
        if not candidate["paused"]:
            raise TaskStateError("Задача не находится на паузе.")
        candidate["paused"] = False
        return candidate, next_task_id, "Задача продолжена с сохранённого шага."
    if action == "execute":
        if candidate["stage"] != "PLANNING" or not candidate["plan"] or candidate["planPath"] is None:
            raise TaskStateError("Сначала получите и прочитайте Markdown-план задачи.")
        candidate = _transition(candidate, "EXECUTION")
        candidate["step"] = len(candidate["done"]) + 1
        candidate["current"] = candidate["plan"][candidate["step"] - 1]
        candidate["expectedAction"] = "Выполнить текущий шаг плана."
        return candidate, next_task_id, "Завершён этап PLANNING. Начат этап EXECUTION."
    if action == "planning":
        if candidate["stage"] != "EXECUTION":
            raise TaskStateError("В планирование можно вернуться только из выполнения.")
        candidate = _transition(candidate, "PLANNING")
        candidate["expectedAction"] = "Уточните требования; агент подготовит обновлённый план."
        return candidate, next_task_id, "Завершён этап EXECUTION. Начат этап PLANNING."
    raise TaskStateError("Неизвестная команда состояния задачи.")


def _plan_steps(markdown):
    lines = markdown.splitlines()
    steps_heading = None
    for index, line in enumerate(lines):
        match = re.fullmatch(r"\s*(#{1,6})\s*шаги\s*#*\s*", line, re.IGNORECASE)
        if match:
            steps_heading = (index, len(match.group(1)))
            break
    if steps_heading is not None:
        start, level = steps_heading
        selected = []
        for line in lines[start + 1:]:
            heading = re.fullmatch(r"\s*(#{1,6})\s+.*", line)
            if heading and len(heading.group(1)) <= level:
                break
            selected.append(line)
        lines = selected
    steps = []
    for line in lines:
        match = re.fullmatch(r"\s*(?:(\d+)\.\s+|- \[[ xX]\]\s+)(.+?)\s*", line)
        if match:
            number, text = match.groups()
            if (number is not None and int(number) != len(steps) + 1) or not _text(text, MAX_STEP_LENGTH):
                raise TaskStateError("Шаги плана должны быть последовательным нумерованным списком.")
            steps.append(text.strip())
    if not 1 <= len(steps) <= MAX_STEPS:
        raise TaskStateError("План должен содержать от 1 до 24 нумерованных шагов.")
    return steps


def task_plan_markdown(task):
    """Render the canonical plan from state, never trusting stale plan-file markup."""
    _require_task(task)
    if not task["plan"]:
        return ""
    completed = len(task["done"])
    lines = ["# План"]
    lines.extend(f"- [{'x' if index < completed else ' '}] {step}" for index, step in enumerate(task["plan"]))
    return "\n".join(lines)


def extract_task_plan(answer, task):
    _require_task(task)
    if task["stage"] != "PLANNING":
        return {"task": copy.deepcopy(task), "markdown": None}, answer, []
    if not isinstance(answer, str) or answer.count(TASK_PLAN_OPEN) != 1 or answer.count(TASK_PLAN_CLOSE) != 1:
        return {"task": copy.deepcopy(task), "markdown": None}, answer, []
    start, end = answer.index(TASK_PLAN_OPEN), answer.index(TASK_PLAN_CLOSE)
    if end <= start:
        raise TaskStateError("Контейнер Markdown-плана имеет неверный формат.")
    markdown = answer[start + len(TASK_PLAN_OPEN):end].strip()
    steps = _plan_steps(markdown)
    previous_done = task["done"]
    if previous_done and (steps[:len(previous_done)] != previous_done or len(steps) == len(previous_done)):
        raise TaskStateError("Обновлённый план должен сохранить выполненные шаги.")
    candidate = copy.deepcopy(task)
    candidate.update({"plan": steps, "total": len(steps), "planPath": f"{task['id']}.md"})
    if previous_done:
        candidate["step"] = len(previous_done) + 1
        candidate["current"] = steps[candidate["step"] - 1]
    else:
        candidate["step"] = 0
        candidate["current"] = "План готов к просмотру"
    candidate["expectedAction"] = "Прочитайте план и используйте /execute для запуска выполнения."
    rendered = task_plan_markdown(candidate)
    return {"task": candidate, "markdown": rendered}, rendered, steps


def _last_nonempty_line(answer):
    lines = answer.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip():
            return index, lines[index].strip()
    return None, None


def _visible_without_markers(answer):
    lines = TASK_MARKER_PATTERN.sub("", answer).splitlines()
    return "\n".join(line for line in lines if line.strip()).strip()


def _marker_result(task, visible, notice, continue_task=False, accepted=False):
    return {"task": task, "visible": visible, "notice": notice,
            "continueTask": continue_task, "accepted": accepted}


def apply_execution_markers(answer, task):
    _require_task(task)
    candidate = copy.deepcopy(task)
    matches = list(TASK_MARKER_PATTERN.finditer(answer))
    if not matches:
        return _marker_result(candidate, answer, None)
    visible = _visible_without_markers(answer)
    index, marker = _last_nonempty_line(answer)
    if len(matches) != 1 or marker != matches[0].group(0):
        return _marker_result(candidate, visible,
                              "Текущий этап не позволяет выполнить служебный переход: метка должна быть единственной последней строкой.")
    if marker == TASK_STEP_DONE and candidate["stage"] == "EXECUTION":
        candidate["done"].append(candidate["current"])
        if candidate["step"] == candidate["total"]:
            candidate = _transition(candidate, "VALIDATION")
            candidate.update({"current": "Проверить результат", "expectedAction": "Агент проверяет результат."})
            notice = "Завершён этап EXECUTION. Начат этап VALIDATION."
        else:
            candidate["step"] += 1
            candidate["current"] = candidate["plan"][candidate["step"] - 1]
            notice = f"Завершён шаг {candidate['step'] - 1}. Начат шаг {candidate['step']}."
        return _marker_result(candidate, visible, notice,
                              not candidate["paused"] and candidate["stage"] in {"EXECUTION", "VALIDATION"}, True)
    if marker == TASK_DONE:
        return _marker_result(candidate, visible, "Сначала завершите проверку результата.")
    if marker == TASK_VALIDATION_PASSED and candidate["stage"] == "VALIDATION":
        candidate = _transition(candidate, "DONE")
        candidate.update({"paused": False, "current": "Задача завершена", "expectedAction": "Создайте новую задачу командой /task Название."})
        return _marker_result(candidate, visible, "Завершён этап VALIDATION. Задача завершена.", False, True)
    failure = re.fullmatch(r"\[\[TASK_VALIDATION_FAILED:(\d+)\]\]", marker or "")
    if failure and candidate["stage"] == "VALIDATION":
        step = int(failure.group(1))
        if 1 <= step <= candidate["total"]:
            candidate = _transition(candidate, "EXECUTION")
            candidate["done"] = candidate["plan"][:step - 1]
            candidate["step"] = step
            candidate["current"] = candidate["plan"][step - 1]
            candidate["expectedAction"] = "Выполнить текущий шаг плана."
            return _marker_result(candidate, visible, f"Проверка вернула задачу к шагу {step}.",
                                  not candidate["paused"], True)
    if marker.startswith(TASK_VALIDATION_FAILED_PREFIX):
        return _marker_result(candidate, visible,
                              "Нельзя вернуть задачу на доработку: укажите существующий шаг после начала валидации.")
    return _marker_result(candidate, visible,
                          "Текущий этап не позволяет выполнить служебный переход. Следуйте ожидаемому действию задачи.")


def task_prompt_block(task, plan_markdown):
    if task is None:
        return ""
    done = "; ".join(task["done"]) or "—"
    plan = task_plan_markdown(task) or "План ещё не сформирован."
    return ("[TASK_STATE]\n"
            f"Задача: {task['title']}\nЭтап: {task['stage']} ({TASK_STAGES.index(task['stage']) + 1}/4)\nПрогресс шагов: {task['step']}/{task['total']}\n"
            f"Текущий шаг: {task['current']}\nВыполнено: {done}\nОжидаемое действие: {task['expectedAction']}\n"
            f"План:\n{plan}\n\n"
            "Не пропускай этапы. В PLANNING собери требования и при готовности верни Markdown в [[TASK_PLAN]]...[[/TASK_PLAN]]. "
            "В EXECUTION ставь [[TASK_STEP_DONE]] только последней непустой строкой после завершения текущего шага. "
            "В VALIDATION ставь [[TASK_VALIDATION_PASSED]] только последней непустой строкой после успешной проверки, "
            "или [[TASK_VALIDATION_FAILED:<номер шага>]] для возврата на доработку.")


def task_status(task):
    _require_task(task)
    done = "; ".join(task["done"]) or "—"
    pause = " · на паузе" if task["paused"] else ""
    return (f"Задача: {task['title']}\nЭтап: {task['stage']} ({TASK_STAGES.index(task['stage']) + 1}/4){pause}\nПрогресс шагов: {task['step']}/{task['total']}\n"
            f"Текущий шаг: {task['current']}\nВыполнено: {done}\nОжидаемое действие: {task['expectedAction']}")
