import concurrent.futures

from agent import MODEL, SYSTEM_PROMPT, _response_content_and_usage

DIRECT_SYSTEM = (
    "Реши задачу самостоятельно. Дай точный, краткий и проверяемый ответ на русском языке. "
    "Последней строкой напиши «Ответ: ...»."
)
STEPWISE_SYSTEM = " ".join([
    "Решай задачу строго пошагово.",
    "Сначала перечисли, что дано и что требуется найти.",
    "Затем разбей решение на пронумерованные шаги и после каждого шага выписывай "
    "промежуточный результат, а не только рассуждение.",
    "Перед финалом проверь решение подстановкой или встречной прикидкой.",
    "Последней строкой ответа напиши «Ответ: ...» — одну строку с итогом и без пояснений.",
])
PROMPT_ENGINEER_SYSTEM = " ".join([
    "Ты — инженер промптов. Тебе дают задачу пользователя.",
    "Не решай её. Твоя работа — написать промпт, по которому языковая модель решит эту задачу "
    "максимально точно.",
    "В промпте укажи подходящую роль исполнителя, метод решения, порядок шагов, "
    "типичные ошибки этого класса задач, способ самопроверки и формат ответа.",
    "Верни только текст промпта, без вступлений, комментариев и кавычек вокруг него.",
])
PROMPT_EXECUTOR_SYSTEM = (
    "Ты исполнитель решения. Следуй инструкции пользователя как промпту для решения исходной задачи. "
    "Реши задачу точно и последней строкой напиши «Ответ: ...»."
)
ANALYST_SYSTEM = " ".join([
    "Ты — аналитик. Начни с разбора условия: выпиши все данные, явные и неявные допущения, "
    "что именно требуется найти и чего в условии не хватает.",
    "Только после разбора дай своё решение и итоговый ответ.",
    "Пиши сжато, без воды. Последней строкой — «Ответ: ...».",
])
ENGINEER_SYSTEM = " ".join([
    "Ты — инженер. Тебя интересует работающая процедура, а не рассуждения вокруг задачи.",
    "Дай конкретный алгоритм или расчёт и доведи его до числа, формулы или готового ответа.",
    "Если задача вычислительная — покажи вычисления. Если алгоритмическая — покажи алгоритм "
    "и его сложность. Пиши сжато. Последней строкой — «Ответ: ...».",
])
CRITIC_SYSTEM = " ".join([
    "Ты — критик и скептик. Сначала перечисли ловушки этой задачи и типичные ошибки, "
    "на которых обычно спотыкаются при её решении: неверная трактовка условия, "
    "подмена вопроса, ошибки в арифметике, забытые граничные случаи.",
    "Затем реши задачу сам, обходя перечисленные ловушки.",
    "Пиши сжато. Последней строкой — «Ответ: ...».",
])
MODERATOR_SYSTEM = " ".join([
    "Ты — модератор экспертного совета. Ниже даны результаты четырёх независимых способов "
    "решения одной и той же задачи.",
    "Сопоставь их: явно назови, в чём они сходятся и в чём расходятся, "
    "и если расходятся — реши, кто прав, и объясни почему.",
    "Оцени ответы по заданным критериям и заверши разбор итоговым ответом.",
    "Последней строкой — «Ответ: ...» без пояснений.",
])
EXPERT_ROLES = (
    ("analyst", "Аналитик", ANALYST_SYSTEM),
    ("engineer", "Инженер", ENGINEER_SYSTEM),
    ("critic", "Критик", CRITIC_SYSTEM),
)
DAY_THREE_TASKS = {
    "server-messages": {
        "title": "Три сервера",
        "task": (
            "В системе три сервера: A, B и C. Ровно один неисправен. Сообщения: A — «B неисправен»; "
            "B — «A и C находятся в одинаковом состоянии»; C — «A исправен». "
            "Ровно одно сообщение правдиво. Какой сервер неисправен? Докажите вывод."
        ),
        "referenceSolution": (
            "Неисправен C. Если неисправен A, правдивых сообщений нет; если B — правдивы все три; "
            "если C — правдиво только сообщение C."
        ),
        "criteria": (
            "верно определён неисправный сервер",
            "учтено условие о ровно одном правдивом сообщении",
            "перебором исключены остальные варианты",
        ),
    },
    "channel-analysis": {
        "title": "Выбор канала",
        "task": (
            "Канал A получил 1 000 показов и 30 регистраций. Канал B получил 200 показов и 10 регистраций. "
            "Какой канал эффективнее по конверсии? Если дополнительно купить 100 показов и предположить, "
            "что конверсия не изменится, сколько регистраций ожидается от каждого канала?"
        ),
        "referenceSolution": (
            "Конверсия A — 3 %, B — 5 %. На 100 дополнительных показов ожидается 3 регистрации от A "
            "и 5 от B; по конверсии эффективнее B."
        ),
        "criteria": (
            "корректно рассчитаны конверсии обоих каналов",
            "правильно рассчитан прогноз на 100 показов",
            "вывод соответствует рассчитанной конверсии",
        ),
    },
}
MAX_DAY_THREE_REQUEST_BYTES = 32_768

def _run_day_three_call(ask_model, user_prompt, system_prompt, title="Вызов API"):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    answer, _ = _response_content_and_usage(ask_model(payload))
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("empty API response")
    return {
        "title": title,
        "systemPrompt": system_prompt,
        "userPrompt": user_prompt,
        "answer": answer,
    }


def _experts_answer(calls):
    return "\n\n".join(f"{call['title']}:\n{call['answer']}" for call in calls)


def _method_answer(method):
    return method.get("answer", method["calls"][-1]["answer"])


def _comparison_prompt(task_config, methods):
    criteria = "\n".join(f"- {criterion}" for criterion in task_config["criteria"])
    answers = "\n\n".join(
        f"{method['title']}:\n{_method_answer(method)}" for method in methods
    )
    return (
        "Сопоставь результаты четырёх способов решения задачи.\n\n"
        f"Задача:\n{task_config['task']}\n\n"
        f"Эталонное решение:\n{task_config['referenceSolution']}\n\n"
        f"Критерии:\n{criteria}\n\n"
        f"Ответы способов:\n{answers}"
    )


def _run_experts(ask_model, task, parallel=False):
    if not parallel:
        return [
            _run_day_three_call(ask_model, task, system_prompt, title)
            for _, title, system_prompt in EXPERT_ROLES
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(EXPERT_ROLES)) as executor:
        pending = {
            role_id: executor.submit(_run_day_three_call, ask_model, task, system_prompt, title)
            for role_id, title, system_prompt in EXPERT_ROLES
        }
        results = {role_id: future.result() for role_id, future in pending.items()}
    return [results[role_id] for role_id, _, _ in EXPERT_ROLES]


def run_day_three_experiment(ask_model, task_id):
    task_config = DAY_THREE_TASKS[task_id]
    task = task_config["task"]
    direct_call = _run_day_three_call(ask_model, task, DIRECT_SYSTEM, "Напрямую")
    step_by_step_call = _run_day_three_call(ask_model, task, STEPWISE_SYSTEM, "Пошагово")
    generated_prompt_call = _run_day_three_call(
        ask_model,
        task,
        PROMPT_ENGINEER_SYSTEM,
        "Составить промпт",
    )
    generated_solution_call = _run_day_three_call(
        ask_model,
        generated_prompt_call["answer"],
        PROMPT_EXECUTOR_SYSTEM,
        "Решить по своему промпту",
    )
    experts_calls = _run_experts(ask_model, task)
    methods = [
        {
            "id": "direct",
            "title": "Напрямую",
            "description": "Только текст задачи без дополнительных инструкций.",
            "calls": [direct_call],
        },
        {
            "id": "step_by_step",
            "title": "Пошагово",
            "description": "Задача с явной просьбой решать пошагово.",
            "calls": [step_by_step_call],
        },
        {
            "id": "generated_prompt",
            "title": "Свой промпт",
            "description": "Модель сначала создаёт промпт, а затем решает задачу по нему.",
            "calls": [generated_prompt_call, generated_solution_call],
        },
        {
            "id": "experts",
            "title": "Группа экспертов",
            "description": "Независимые решения аналитика, инженера и критика.",
            "calls": experts_calls,
            "answer": _experts_answer(experts_calls),
        },
    ]
    try:
        comparison = {"status": "success", "call": _run_day_three_call(
            ask_model,
            _comparison_prompt(task_config, methods),
            MODERATOR_SYSTEM,
            "Модератор",
        )}
    except Exception:
        comparison = {"status": "error", "message": "Не удалось получить сравнение DeepSeek."}
    return {
        "taskId": task_id,
        "title": task_config["title"],
        "task": task,
        "referenceSolution": task_config["referenceSolution"],
        "criteria": task_config["criteria"],
        "systemPrompt": SYSTEM_PROMPT,
        "methods": methods,
        "comparison": comparison,
    }


def _run_day_three_method(ask_model, task, method_id):
    if method_id == "direct":
        calls = [_run_day_three_call(ask_model, task, DIRECT_SYSTEM, "Напрямую")]
        title, description = "Напрямую", "Только текст задачи без дополнительных инструкций."
    elif method_id == "step_by_step":
        calls = [_run_day_three_call(ask_model, task, STEPWISE_SYSTEM, "Пошагово")]
        title, description = "Пошагово", "Задача с явной просьбой решать пошагово."
    elif method_id == "generated_prompt":
        prompt_call = _run_day_three_call(
            ask_model,
            task,
            PROMPT_ENGINEER_SYSTEM,
            "Составить промпт",
        )
        calls = [prompt_call, _run_day_three_call(
            ask_model,
            prompt_call["answer"],
            PROMPT_EXECUTOR_SYSTEM,
            "Решить по своему промпту",
        )]
        title, description = "Свой промпт", "Модель сначала создаёт промпт, а затем решает задачу по нему."
    else:
        calls = _run_experts(ask_model, task, parallel=True)
        title, description = "Группа экспертов", "Независимые решения аналитика, инженера и критика."
    method = {"id": method_id, "title": title, "description": description, "calls": calls}
    if method_id == "experts":
        method["answer"] = _experts_answer(calls)
    return method


def stream_day_three_experiment(ask_model, task_id):
    task_config = DAY_THREE_TASKS[task_id]
    yield {"type": "start", "experiment": {
        "taskId": task_id, "title": task_config["title"], "task": task_config["task"],
        "referenceSolution": task_config["referenceSolution"], "criteria": task_config["criteria"],
    }}
    method_ids = ("direct", "step_by_step", "generated_prompt", "experts")
    methods_by_id = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        pending = {
            executor.submit(_run_day_three_method, ask_model, task_config["task"], method_id): method_id
            for method_id in method_ids
        }
        for future in concurrent.futures.as_completed(pending):
            try:
                method = future.result()
            except Exception:
                yield {"type": "error", "message": "Не удалось получить один из ответов DeepSeek."}
                return
            methods_by_id[method["id"]] = method
            yield {"type": "method", "method": method}
    methods = [methods_by_id[method_id] for method_id in method_ids]
    try:
        comparison = {"status": "success", "call": _run_day_three_call(
            ask_model,
            _comparison_prompt(task_config, methods),
            MODERATOR_SYSTEM,
            "Модератор",
        )}
    except Exception:
        comparison = {"status": "error", "message": "Не удалось получить сравнение DeepSeek."}
    yield {"type": "comparison", "comparison": comparison}
    yield {"type": "complete"}
