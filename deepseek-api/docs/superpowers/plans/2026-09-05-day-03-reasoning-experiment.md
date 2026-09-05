# День 3: четыре способа рассуждения Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить в локальный веб-чат переключаемую лабораторию дня 3, которая через API решает задачу о трёх выключателях четырьмя методами и даёт прозрачное сравнение результатов.

**Architecture:** Сервер в `main.py` получит изолированный маршрут `POST /api/day-03/run` и чистую функцию сборки эксперимента, использующую уже внедряемый `ask_model`. Страница останется единым HTML/CSS/JS-документом: переключатель представлений скроет обычный чат или лабораторию, а клиент отрисует безопасными DOM-операциями четыре карточки из ответа сервера. Эксперимент не пишет в хранилище сессий и не использует настройки активного чата.

**Tech Stack:** Python 3, `http.server`, встроенные HTML/CSS/JavaScript, `unittest`, OpenAI-совместимый DeepSeek API.

---

## File structure

- Modify: `deepseek-api/main.py` — константы задания, функция четырёх API-стратегий, защищённый маршрут, переключатель представлений и UI лаборатории.
- Modify: `deepseek-api/tests/test_main.py` — HTTP-контракты эксперимента, ошибки, последовательность payload и контракт разметки/клиентской логики.
- Modify: `deepseek-api/README.md` — инструкция выполнения задания третьего дня и критерии сравнения.

### Task 1: Зафиксировать серверный контракт эксперимента тестами

**Files:**
- Modify: `deepseek-api/tests/test_main.py:11-47, 321-371`

- [ ] **Step 1: Добавить управляемую очередь ответов к подставной модели**

В `setUp` добавить `self.answers = []`. В `ask_model` после записи вызова возвращать `self.answers.pop(0)`, если очередь непуста, иначе оставить существующий `self.answer`. Это позволит проверить пять вызовов API без обращения к сети.

- [ ] **Step 2: Написать падающий тест успешного запуска четырёх методов**

Добавить `test_day_three_run_returns_four_methods_and_uses_generated_prompt`. До запроса установить:

```python
self.answers = [
    "Прямое решение",
    "Пошаговое решение",
    "Решай задачу о трёх выключателях точно и кратко.",
    "Решение по своему промпту",
    "Аналитик: ...\nИнженер: ...\nКритик: ...",
]
status, body, _ = self.json_request("POST", "/api/day-03/run")
```

Проверить статус 200, наличие `task`, `referenceSolution`, `systemPrompt`, четыре результата с идентификаторами `direct`, `step_by_step`, `generated_prompt`, `experts` и количеством `calls` `[1, 1, 2, 1]`. Проверить, что третий элемент второго вызова содержит в `userPrompt` буквально третий ответ подставной модели, а его `answer` равен четвёртому. Проверить пять записанных payload: у каждого `model == main.MODEL`, первый пользовательский текст равен `main.DAY_THREE_TASK`, второй содержит «Решай пошагово», третий содержит задачу и просьбу составить промпт, четвёртый использует сгенерированный текст, пятый содержит роли «аналитик», «инженер» и «критик».

- [ ] **Step 3: Написать падающие тесты ошибок маршрута**

Добавить:

```python
def test_day_three_rejects_nonempty_request_body(self):
    status, body, _ = self.json_request("POST", "/api/day-03/run", {"task": "другая"})
    self.assertEqual(status, 400)
    self.assertEqual(body, {"error": "Маршрут не принимает параметры."})

def test_day_three_provider_error_returns_no_partial_experiment(self):
    self.server.ask_model = lambda payload: (_ for _ in ()).throw(RuntimeError("network"))
    status, body, _ = self.json_request("POST", "/api/day-03/run")
    self.assertEqual(status, 502)
    self.assertEqual(body, {"error": "Не удалось получить ответы DeepSeek."})

def test_day_three_missing_key_returns_503(self):
    with patch.dict(os.environ, {}, clear=True):
        status, body, _ = self.json_request("POST", "/api/day-03/run")
    self.assertEqual(status, 503)
    self.assertEqual(body, {"error": "Не задан DEEPSEEK_API_KEY."})
```

Расширить существующий тест чужого `Origin` парой `("/api/day-03/run", None)` и проверить, что подставная модель не вызывалась.

- [ ] **Step 4: Запустить тесты и убедиться в ожидаемом падении**

Run: `python3 -m unittest discover -s tests -v`

Expected: новые тесты падают из-за отсутствия `DAY_THREE_TASK` и маршрута `/api/day-03/run`; существующие тесты остаются зелёными.

- [ ] **Step 5: Сделать отдельный коммит RED-фазы**

```bash
git add deepseek-api/tests/test_main.py
git commit -m "test: define day three experiment contract"
```

### Task 2: Реализовать API четырёх стратегий

**Files:**
- Modify: `deepseek-api/main.py:11-19, 139-177, 283-308`
- Test: `deepseek-api/tests/test_main.py`

- [ ] **Step 1: Добавить неизменяемые данные и помощник одного вызова**

После `DEFAULT_SETTINGS` объявить `DAY_THREE_TASK` с выбранной русской задачей и `DAY_THREE_REFERENCE_SOLUTION` с полной последовательностью «первый включить и выключить; второй включить; горит — второй, тёплая — первый, холодная — третий». Добавить `run_day_three_experiment(ask_model)` и внутренний помощник, который строит payload:

```python
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
}
answer = ask_model(payload)
if not isinstance(answer, str) or not answer.strip():
    raise ValueError("empty API response")
return {"systemPrompt": SYSTEM_PROMPT, "userPrompt": user_prompt, "answer": answer}
```

В `run_day_three_experiment` создать результаты в порядке `direct`, `step_by_step`, `generated_prompt`, `experts`. Для третьего сначала вызвать модель с запросом составить русский промпт для решения `DAY_THREE_TASK`, затем передать её непустой ответ второму вызову. Вернуть словарь `task`, `referenceSolution`, `systemPrompt` и `methods`; у каждого метода должны быть `id`, `title`, `description`, `calls`. Не добавлять эти вызовы в `SessionStore` и не передавать параметры настроек чата.

- [ ] **Step 2: Добавить обработчик маршрута и сохранение семантики ошибок**

В `do_POST` до разбора путей сессий обработать строгое совпадение `"/api/day-03/run"`. Реализовать `_handle_day_three`: если `Content-Length` больше нуля, прочитать JSON только для проверки и ответить `400 {"error": "Маршрут не принимает параметры."}`; при отсутствии ключа ответить `503 {"error": "Не задан DEEPSEEK_API_KEY."}`; при любой ошибке модели или пустом ответе ответить `502 {"error": "Не удалось получить ответы DeepSeek."}`. При успехе отправить `200 {"experiment": run_day_three_experiment(self.server.ask_model)}`. Не возвращать результаты, собранные до исключения.

- [ ] **Step 3: Запустить тесты и убедиться в GREEN-фазе**

Run: `python3 -m unittest discover -s tests -v`

Expected: весь набор проходит; в том числе новый контракт выполняет пять предсказуемых локальных вызовов подставной модели.

- [ ] **Step 4: Сделать коммит реализации API**

```bash
git add deepseek-api/main.py deepseek-api/tests/test_main.py
git commit -m "feat: add day three reasoning experiment API"
```

### Task 3: Зафиксировать контракт переключателя и карточек UI тестами

**Files:**
- Modify: `deepseek-api/tests/test_main.py:55-109`

- [ ] **Step 1: Написать падающий тест разметки лаборатории**

Добавить `test_page_contains_day_three_view_switcher_and_four_cards`. Получить `GET /` и проверить наличие `view-chat`, `view-day-three`, `chat-view`, `day-three-view`, `run-day-three`, названий четырёх методов и текстов «Эталонное решение» и «Что будет отправлено в модель». Проверить, что страница содержит `setView`, `renderExperiment`, `textContent`, путь `"/api/day-03/run"` и не содержит `innerHTML`.

- [ ] **Step 2: Проверить CSS-контракты раскладки**

В этом же тесте извлечь блок `.day-three-grid` через существующий `css_block` и проверить `display:grid` и `grid-template-columns:repeat(2`. В части после `@media (max-width:860px)` проверить переопределение `.day-three-grid` с `grid-template-columns:1fr`. Проверить, что `.day-three-view[hidden]` и `.chat-view[hidden]` не занимают место.

- [ ] **Step 3: Запустить тесты и убедиться в RED-фазе**

Run: `python3 -m unittest discover -s tests -v`

Expected: новый UI-тест падает из-за отсутствия переключателя, лаборатории и её CSS; API-тесты из Task 2 остаются зелёными.

- [ ] **Step 4: Сделать отдельный коммит RED-фазы**

```bash
git add deepseek-api/tests/test_main.py
git commit -m "test: define day three laboratory layout"
```

### Task 4: Реализовать переключатель и безопасную отрисовку эксперимента

**Files:**
- Modify: `deepseek-api/main.py:27-78`
- Test: `deepseek-api/tests/test_main.py`

- [ ] **Step 1: Добавить светлую раскладку лаборатории**

В CSS сохранить существующие цвета. Добавить стили `.view-toggle`, `.view-toggle button`, `.chat-view`, `.day-three-view`, `.day-three-grid`, `.method-card`, `.prompt-call`, `.prompt-label`, `.reference-panel` и `[hidden]`. Сетка `.day-three-grid` на широких экранах должна использовать две колонки; в существующем breakpoint 860px — одну. Не менять сетку трёх основных панелей приложения и не добавлять тёмную тему.

- [ ] **Step 2: Обернуть обычный чат и добавить доступный переключатель**

Внутри `<main class="chat">` добавить две кнопки `id="view-chat"` и `id="view-day-three"` с `type="button"`, `aria-pressed` и понятными русскими подписями. Обернуть существующие `.thread` и `.composer-area` в `id="chat-view"`. Добавить соседний `section id="day-three-view" hidden`, верхнюю панель с неизменяемой задачей и кнопкой `id="run-day-three"`, контейнер `id="day-three-cards"` и нижнюю панель эталона/критериев. В разметке подготовить ровно четыре пустые карточки с названиями методов и текстом ожидания.

- [ ] **Step 3: Добавить состояние и функции отрисовки без HTML-строк**

Расширить `state` полями `view: "chat"`, `experiment: null`, `experimentRunning: false`. Реализовать `setView(view)`, который меняет `hidden`, `aria-pressed` и не трогает `sessions`, `activeId`, `metadata` или `experiment`. Реализовать `renderExperiment()` через `replaceChildren`, `document.createElement` и `textContent`: для каждого метода отрисовать итоговый ответ и `<details>` с каждым элементом `calls`, показывая его подпись, `systemPrompt` и `userPrompt`. Для метода с двумя вызовами показать обе подписанные пары и ответ первого вызова. После успешного ответа дополнить нижнюю панель эталонным решением и критериями: порядок действий, нагрев лампы и сопоставление всех трёх состояний.

- [ ] **Step 4: Связать кнопку с маршрутом и сохранить состояние при ошибке**

Обработчик `run-day-three` должен установить `experimentRunning`, отключить кнопку, показать состояние загрузки и выполнить `POST /api/day-03/run` без тела. При `response.ok` присвоить `state.experiment = body.experiment` и вызвать `renderExperiment`. При ошибке показать текст ошибки, не присваивать `state.experiment` и не очищать уже показанный успешный эксперимент. В `finally` снова включить кнопку. Добавить обработчики кнопок переключателя и вызвать `setView("chat")` при инициализации.

- [ ] **Step 5: Запустить полный набор тестов и проверить GREEN-фазу**

Run: `python3 -m unittest discover -s tests -v`

Expected: все тесты проходят, включая проверку двухколоночной/мобильной сетки, безопасной отрисовки и нового API-контракта.

- [ ] **Step 6: Сделать коммит реализации интерфейса**

```bash
git add deepseek-api/main.py deepseek-api/tests/test_main.py
git commit -m "feat: add day three laboratory view"
```

### Task 5: Описать запуск и сравнение результатов

**Files:**
- Modify: `deepseek-api/README.md:35-50`

- [ ] **Step 1: Добавить раздел «День 3 — способы рассуждения»**

После описания существующих настроек добавить инструкцию: открыть переключатель «День 3», нажать «Запустить 4 стратегии» и дождаться результата. Перечислить четыре метода и объяснить, что «Свой промпт» выполняет два API-вызова. Указать, что результаты не попадают в историю обычного чата и не используют его настройки.

- [ ] **Step 2: Зафиксировать эталон и критерии сравнения**

В README изложить эталонное решение задачи о выключателях и назвать три критерия точности: правильная последовательность, использование тепла лампы, верное сопоставление всех трёх исходов. Предложить сравнивать четыре карточки по этой шкале, а не делать необъяснимое автоматическое ранжирование.

- [ ] **Step 3: Запустить финальную проверку**

Run: `python3 -m unittest discover -s tests -v && git diff --check`

Expected: все тесты проходят и проверка пробелов в diff завершается без вывода.

- [ ] **Step 4: Сделать коммит документации**

```bash
git add deepseek-api/README.md
git commit -m "docs: describe day three reasoning comparison"
```
