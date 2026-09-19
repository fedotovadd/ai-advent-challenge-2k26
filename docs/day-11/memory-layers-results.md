# День 11 — результат: слои памяти агента

| Слой | Хранилище | Срок жизни | В prompt | Очистка |
|---|---|---|---|---|
| Краткосрочный | `messages` агента в `deepseek-api/data/agents.json` | Ограничен выбранной стратегией Day 10; для Sliding Window — `windowSize` | Нативные сообщения `user`/`assistant` | Только новая история/стратегия или удаление агента |
| Рабочий | `context.memoryLayers.working` в том же JSON | До явной замены или очистки | `[WORKING_MEMORY]` с task и data | `/clear-working` сохраняет `{ "task": "", "data": {} }` |
| Долговременный | `context.memoryLayers.longTerm` в том же JSON | До явного изменения или удаления агента | `[LONG_TERM_MEMORY]` с profile, decisions и knowledge | `/clear-long` очищает только этот слой |

Все слои восстанавливаются при перезапуске сервера. `working` и `longTerm`
сохраняются отдельно и не смешиваются с историей диалога.

Данные добавляются командами в поле чата:

- `/working Текст текущей задачи` — заменяет описание рабочей задачи;
- `/working-data Ключ: значение` — добавляет или обновляет данные текущей задачи;
- `/profile Ключ: значение` — добавляет или обновляет поле профиля;
- `/decision Текст решения` — добавляет отдельное принятое решение;
- `/knowledge Ключ: значение` — добавляет или обновляет сохранённое знание;
- `/clear-working` — очищает только рабочий слой;
- `/clear-working-data` — очищает только `working.data` и оставляет текущую задачу;
- `/clear-profile`, `/clear-decisions`, `/clear-knowledge` — очищают только указанную категорию долговременной памяти;
- `/clear-long` — очищает только долговременный слой;
- `/clear-memory` — очищает оба постоянных слоя.

Команды обрабатываются локально: не вызывают DeepSeek, не добавляются в
краткосрочную историю и сразу сохраняются в `agents.json`. Панель «Память»
только показывает содержание слоёв и трассировку последнего prompt.

Проверка выполнена тестами:

- `test_memory_layers_are_injected_separately_and_traced` доказывает порядок
  блоков и точную трассировку short-term, working и long-term данных.
- `test_registry_persists_working_and_long_term_memory_across_restart`
  подтверждает восстановление после нового `AgentRegistry`.
- `test_memory_commands_are_local_persistent_and_clear_only_the_requested_layer`
  подтверждает команды, отсутствие вызова модели, адресную очистку и
  восстановление после перезапуска `ChatServer`.
- `test_panel_exposes_three_read_only_layers` проверяет, что интерфейс
  показывает слои без полей и кнопок редактирования.

Последняя проверка: `python3 -m unittest discover -s tests -v`.
