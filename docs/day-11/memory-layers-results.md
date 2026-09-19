# День 11 — результат: слои памяти агента

| Слой | Хранилище | Срок жизни | В prompt | Очистка |
|---|---|---|---|---|
| Краткосрочный | `messages` агента в `deepseek-api/data/agents.json` | Ограничен выбранной стратегией Day 10; для Sliding Window — `windowSize` | Нативные сообщения `user`/`assistant` | Только новая история/стратегия или удаление агента |
| Рабочий | `context.memoryLayers.working` в том же JSON | До явной замены или очистки | `[WORKING_MEMORY]` с task и data | «Очистить рабочую память» сохраняет `{ "task": "", "data": {} }` |
| Долговременный | `context.memoryLayers.longTerm` в том же JSON | До явного изменения или удаления агента | `[LONG_TERM_MEMORY]` с profile, decisions и knowledge | Отдельное сохранение пустой категории |

Все слои восстанавливаются при перезапуске сервера. `working` и `longTerm`
сохраняются отдельными API-операциями и не смешиваются с историей диалога.
Кнопка очистки рабочей памяти не меняет профиль, решения или знания.

Проверка выполнена тестами:

- `test_memory_layers_are_injected_separately_and_traced` доказывает порядок
  блоков и точную трассировку short-term, working и long-term данных.
- `test_registry_persists_working_and_long_term_memory_across_restart`
  подтверждает восстановление после нового `AgentRegistry`.
- `test_memory_api_updates_each_layer_and_persists` подтверждает явное
  адресное сохранение по HTTP и перезапуск `ChatServer`.
- `test_panel_exposes_three_layers_and_explicit_save_actions` проверяет
  раздельные действия постоянной панели интерфейса.

Последняя проверка: `python3 -m unittest discover -s tests -v` — 127 тестов
пройдено.
