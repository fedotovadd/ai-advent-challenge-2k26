import concurrent.futures
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from openai import OpenAI

from agent import AgentRegistry, MissingApiKeyError


MODEL = "deepseek-v4-flash"
MODELS = ("deepseek-v4-flash", "glm-4.7-flash", "deepseek-v4-pro")
MODEL_PRICING = {
    "deepseek-v4-flash": {"input": 0.22, "output": 0.66},
    "glm-4.7-flash": None,
    "deepseek-v4-pro": {"input": 0.66, "output": 1.98},
}
SYSTEM_PROMPT = (
    "Ты AI-помощник. Отвечай ясно, кратко, по-русски. "
    "Возвращай обычный текст без Markdown-разметки."
)
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
JSON_OUTPUT_INSTRUCTION = "Верни только валидный JSON без Markdown-разметки."
DEFAULT_SETTINGS = {
    "model": MODEL,
    "systemPrompt": SYSTEM_PROMPT,
    "format": "text",
    "maxTokens": None,
    "stop": "",
}
DEFAULT_TEMPERATURE = 1
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

PAGE = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepSeek — чат</title>
  <style>
    :root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color:#3d2922; background:#f8f2ed; }
    * { box-sizing:border-box; } body { margin:0; min-height:100vh; background:#f8f2ed; } button, textarea, input, select { font:inherit; } button { cursor:pointer; }
    .app { height:100vh; overflow:hidden; display:grid; grid-template-columns:240px minmax(0,1fr) 330px; background:#fffdfa; } .app.day-three-active { grid-template-columns:240px minmax(0,1fr); }
    .sidebar, .metadata { padding:20px 14px; background:#fbf5f0; } .sidebar { border-right:1px solid #ecdcd2; display:flex; flex-direction:column; } .metadata { border-left:1px solid #ecdcd2; overflow:auto; } .metadata[hidden] { display:none; }
    .brand { display:flex; align-items:center; gap:9px; padding:4px 8px 24px; font-size:18px; font-weight:750; color:#3a251e; } .mark { display:grid; place-items:center; width:28px; height:28px; border-radius:9px; color:#fff; background:linear-gradient(135deg,#9d3f2f,#d77951); }
    .new-chat, .send { border:0; border-radius:10px; background:#983d2d; color:#fffaf6; font-weight:650; } .new-chat { padding:11px 12px; text-align:left; margin-bottom:22px; } .new-chat:hover, .send:hover { background:#813224; }
    .sessions-label { margin:0 8px 8px; color:#a39288; font-size:11px; text-transform:uppercase; letter-spacing:.08em; } #agent-list { display:grid; gap:3px; }
    .session { width:100%; border:0; border-radius:9px; padding:10px 9px; text-align:left; background:transparent; color:#79675e; font-size:13px; } .session:hover { background:#f3e7df; } .session.active { background:#f3dfd4; color:#7c3529; font-weight:700; }
    .session-date { display:block; margin-top:3px; color:#a8988f; font-size:10px; font-weight:400; } .account { margin-top:auto; padding:15px 8px 2px; border-top:1px solid #ebddd6; color:#7c6a60; font-size:12px; }
    .chat { height:100vh; min-height:0; overflow:hidden; display:flex; min-width:0; flex-direction:column; background:#fffdfa; } header { height:68px; flex:0 0 auto; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:0 28px; border-bottom:1px solid #f0e6e0; } #chat-title { color:#402a22; font-size:16px; font-weight:750; } .model { margin-top:3px; color:#a39289; font-size:11px; }
    .view-toggle { display:flex; gap:3px; padding:3px; border:1px solid #eadbd3; border-radius:10px; background:#f8eee8; } .view-toggle button { border:0; border-radius:7px; padding:7px 12px; color:#8b7065; background:transparent; font-size:12px; font-weight:700; } .view-toggle button[aria-pressed="true"] { color:#fffaf6; background:#983d2d; }
    .chat-view { flex:1; min-height:0; display:flex; flex-direction:column; } .chat-view[hidden] { display:none; } .day-three-view[hidden] { display:none; }
    .thread { flex:1; min-height:0; overflow:auto; padding:32px clamp(18px,8vw,100px); } .message { display:flex; gap:10px; max-width:88%; margin-bottom:22px; } .message.user { margin-left:auto; justify-content:flex-end; }
    .avatar { display:grid; place-items:center; flex:0 0 30px; width:30px; height:30px; border-radius:9px; color:#fff; background:linear-gradient(135deg,#9d3f2f,#d77951); font-size:12px; font-weight:750; } .bubble { color:#5d4940; font-size:14px; line-height:1.55; white-space:pre-wrap; } .message.user .bubble { padding:11px 14px; border-radius:14px 14px 3px 14px; color:#5e382f; background:#f8e7de; } .message.chat-status .bubble { padding:10px 12px; border-radius:4px 14px 14px; color:#7d6257; background:#f7eee8; font-size:13px; } .message.chat-status.error .bubble { color:#a72c23; background:#fae3df; } .author { margin-bottom:3px; color:#4a3027; font-size:11px; font-weight:700; }
    .composer-area { flex:0 0 auto; padding:0 28px 20px; } .composer { display:flex; align-items:end; gap:10px; padding:8px 9px 8px 14px; border:1px solid #e7d9d1; border-radius:14px; background:#fff; } #message-input { min-height:28px; max-height:120px; flex:1; resize:vertical; border:0; outline:0; color:#4d352c; background:transparent; } #message-input::placeholder { color:#ad9d94; } .send { min-width:100px; padding:10px 12px; } .send:disabled { cursor:wait; opacity:.65; } .hint { margin-top:8px; color:#ae9e95; text-align:center; font-size:11px; }
    .day-three-view { flex:1; min-height:0; overflow:auto; padding:24px 28px 32px; background:#fffdfa; } .day-three-top, .reference-panel { padding:20px; border:1px solid #eadbd3; border-radius:14px; background:#fbf5f0; } .day-three-heading { display:flex; align-items:start; justify-content:space-between; gap:18px; } .day-three-heading h1 { margin:0 0 5px; color:#402a22; font-size:20px; } .day-three-heading p { margin:0; color:#8b7469; font-size:12px; } .day-three-task-field { display:grid; gap:6px; margin:16px 0; color:#7d6257; font-size:11px; font-weight:750; letter-spacing:.06em; text-transform:uppercase; } #day-three-task { width:100%; min-height:110px; resize:vertical; padding:12px 13px; border:1px solid #dfcec4; border-radius:10px; outline:none; color:#5d4940; background:#fffdfa; font-size:14px; font-weight:400; line-height:1.55; letter-spacing:normal; text-transform:none; } #day-three-task:focus { border-color:#b85b46; box-shadow:0 0 0 3px rgba(184,91,70,.12); } .day-three-actions { display:flex; align-items:center; gap:12px; } #run-day-three { border:0; border-radius:10px; padding:10px 14px; color:#fffaf6; background:#983d2d; font-weight:700; } #run-day-three:disabled { cursor:wait; opacity:.65; } #day-three-status { color:#8b7469; font-size:12px; } #day-three-status[data-kind="error"] { color:#a72c23; }
    .day-three-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin:16px 0; } .method-card { min-width:0; padding:18px; border:1px solid #eaded7; border-radius:14px; background:#fff; } .method-card h2 { margin:0 0 7px; color:#4a3027; font-size:16px; } .method-description { margin:0 0 14px; color:#8b7469; font-size:12px; line-height:1.45; } .method-answer { margin:0; color:#5d4940; font-size:13px; line-height:1.55; white-space:pre-wrap; word-break:break-word; } .method-card details { margin-top:15px; border-top:1px solid #f0e5df; padding-top:12px; } .method-card summary { cursor:pointer; color:#913d2d; font-size:12px; font-weight:700; } .prompt-call { margin-top:12px; padding:12px; border-radius:10px; background:#fbf5f0; } .prompt-call h3 { margin:0 0 9px; color:#6f5044; font-size:12px; } .prompt-label { display:block; margin:8px 0 4px; color:#9a7f73; font-size:10px; font-weight:750; letter-spacing:.06em; text-transform:uppercase; } .prompt-value { margin:0; color:#5e4a40; font-size:11px; line-height:1.5; white-space:pre-wrap; word-break:break-word; } .reference-panel { margin-top:16px; } .reference-panel h2 { margin:0 0 10px; color:#4a3027; font-size:16px; } .reference-panel p, .reference-panel li { color:#5d4940; font-size:13px; line-height:1.5; } .reference-panel ul { margin:10px 0 0; padding-left:20px; } .comparison-panel { margin-top:18px; padding-top:18px; border-top:1px solid #e6d4ca; } .comparison-panel details { margin-top:12px; } .comparison-panel summary { cursor:pointer; color:#913d2d; font-size:12px; font-weight:700; }
    .settings-panel { padding-bottom:17px; border-bottom:1px solid #eaded7; } .metadata-section { padding-top:18px; } .metadata-title { display:flex; align-items:center; justify-content:space-between; margin:4px 3px 14px; } .metadata-title h2 { margin:0; color:#4a3027; font-size:14px; } .metadata-title span { color:#a34b38; font-size:10px; } .setting-field { display:grid; gap:5px; margin-bottom:10px; color:#8b796e; font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; } .setting-field textarea, .setting-field input, .setting-field select { width:100%; padding:9px 10px; border:1px solid #e7d9d1; border-radius:8px; outline-color:#c9634c; color:#5e4a40; background:#fffdfa; font-size:12px; font-weight:400; letter-spacing:normal; text-transform:none; } .setting-field textarea { min-height:72px; resize:vertical; line-height:1.4; } .temperature-label { display:flex; align-items:center; justify-content:space-between; } .temperature-label output { border-radius:8px; padding:3px 6px; color:#7c3529; background:#f3dfd4; font-size:12px; } .setting-field input[type="range"] { padding:0; accent-color:#983d2d; } .setting-hint { margin:-4px 0 10px; color:#aa9990; font-size:10px; line-height:1.4; } .save-settings { width:100%; border:0; border-radius:9px; padding:10px; color:#fffaf6; background:#983d2d; font-size:12px; font-weight:700; } .save-settings:hover { background:#813224; } .save-settings:disabled { cursor:wait; opacity:.65; } .meta-card { margin-bottom:11px; overflow:hidden; border:1px solid #eaded7; border-radius:11px; background:#fffdfa; } .meta-label { padding:10px 11px; border-bottom:1px solid #f2eae5; color:#8b796e; font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; } .meta-value { min-height:40px; padding:11px; color:#5e4a40; font-size:12px; line-height:1.5; white-space:pre-wrap; word-break:break-word; } pre.meta-value { margin:0; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; }
    @media (max-width:860px) { .app, .app.day-three-active { height:auto; overflow:visible; grid-template-columns:1fr; } .chat { height:100vh; } .sidebar { border-right:0; border-bottom:1px solid #ecdcd2; } .sidebar, .metadata { padding:14px; } .brand { padding-bottom:12px; } .new-chat { margin-bottom:12px; } #agent-list { display:flex; overflow:auto; } .session { min-width:150px; } .account { display:none; } .metadata { border-top:1px solid #ecdcd2; border-left:0; } header { height:auto; min-height:68px; flex-wrap:wrap; padding:10px 18px; } .view-toggle { order:3; width:100%; } .view-toggle button { flex:1; } .thread { min-height:340px; padding:24px 18px; } .composer-area { padding:0 18px 16px; } .day-three-view { padding:18px; } .day-three-grid { grid-template-columns:1fr; } .day-three-heading { flex-direction:column; } }
    .day-three-task-field select { width:100%; padding:12px 13px; border:1px solid #dfcec4; border-radius:10px; outline:none; color:#5d4940; background:#fffdfa; font-size:14px; font-weight:400; letter-spacing:normal; text-transform:none; } .day-three-task-text { margin:10px 0 0; color:#5d4940; font-size:13px; font-weight:400; line-height:1.55; letter-spacing:normal; text-transform:none; white-space:pre-wrap; }
    @media (max-width:860px) { .app { height:auto; overflow:visible; grid-template-columns:1fr; } .chat { height:100vh; } }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar"><div class="brand"><span class="mark">D</span>DeepSeek</div><button class="new-chat" id="create-agent" type="button">＋ Создать агента</button><div class="sessions-label">Агенты · <span id="agent-total">0</span></div><nav id="agent-list" aria-label="Агенты"></nav><div class="account">Один сервер · N агентов<br><span class="model">История хранится до перезапуска</span></div></aside>
    <main class="chat"><header><div><div id="chat-title">Агент 1</div><div class="model">DeepSeek V4 Flash</div></div><div class="view-toggle" role="group" aria-label="Переключить режим"><button id="view-chat" type="button" aria-pressed="true">Чат</button><button id="view-day-three" type="button" aria-pressed="false">День 3</button></div></header><section class="chat-view" id="chat-view"><section class="thread" id="thread" aria-live="polite"></section><div class="composer-area"><form class="composer" id="composer"><textarea id="message-input" aria-label="Сообщение" placeholder="Напишите сообщение…" required></textarea><button class="send" id="send" type="submit" disabled>Отправить</button></form><div class="hint">⌘ Enter или Ctrl Enter — отправить</div></div></section><section class="day-three-view" id="day-three-view" hidden><section class="day-three-top"><div class="day-three-heading"><div><h1>День 3 — способы рассуждения</h1><p>Одна задача, четыре стратегии</p></div></div><label class="day-three-task-field" for="day-three-task">Условие задачи<textarea id="day-three-task" aria-label="Условие задачи">За закрытой дверью находится лампочка. Снаружи — три выключателя, и только один из них подключён к лампочке. В комнату можно войти только один раз. Как определить, какой выключатель подключён к лампочке?</textarea></label><div class="day-three-actions"><button id="run-day-three" type="button">Запустить 4 стратегии</button><span id="day-three-status" role="status">Готово к запуску</span></div></section><section class="day-three-grid" id="day-three-cards"><article class="method-card"><h2>Напрямую</h2><p class="method-description">Только текст задачи.</p><p class="method-answer">Результат ещё не получен.</p></article><article class="method-card"><h2>Пошагово</h2><p class="method-description">Явная просьба решать по шагам.</p><p class="method-answer">Результат ещё не получен.</p></article><article class="method-card"><h2>Свой промпт</h2><p class="method-description">Два вызова: создать промпт, затем решить задачу.</p><p class="method-answer">Результат ещё не получен.</p></article><article class="method-card"><h2>Группа экспертов</h2><p class="method-description">Независимые взгляды аналитика, инженера и критика.</p><p class="method-answer">Результат ещё не получен.</p></article></section><section class="reference-panel" id="day-three-reference" hidden></section></section></main>
    <aside class="metadata"><section class="settings-panel"><div class="metadata-title"><h2>Настройки ответа</h2><span id="settings-status">агент</span></div><div id="settings-form"><label class="setting-field" for="system-prompt-input">System prompt<textarea id="system-prompt-input" aria-label="System prompt" required></textarea></label><label class="setting-field" for="response-format-input">Формат ответа<select id="response-format-input" aria-label="Формат ответа"><option value="text">Обычный текст</option><option value="json">JSON</option></select></label><label class="setting-field" for="max-tokens-input">Максимум токенов<input id="max-tokens-input" aria-label="Максимум токенов" type="number" min="1" step="1" placeholder="Без ограничения"></label><label class="setting-field" for="stop-input">Стоп-последовательность<input id="stop-input" aria-label="Стоп-последовательность" type="text" placeholder="Например: &lt;END&gt;"></label><label class="setting-field" for="temperature"><span class="temperature-label">Температура <output id="temperature-value" for="temperature">1</output></span><input id="temperature" aria-label="Температура" type="range" min="0" max="2" step="0.1" value="1"></label><p class="setting-hint">Настройки сохраняются автоматически для текущего агента.</p></div></section><section class="metadata-section"><div class="metadata-title"><h2>Метаданные</h2><span id="meta-status">ожидание</span></div><section class="meta-card"><div class="meta-label">User prompt</div><div class="meta-value" id="user-prompt">—</div></section><section class="meta-card"><div class="meta-label">System prompt</div><div class="meta-value" id="system-prompt">—</div></section><section class="meta-card"><div class="meta-label">Отправлено в API</div><pre class="meta-value" id="payload">—</pre></section><section class="meta-card"><div class="meta-label">Статус</div><div class="meta-value" id="api-status">—</div></section></section></aside>
  </div>
  <script>
    const state = { agents: [], activeId: null, metadata: null, settingsTimer: null, pendingSettings: null, settingsSaveInFlight: null, pendingAgentIds:new Set(), chatStatuses:{}, drafts:{}, globalChatStatus:null, view:"chat", experiment:null, experimentRunning:false };
    const elements = { app:document.querySelector(".app"), metadata:document.querySelector(".metadata"), agents:document.querySelector("#agent-list"), agentTotal:document.querySelector("#agent-total"), createAgent:document.querySelector("#create-agent"), title:document.querySelector("#chat-title"), thread:document.querySelector("#thread"), input:document.querySelector("#message-input"), send:document.querySelector("#send"), settingsStatus:document.querySelector("#settings-status"), systemPromptInput:document.querySelector("#system-prompt-input"), responseFormatInput:document.querySelector("#response-format-input"), maxTokensInput:document.querySelector("#max-tokens-input"), stopInput:document.querySelector("#stop-input"), temperature:document.querySelector("#temperature"), temperatureValue:document.querySelector("#temperature-value"), metaStatus:document.querySelector("#meta-status"), userPrompt:document.querySelector("#user-prompt"), systemPrompt:document.querySelector("#system-prompt"), payload:document.querySelector("#payload"), apiStatus:document.querySelector("#api-status"), viewChat:document.querySelector("#view-chat"), viewDayThree:document.querySelector("#view-day-three"), chatView:document.querySelector("#chat-view"), dayThreeView:document.querySelector("#day-three-view"), dayThreeTask:document.querySelector("#day-three-task"), runDayThree:document.querySelector("#run-day-three"), dayThreeStatus:document.querySelector("#day-three-status"), dayThreeCards:document.querySelector("#day-three-cards"), dayThreeReference:document.querySelector("#day-three-reference") };
    state.activeDayThreeTaskId="server-messages"; state.experiments={};
    const dayThreeTasks=[{id:"server-messages",title:"Три сервера",task:"В системе три сервера: A, B и C. Ровно один неисправен. Сообщения: A — «B неисправен»; B — «A и C находятся в одинаковом состоянии»; C — «A исправен». Ровно одно сообщение правдиво. Какой сервер неисправен? Докажите вывод."},{id:"channel-analysis",title:"Выбор канала",task:"Канал A получил 1 000 показов и 30 регистраций. Канал B получил 200 показов и 10 регистраций. Какой канал эффективнее по конверсии? Если дополнительно купить 100 показов и предположить, что конверсия не изменится, сколько регистраций ожидается от каждого канала?"}];
    const taskSelect=document.createElement("select"); taskSelect.id="day-three-task-select"; taskSelect.setAttribute("aria-label","Задача для сравнения"); dayThreeTasks.forEach((task)=>{const option=document.createElement("option");option.value=task.id;option.textContent=task.title;taskSelect.append(option);}); const taskText=document.createElement("p"); taskText.id="day-three-task-text"; taskText.className="day-three-task-text"; taskText.textContent=dayThreeTasks[0].task; elements.dayThreeTask.parentElement.htmlFor=taskSelect.id; elements.dayThreeTask.replaceWith(taskSelect); taskSelect.after(taskText); elements.dayThreeTaskSelect=taskSelect; elements.dayThreeTaskText=taskText;
    const modelField=document.createElement("label"); modelField.className="setting-field"; modelField.textContent="Модель"; const modelInput=document.createElement("select"); modelInput.id="model-input"; modelInput.setAttribute("aria-label","Модель"); [["deepseek-v4-flash","deepseek-v4-flash"],["glm-4.7-flash","glm-4.7-flash"],["deepseek-v4-pro","deepseek-v4-pro"]].forEach(([value,label])=>{const option=document.createElement("option");option.value=value;option.textContent=label;modelInput.append(option);}); modelField.append(modelInput); document.querySelector("#settings-form").prepend(modelField); elements.modelInput=modelInput;
    function usageCard(label,id) { const card=document.createElement("section"); card.className="meta-card"; const title=document.createElement("div"); title.className="meta-label"; title.textContent=label; const value=document.createElement("div"); value.className="meta-value"; value.id=id; value.textContent="—"; card.append(title,value); return card; }
    const metadataSection=document.querySelector(".metadata-section"); const metadataBefore=elements.userPrompt.parentElement; [["Время ответа","response-time"],["Токены","token-count"],["Стоимость запроса","request-cost"]].forEach(([label,id])=>metadataSection.insertBefore(usageCard(label,id),metadataBefore)); elements.responseTime=document.querySelector("#response-time"); elements.tokenCount=document.querySelector("#token-count"); elements.requestCost=document.querySelector("#request-cost");
    async function api(url, options={}) { const response=await fetch(url,options); const body=await response.json().catch(()=>({error:"Сервер вернул некорректный ответ."})); return {response,body}; }
    function activeAgent() { return state.agents.find((agent)=>agent.id===state.activeId); }
    function syncSendButton() { elements.send.disabled=state.pendingAgentIds.has(state.activeId)||!elements.input.value.trim(); }
    function setView(view) { state.view=view; const chatSelected=view==="chat"; elements.chatView.hidden=!chatSelected; elements.dayThreeView.hidden=chatSelected; elements.metadata.hidden=!chatSelected; elements.app.classList.toggle("day-three-active",!chatSelected); elements.viewChat.setAttribute("aria-pressed",String(chatSelected)); elements.viewDayThree.setAttribute("aria-pressed",String(!chatSelected)); }
    function renderExperiment() { const experiment=state.experiment; if(!experiment)return; elements.dayThreeCards.replaceChildren(); experiment.methods.forEach((method)=>{ const card=document.createElement("article"); card.className="method-card"; const title=document.createElement("h2"); title.textContent=method.title; const description=document.createElement("p"); description.className="method-description"; description.textContent=method.description; const answer=document.createElement("p"); answer.className="method-answer"; answer.textContent=method.calls[method.calls.length-1].answer; const details=document.createElement("details"); const summary=document.createElement("summary"); summary.textContent="Что будет отправлено в модель"; details.append(summary); method.calls.forEach((call,index)=>{ const callBlock=document.createElement("section"); callBlock.className="prompt-call"; const callTitle=document.createElement("h3"); callTitle.textContent=method.id==="generated_prompt"?(index===0?"Вызов 1 — составить промпт":"Вызов 2 — решить с этим промптом"):"Вызов API"; callBlock.append(callTitle); [["System prompt",call.systemPrompt],["User prompt",call.userPrompt]].forEach(([label,value])=>{ const promptLabel=document.createElement("span"); promptLabel.className="prompt-label"; promptLabel.textContent=label; const promptValue=document.createElement("p"); promptValue.className="prompt-value"; promptValue.textContent=value; callBlock.append(promptLabel,promptValue); }); if(method.id==="generated_prompt"&&index===0){ const generatedLabel=document.createElement("span"); generatedLabel.className="prompt-label"; generatedLabel.textContent="Сгенерированный промпт"; const generatedValue=document.createElement("p"); generatedValue.className="prompt-value"; generatedValue.textContent=call.answer; callBlock.append(generatedLabel,generatedValue); } details.append(callBlock); }); card.append(title,description,answer,details); elements.dayThreeCards.append(card); }); elements.dayThreeReference.replaceChildren(); const referenceTitle=document.createElement("h2"); referenceTitle.textContent="Эталонное решение"; const referenceText=document.createElement("p"); referenceText.textContent=experiment.referenceSolution||"Для изменённой задачи нейросеть выводит независимый эталон в сравнении ниже."; const criteriaTitle=document.createElement("h2"); criteriaTitle.textContent="Критерии сравнения"; const criteria=document.createElement("ul"); ["Правильная последовательность действий.","Использование нагрева лампы.","Корректное сопоставление всех трёх состояний с выключателями."].forEach((text)=>{ const item=document.createElement("li"); item.textContent=text; criteria.append(item); }); const comparisonPanel=document.createElement("section"); comparisonPanel.className="comparison-panel"; const comparisonTitle=document.createElement("h2"); comparisonTitle.textContent="Сравнение от нейросети"; comparisonPanel.append(comparisonTitle); if(experiment.comparison.status==="success"){ const comparisonText=document.createElement("p"); comparisonText.textContent=experiment.comparison.call.answer; const comparisonDetails=document.createElement("details"); const comparisonSummary=document.createElement("summary"); comparisonSummary.textContent="Что было отправлено для сравнения"; const comparisonCall=document.createElement("section"); comparisonCall.className="prompt-call"; [["System prompt",experiment.comparison.call.systemPrompt],["User prompt",experiment.comparison.call.userPrompt]].forEach(([label,value])=>{ const promptLabel=document.createElement("span"); promptLabel.className="prompt-label"; promptLabel.textContent=label; const promptValue=document.createElement("p"); promptValue.className="prompt-value"; promptValue.textContent=value; comparisonCall.append(promptLabel,promptValue); }); comparisonDetails.append(comparisonSummary,comparisonCall); comparisonPanel.append(comparisonText,comparisonDetails); } else { const comparisonError=document.createElement("p"); comparisonError.textContent=experiment.comparison.message; comparisonPanel.append(comparisonError); } elements.dayThreeReference.append(referenceTitle,referenceText,criteriaTitle,criteria,comparisonPanel); elements.dayThreeReference.hidden=false; }
    function renderExperiment() {
      const experiment=state.experiments[state.activeDayThreeTaskId];
      elements.dayThreeCards.replaceChildren(); elements.dayThreeReference.replaceChildren(); elements.dayThreeReference.hidden=true;
      elements.runDayThree.textContent=experiment?"Запустить снова":"Запустить 4 стратегии";
      if(!experiment){ const empty=document.createElement("p"); empty.className="method-answer"; empty.textContent="Для этой задачи результат ещё не получен."; elements.dayThreeCards.append(empty); return; }
      experiment.methods.forEach((method)=>{ const card=document.createElement("article"); card.className="method-card"; const title=document.createElement("h2"); title.textContent=method.title; const description=document.createElement("p"); description.className="method-description"; description.textContent=method.description; const answer=document.createElement("p"); answer.className="method-answer"; answer.textContent=method.answer||method.calls.at(-1).answer; const details=document.createElement("details"); const summary=document.createElement("summary"); summary.textContent="Что будет отправлено в модель"; details.append(summary); method.calls.forEach((call,index)=>{ const block=document.createElement("section"); block.className="prompt-call"; const heading=document.createElement("h3"); heading.textContent=call.title|| (method.id==="generated_prompt"?(index===0?"Вызов 1 — составить промпт":"Вызов 2 — решить с этим промптом"):"Вызов API"); block.append(heading); [["System prompt",call.systemPrompt],["User prompt",call.userPrompt]].forEach(([label,value])=>{const name=document.createElement("span");name.className="prompt-label";name.textContent=label;const text=document.createElement("p");text.className="prompt-value";text.textContent=value;block.append(name,text);}); details.append(block); }); card.append(title,description,answer,details); elements.dayThreeCards.append(card); });
      const referenceTitle=document.createElement("h2"); referenceTitle.textContent="Эталонное решение"; const reference=document.createElement("p"); reference.textContent=experiment.referenceSolution; const criteriaTitle=document.createElement("h2"); criteriaTitle.textContent="Критерии сравнения"; const criteria=document.createElement("ul"); experiment.criteria.forEach((criterion)=>{const item=document.createElement("li");item.textContent=criterion;criteria.append(item);}); const comparison=document.createElement("section"); comparison.className="comparison-panel"; const comparisonTitle=document.createElement("h2"); comparisonTitle.textContent="Сравнение от нейросети"; comparison.append(comparisonTitle); const comparisonText=document.createElement("p"); comparisonText.textContent=experiment.comparison.status==="success"?experiment.comparison.call.answer:experiment.comparison.message; comparison.append(comparisonText); elements.dayThreeReference.append(referenceTitle,reference,criteriaTitle,criteria,comparison); elements.dayThreeReference.hidden=false;
    }
    function renderAgents() { elements.agents.replaceChildren(); elements.agentTotal.textContent=state.agents.length; state.agents.forEach((agent)=>{ const button=document.createElement("button"); button.type="button"; button.className="session"+(agent.id===state.activeId?" active":""); const title=document.createElement("span"); title.textContent=agent.name; const detail=document.createElement("span"); detail.className="session-date"; detail.textContent=agent.settings.model+" · "+agent.messages.length+" сообщений"; button.append(title,detail); button.addEventListener("click",async()=>{await flushSettingsSave(); state.activeId=agent.id; render(); elements.input.focus();}); elements.agents.append(button); }); }
    function appendChatStatus(agent,status) { const row=document.createElement("article"); row.className="message assistant chat-status "+status.kind; const avatar=document.createElement("div"); avatar.className="avatar"; avatar.textContent="D"; const content=document.createElement("div"); content.className="bubble"; const author=document.createElement("div"); author.className="author"; author.textContent=agent?agent.name:"Система"; const text=document.createElement("div"); text.textContent=status.text; content.append(author,text); row.append(avatar,content); elements.thread.append(row); }
    function chatStatus(agent) { if(!agent)return state.globalChatStatus; if(state.pendingAgentIds.has(agent.id))return {text:"Думаю над ответом…",kind:"waiting"}; if(state.chatStatuses[agent.id])return state.chatStatuses[agent.id]; if(agent.metadata&&agent.metadata.status.kind==="error")return {text:"Не удалось получить ответ. Попробуйте ещё раз.",kind:"error"}; return null; }
    function renderThread() { elements.thread.replaceChildren(); const agent=activeAgent(); const status=chatStatus(agent); elements.title.textContent=agent?agent.name:"Агент не выбран"; if(!agent||!agent.messages.length){if(status)appendChatStatus(agent,status);else {const empty=document.createElement("div"); empty.className="bubble"; empty.textContent="Выберите агента и начните диалог — у каждого будет собственная история."; elements.thread.append(empty);} return;} agent.messages.forEach((message)=>{const row=document.createElement("article"); row.className="message "+message.role; if(message.role==="assistant"){const avatar=document.createElement("div"); avatar.className="avatar"; avatar.textContent="D"; row.append(avatar);} const content=document.createElement("div"); content.className="bubble"; if(message.role==="assistant"){const author=document.createElement("div"); author.className="author"; author.textContent=agent.name; content.append(author);} const text=document.createElement("div"); text.textContent=message.content; content.append(text); row.append(content); elements.thread.append(row);}); if(status)appendChatStatus(agent,status); elements.thread.scrollTop=elements.thread.scrollHeight; }
    function renderSettings() { const settings=activeAgent()?activeAgent().settings:null; if(!settings)return; elements.modelInput.value=settings.model; document.querySelector("header .model").textContent=settings.model; elements.systemPromptInput.value=settings.systemPrompt; elements.responseFormatInput.value=settings.format; elements.maxTokensInput.value=settings.maxTokens===null?"":settings.maxTokens; elements.stopInput.value=settings.stop; }
    function renderMetadata() { const agent=activeAgent(); const metadata=agent?agent.metadata:null; elements.userPrompt.textContent=metadata?metadata.userPrompt:"—"; elements.systemPrompt.textContent=metadata?metadata.systemPrompt:"—"; elements.payload.textContent=metadata?JSON.stringify(metadata.payload,null,2):"—"; elements.apiStatus.textContent=metadata?metadata.status.label:"—"; elements.responseTime.textContent=metadata&&metadata.responseTimeMs!==null?(metadata.responseTimeMs/1000).toFixed(2)+" с":"—"; elements.tokenCount.textContent=metadata&&metadata.usage&&metadata.usage.totalTokens!==null?metadata.usage.totalTokens:"—"; elements.requestCost.textContent=metadata&&metadata.cost?(metadata.cost.kind==="free"?"Бесплатно":"$"+metadata.cost.usd.toFixed(6)):"—"; elements.metaStatus.textContent=metadata?(metadata.status.kind==="success"?"последний запрос":"ошибка"):"ожидание"; }
    function renderComposer() { elements.input.value=state.drafts[state.activeId]||""; syncSendButton(); }
    function render() { renderAgents(); renderThread(); renderSettings(); renderMetadata(); renderComposer(); }
    function upsertAgent(agent, activate=true) { const index=state.agents.findIndex((item)=>item.id===agent.id); if(index===-1) state.agents.push(agent); else state.agents[index]=agent; if(activate)state.activeId=agent.id; }
    function replaceAgent(agent) { const index=state.agents.findIndex((item)=>item.id===agent.id); if(index===-1)state.agents.push(agent); else state.agents[index]=agent; }
    function addOptimisticUserMessage(agentId,text) { const agent=state.agents.find((item)=>item.id===agentId); if(!agent)return; replaceAgent({...agent,messages:[...agent.messages,{role:"user",content:text}]}); if(state.activeId===agentId)renderThread(); }
    function setChatStatus(agentId,text,kind="error") { if(agentId)state.chatStatuses[agentId]={text,kind}; else state.globalChatStatus={text,kind}; if(state.activeId===agentId||!state.activeId)renderThread(); }
    function clearChatStatus(agentId) { if(agentId)delete state.chatStatuses[agentId]; else state.globalChatStatus=null; }
    function settingsValues() { const maxTokens=elements.maxTokensInput.value.trim(); return {model:elements.modelInput.value,systemPrompt:elements.systemPromptInput.value,format:elements.responseFormatInput.value,maxTokens:maxTokens?Number(maxTokens):null,stop:elements.stopInput.value}; }
    function settingsMatch(left,right) { return JSON.stringify(left)===JSON.stringify(right); }
    async function persistSettings(pending) { if(pending.agentId===state.activeId)elements.settingsStatus.textContent="сохраняем"; try { const {response,body}=await api("/api/agents/"+pending.agentId+"/settings",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(pending.settings)}); if(response.ok&&body.agent){replaceAgent(body.agent); if(pending.agentId===state.activeId&&settingsMatch(settingsValues(),pending.settings)){renderSettings();elements.settingsStatus.textContent="сохранено";} return true;} if(pending.agentId===state.activeId){elements.settingsStatus.textContent="ошибка";setChatStatus(pending.agentId,body.error||"Не удалось сохранить настройки.");} return false; } catch(error) { if(pending.agentId===state.activeId){elements.settingsStatus.textContent="ошибка";setChatStatus(pending.agentId,"Не удалось соединиться с локальным сервером.");} return false; }}
    async function savePendingSettings() { if(!state.pendingSettings)return state.settingsSaveInFlight?state.settingsSaveInFlight:true; const pending=state.pendingSettings; state.pendingSettings=null; const promise=persistSettings(pending); state.settingsSaveInFlight=promise; const saved=await promise; if(state.settingsSaveInFlight===promise)state.settingsSaveInFlight=null; return saved; }
    function scheduleSettingsSave(delay=500) { const agent=activeAgent(); if(!agent)return; if(state.settingsTimer)clearTimeout(state.settingsTimer); state.pendingSettings={agentId:agent.id,settings:settingsValues()}; elements.settingsStatus.textContent="изменено"; state.settingsTimer=setTimeout(()=>{state.settingsTimer=null;savePendingSettings();},delay); }
    async function flushSettingsSave() { if(state.settingsTimer){clearTimeout(state.settingsTimer);state.settingsTimer=null;} if(state.settingsSaveInFlight&&!await state.settingsSaveInFlight)return false; return savePendingSettings(); }
    async function loadAgents() { const {response,body}=await api("/api/agents"); if(!response.ok) throw new Error(body.error||"Не удалось загрузить агентов."); state.agents=body.agents; state.activeId=state.agents[0]?state.agents[0].id:null; render(); }
    elements.createAgent.addEventListener("click",async()=>{if(!await flushSettingsSave())return;const {body}=await api("/api/agents",{method:"POST"}); if(body.agent){upsertAgent(body.agent); state.metadata=null; render(); elements.input.focus();}else setChatStatus(state.activeId,body.error||"Не удалось создать агента.");});
    elements.viewChat.addEventListener("click",()=>setView("chat"));
    elements.viewDayThree.addEventListener("click",()=>setView("day-three"));
    elements.runDayThree.addEventListener("click",async()=>{ if(state.experimentRunning)return; state.experimentRunning=true; elements.runDayThree.disabled=true; elements.dayThreeStatus.textContent="Запускаем четыре стратегии…"; elements.dayThreeStatus.removeAttribute("data-kind"); const task=elements.dayThreeTask.value.trim(); try { const {response,body}=await api("/api/day-03/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({task})}); if(!response.ok){elements.dayThreeStatus.textContent=body.error||"Не удалось запустить эксперимент.";elements.dayThreeStatus.dataset.kind="error";return;} state.experiment=body.experiment; renderExperiment(); elements.runDayThree.textContent="Запустить снова"; elements.dayThreeStatus.textContent="Все четыре стратегии готовы"; } catch(error) { elements.dayThreeStatus.textContent="Не удалось соединиться с локальным сервером."; elements.dayThreeStatus.dataset.kind="error"; } finally { state.experimentRunning=false; elements.runDayThree.disabled=false; }});
    const dayThreeButton=elements.runDayThree.cloneNode(true); elements.runDayThree.replaceWith(dayThreeButton); elements.runDayThree=dayThreeButton;
    elements.dayThreeTaskSelect.addEventListener("change",()=>{state.activeDayThreeTaskId=elements.dayThreeTaskSelect.value; elements.dayThreeStatus.textContent="Готово к запуску"; elements.dayThreeStatus.removeAttribute("data-kind"); renderExperiment();});
    elements.runDayThree.addEventListener("click",async()=>{ if(state.experimentRunning)return; const taskId=state.activeDayThreeTaskId; state.experimentRunning=true; elements.runDayThree.disabled=true; elements.dayThreeStatus.textContent="Запускаем четыре стратегии…"; elements.dayThreeStatus.removeAttribute("data-kind"); try { const {response,body}=await api("/api/day-03/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({taskId})}); if(!response.ok){elements.dayThreeStatus.textContent=body.error||"Не удалось запустить эксперимент.";elements.dayThreeStatus.dataset.kind="error";return;} state.experiments[body.experiment.taskId]=body.experiment; renderExperiment(); elements.dayThreeStatus.textContent="Все четыре стратегии готовы"; } catch(error) { elements.dayThreeStatus.textContent="Не удалось соединиться с локальным сервером."; elements.dayThreeStatus.dataset.kind="error"; } finally { state.experimentRunning=false; elements.runDayThree.disabled=false; }});
    renderExperiment();
    const streamingButton=elements.runDayThree.cloneNode(true); elements.runDayThree.replaceWith(streamingButton); elements.runDayThree=streamingButton;
    elements.dayThreeTaskSelect.addEventListener("change",()=>{const task=dayThreeTasks.find((item)=>item.id===elements.dayThreeTaskSelect.value); elements.dayThreeTaskText.textContent=task.task;});
    elements.runDayThree.addEventListener("click",async()=>{ if(state.experimentRunning)return; const taskId=state.activeDayThreeTaskId; state.experimentRunning=true; elements.runDayThree.disabled=true; elements.dayThreeStatus.textContent="Запускаем четыре стратегии…"; elements.dayThreeStatus.removeAttribute("data-kind"); try { const response=await fetch("/api/day-03/stream",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({taskId})}); if(!response.ok){const body=await response.json().catch(()=>({})); throw new Error(body.error||"Не удалось запустить эксперимент.");} const reader=response.body.getReader(); const decoder=new TextDecoder(); let buffer=""; while(true){const chunk=await reader.read(); buffer+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done}); const lines=buffer.split("\n"); buffer=lines.pop(); for(const line of lines){if(!line)continue; const event=JSON.parse(line); if(event.type==="start"){state.experiments[taskId]={...event.experiment,methods:[],comparison:{status:"pending",message:"Собираем итоговое сравнение…"}}; renderExperiment();} else if(event.type==="method"){const experiment=state.experiments[taskId]; experiment.methods=experiment.methods.filter((method)=>method.id!==event.method.id); experiment.methods.push(event.method); renderExperiment(); elements.dayThreeStatus.textContent="Получен ответ: "+event.method.title;} else if(event.type==="comparison"){state.experiments[taskId].comparison=event.comparison; renderExperiment();} else if(event.type==="error"){throw new Error(event.message);} } if(chunk.done)break;} elements.dayThreeStatus.textContent="Все четыре стратегии готовы"; } catch(error) { elements.dayThreeStatus.textContent=error.message||"Не удалось соединиться с локальным сервером."; elements.dayThreeStatus.dataset.kind="error"; } finally { state.experimentRunning=false; elements.runDayThree.disabled=false; }});
    elements.modelInput.addEventListener("change",()=>scheduleSettingsSave(0));
    elements.responseFormatInput.addEventListener("change",()=>scheduleSettingsSave(0));
    [elements.systemPromptInput,elements.maxTokensInput,elements.stopInput].forEach((input)=>input.addEventListener("input",()=>scheduleSettingsSave()));
    elements.temperature.addEventListener("input",()=>{elements.temperatureValue.textContent=elements.temperature.value;});
    document.querySelector("#composer").addEventListener("submit",async(event)=>{event.preventDefault(); const text=elements.input.value.trim(); const agentId=state.activeId; if(!text||!agentId||state.pendingAgentIds.has(agentId))return; if(!await flushSettingsSave()){syncSendButton();return;} state.drafts[agentId]=""; elements.input.value=""; state.pendingAgentIds.add(agentId); addOptimisticUserMessage(agentId,text); syncSendButton(); try { const {response,body}=await api("/api/agents/"+agentId+"/messages",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text,temperature:Number(elements.temperature.value)})}); state.pendingAgentIds.delete(agentId); if(body.agent)upsertAgent(body.agent,false); if(body.metadata)state.metadata=body.metadata; if(response.ok)clearChatStatus(agentId); else setChatStatus(agentId,body.error||"Не удалось получить ответ."); render(); } catch(error) { state.pendingAgentIds.delete(agentId); setChatStatus(agentId,"Не удалось соединиться с локальным сервером."); render(); } finally { syncSendButton(); elements.input.focus(); }});
    elements.input.addEventListener("input",()=>{if(state.activeId)state.drafts[state.activeId]=elements.input.value;syncSendButton();});
    elements.input.addEventListener("keydown",(event)=>{if((event.metaKey||event.ctrlKey)&&event.key==="Enter")document.querySelector("#composer").requestSubmit();});
    syncSendButton();
    setView("chat");
    loadAgents().catch((error)=>setChatStatus(null,error.message));
  </script>
</body>
</html>
"""


def _provider_settings(model):
    if model == "glm-4.7-flash":
        return "ZAI_API_KEY", "https://api.z.ai/api/paas/v4"
    return "DEEPSEEK_API_KEY", "https://api.deepseek.com"


def ask_deepseek(payload, **options):
    key_name, base_url = _provider_settings(payload["model"])
    api_key = os.getenv(key_name)
    if not api_key:
        raise MissingApiKeyError(key_name)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=payload["model"],
        messages=payload["messages"],
        temperature=payload["temperature"],
        **options,
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("empty API response")
    return {"content": response.choices[0].message.content, "usage": response.usage}


def _response_content_and_usage(response):
    if isinstance(response, str):
        return response, None
    if isinstance(response, dict):
        return response.get("content"), response.get("usage")
    return None, None


def _usage_value(usage, name):
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _request_usage_and_cost(model, usage):
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    details = None if usage is None else {
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "totalTokens": total_tokens,
    }
    pricing = MODEL_PRICING[model]
    if pricing is None:
        return details, {"kind": "free"}
    if prompt_tokens is None or completion_tokens is None:
        return details, None
    return details, {
        "kind": "paid",
        "usd": round((prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000, 12),
    }


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


class ChatRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(200, PAGE)
        elif path == "/api/agents":
            self._send_json(200, {"agents": self.server.registry.agents()})
        elif path.startswith("/api/"):
            self._send_json(404, {"error": "Маршрут не найден."})
        else:
            self._send_html(404, "<h1>Страница не найдена</h1>")

    def do_POST(self):
        if not self._same_origin():
            self._send_json(403, {"error": "Запрос с другого источника запрещён."})
            return
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        if path == "/api/day-03/run":
            self._handle_day_three(parsed_path.query)
            return
        if path == "/api/day-03/stream":
            self._handle_day_three_stream(parsed_path.query)
            return
        if path == "/api/agents":
            self._send_json(201, {"agent": self.server.registry.create()})
            return
        if path == "/api/agents/bulk":
            self._handle_bulk_create()
            return
        parts = path.split("/")
        if len(parts) == 5 and parts[:3] == ["", "api", "agents"] and parts[4] == "messages":
            self._handle_message(parts[3])
            return
        self._send_json(404, {"error": "Маршрут не найден."})

    def do_PUT(self):
        if not self._same_origin():
            self._send_json(403, {"error": "Запрос с другого источника запрещён."})
            return
        parts = urlparse(self.path).path.split("/")
        if len(parts) == 5 and parts[:3] == ["", "api", "agents"] and parts[4] == "settings":
            self._handle_settings(parts[3])
            return
        self._send_json(404, {"error": "Маршрут не найден."})

    def _same_origin(self):
        origin = self.headers.get("Origin")
        return not origin or origin == f"http://{self.headers.get('Host')}"

    def _handle_day_three(self, query):
        task_id = self._read_day_three_task_id(query)
        if task_id is None:
            return
        if not os.getenv("DEEPSEEK_API_KEY"):
            self._send_json(503, {"error": "Не задан DEEPSEEK_API_KEY."})
            return
        try:
            experiment = run_day_three_experiment(self.server.ask_model, task_id)
        except Exception:
            self._send_json(502, {"error": "Не удалось получить ответы DeepSeek."})
            return
        self._send_json(200, {"experiment": experiment})

    def _handle_day_three_stream(self, query):
        task_id = self._read_day_three_task_id(query)
        if task_id is None:
            return
        if not os.getenv("DEEPSEEK_API_KEY"):
            self._send_json(503, {"error": "Не задан DEEPSEEK_API_KEY."})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for event in stream_day_three_experiment(self.server.ask_model, task_id):
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _read_day_three_task_id(self, query):
        if query:
            self._send_json(400, {"error": "Маршрут не принимает query-параметры."})
            return None
        if self.headers.get("Transfer-Encoding") is not None:
            self._send_json(400, {"error": "Transfer-Encoding не поддерживается."})
            return None
        content_lengths = self.headers.get_all("Content-Length", [])
        if len(content_lengths) > 1:
            self._send_json(400, {"error": "Неоднозначный Content-Length."})
            return None
        if content_lengths and (not content_lengths[0].isascii() or not content_lengths[0].isdigit()):
            self._send_json(400, {"error": "Некорректный Content-Length."})
            return None
        try:
            length = int(content_lengths[0]) if content_lengths else 0
        except ValueError:
            self._send_json(400, {"error": "Некорректный Content-Length."})
            return None
        if length > MAX_DAY_THREE_REQUEST_BYTES:
            self._send_json(400, {"error": "Тело запроса слишком большое."})
            return None
        task_id = "server-messages"
        if length:
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "Некорректный JSON."})
                return None
            if not isinstance(data, dict) or set(data) != {"taskId"} or not isinstance(data["taskId"], str):
                self._send_json(400, {"error": "Запрос должен содержать только известный taskId."})
                return
            task_id = data["taskId"]
            if task_id not in DAY_THREE_TASKS:
                self._send_json(400, {"error": "Запрос должен содержать только известный taskId."})
                return None
        return task_id

    def _handle_bulk_create(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Некорректный JSON."})
            return
        if not isinstance(data, dict) or set(data) != {"count"}:
            self._send_json(400, {"error": "Количество агентов должно быть целым числом от 1 до 100."})
            return
        try:
            agents = self.server.registry.create_many(data["count"])
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        self._send_json(201, {"agents": agents})

    def _handle_message(self, agent_id):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Некорректный JSON."})
            return
        if not isinstance(data, dict) or "text" not in data or not isinstance(data["text"], str):
            self._send_json(400, {"error": "Поле text должно быть непустой строкой."})
            return
        temperature = data.get("temperature", DEFAULT_TEMPERATURE)
        agent = self.server.registry.get(agent_id)
        if not agent:
            self._send_json(404, {"error": "Агент не найден."})
            return
        try:
            snapshot = agent.respond(data["text"], temperature)
        except MissingApiKeyError as error:
            snapshot = agent.snapshot()
            self._send_json(503, {
                "agent": snapshot,
                "error": f"Не задан {error.key_name}.",
                "metadata": snapshot["metadata"],
            })
            return
        except ValueError as error:
            if str(error) in {"Пустое сообщение.", "Поле temperature должно быть числом от 0 до 2."}:
                self._send_json(400, {"error": str(error)})
                return
            snapshot = agent.snapshot()
            self._send_json(502, {
                "agent": snapshot,
                "error": "Не удалось получить ответ DeepSeek.",
                "metadata": snapshot["metadata"],
            })
            return
        except Exception:
            snapshot = agent.snapshot()
            self._send_json(502, {
                "agent": snapshot,
                "error": "Не удалось получить ответ DeepSeek.",
                "metadata": snapshot["metadata"],
            })
            return
        self._send_json(200, {"agent": snapshot, "metadata": snapshot["metadata"]})

    def _handle_settings(self, agent_id):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Некорректный JSON."})
            return
        settings, error = self._validate_settings(data)
        if error:
            self._send_json(400, {"error": error})
            return
        agent = self.server.registry.get(agent_id)
        if not agent:
            self._send_json(404, {"error": "Агент не найден."})
            return
        self._send_json(200, {"agent": agent.update_settings(settings)})

    def _validate_settings(self, data):
        expected_fields = {"model", "systemPrompt", "format", "maxTokens", "stop"}
        if not isinstance(data, dict) or set(data) != expected_fields:
            return None, "Настройки имеют неверный формат."
        system_prompt = data["systemPrompt"]
        model = data["model"]
        response_format = data["format"]
        max_tokens = data["maxTokens"]
        stop = data["stop"]
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            return None, "System prompt не может быть пустым."
        if model not in MODELS:
            return None, "Неизвестная модель."
        if response_format not in {"text", "json"}:
            return None, "Формат ответа должен быть text или json."
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0):
            return None, "Максимум токенов должен быть положительным целым числом."
        if not isinstance(stop, str):
            return None, "Стоп-последовательность должна быть строкой."
        return {
            "model": model,
            "systemPrompt": system_prompt.strip(),
            "format": response_format,
            "maxTokens": max_tokens,
            "stop": stop.strip(),
        }, None

    def _send_json(self, status, body):
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_html(self, status, body):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ChatServer(ThreadingHTTPServer):
    def __init__(self, address, ask_model):
        super().__init__(address, ChatRequestHandler)
        self.registry = AgentRegistry(ask_model)
        self.ask_model = ask_model


def create_server(host="127.0.0.1", port=8000, ask_model=ask_deepseek):
    return ChatServer((host, port), ask_model)


def main():
    server = create_server()
    print("Откройте http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
