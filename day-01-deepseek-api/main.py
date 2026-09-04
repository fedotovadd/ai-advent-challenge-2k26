import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from openai import OpenAI


MODEL = "deepseek-v4-flash"
SYSTEM_PROMPT = "Ты полезный AI-помощник. Отвечай ясно, практично и по-русски."
JSON_OUTPUT_INSTRUCTION = "Верни только валидный JSON без Markdown-разметки."
DEFAULT_SETTINGS = {
    "systemPrompt": SYSTEM_PROMPT,
    "format": "text",
    "maxTokens": None,
    "stop": "",
}

PAGE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepSeek — чат</title>
  <style>
    :root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color:#3d2922; background:#f8f2ed; }
    * { box-sizing:border-box; } body { margin:0; min-height:100vh; background:#f8f2ed; } button, textarea { font:inherit; } button { cursor:pointer; }
    .app { min-height:100vh; display:grid; grid-template-columns:240px minmax(0,1fr) 300px; background:#fffdfa; }
    .sidebar, .metadata { padding:20px 14px; background:#fbf5f0; } .sidebar { border-right:1px solid #ecdcd2; display:flex; flex-direction:column; } .metadata { border-left:1px solid #ecdcd2; overflow:auto; }
    .brand { display:flex; align-items:center; gap:9px; padding:4px 8px 24px; font-size:18px; font-weight:750; color:#3a251e; } .mark { display:grid; place-items:center; width:28px; height:28px; border-radius:9px; color:#fff; background:linear-gradient(135deg,#9d3f2f,#d77951); }
    .new-chat, .send { border:0; border-radius:10px; background:#983d2d; color:#fffaf6; font-weight:650; } .new-chat { padding:11px 12px; text-align:left; margin-bottom:22px; } .new-chat:hover, .send:hover { background:#813224; }
    .sessions-label { margin:0 8px 8px; color:#a39288; font-size:11px; text-transform:uppercase; letter-spacing:.08em; } #session-list { display:grid; gap:3px; }
    .session { width:100%; border:0; border-radius:9px; padding:10px 9px; text-align:left; background:transparent; color:#79675e; font-size:13px; } .session:hover { background:#f3e7df; } .session.active { background:#f3dfd4; color:#7c3529; font-weight:700; }
    .session-date { display:block; margin-top:3px; color:#a8988f; font-size:10px; font-weight:400; } .account { margin-top:auto; padding:15px 8px 2px; border-top:1px solid #ebddd6; color:#7c6a60; font-size:12px; }
    .chat { display:flex; min-width:0; flex-direction:column; background:#fffdfa; } header { height:68px; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:0 28px; border-bottom:1px solid #f0e6e0; } #chat-title { color:#402a22; font-size:16px; font-weight:750; } .model { margin-top:3px; color:#a39289; font-size:11px; }
    #status { border-radius:99px; padding:6px 10px; background:#f7e6dc; color:#913d2d; font-size:11px; } #status[data-kind="error"] { background:#fae3df; color:#a72c23; }
    .thread { flex:1; overflow:auto; padding:32px clamp(18px,8vw,100px); } .message { display:flex; gap:10px; max-width:88%; margin-bottom:22px; } .message.user { margin-left:auto; justify-content:flex-end; }
    .avatar { display:grid; place-items:center; flex:0 0 30px; width:30px; height:30px; border-radius:9px; color:#fff; background:linear-gradient(135deg,#9d3f2f,#d77951); font-size:12px; font-weight:750; } .bubble { color:#5d4940; font-size:14px; line-height:1.55; white-space:pre-wrap; } .message.user .bubble { padding:11px 14px; border-radius:14px 14px 3px 14px; color:#5e382f; background:#f8e7de; } .author { margin-bottom:3px; color:#4a3027; font-size:11px; font-weight:700; }
    .composer-area { padding:0 28px 20px; } .composer { display:flex; align-items:end; gap:10px; padding:8px 9px 8px 14px; border:1px solid #e7d9d1; border-radius:14px; background:#fff; } #message-input { min-height:28px; max-height:120px; flex:1; resize:vertical; border:0; outline:0; color:#4d352c; background:transparent; } #message-input::placeholder { color:#ad9d94; } .send { min-width:100px; padding:10px 12px; } .send:disabled { cursor:wait; opacity:.65; } .hint { margin-top:8px; color:#ae9e95; text-align:center; font-size:11px; }
    .metadata-title { display:flex; align-items:center; justify-content:space-between; margin:4px 3px 19px; } .metadata-title h2 { margin:0; color:#4a3027; font-size:14px; } .metadata-title span { color:#a34b38; font-size:10px; } .meta-card { margin-bottom:11px; overflow:hidden; border:1px solid #eaded7; border-radius:11px; background:#fffdfa; } .meta-label { padding:10px 11px; border-bottom:1px solid #f2eae5; color:#8b796e; font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; } .meta-value { min-height:40px; padding:11px; color:#5e4a40; font-size:12px; line-height:1.5; white-space:pre-wrap; word-break:break-word; } pre.meta-value { margin:0; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; }
    @media (max-width:860px) { .app { grid-template-columns:1fr; } .sidebar { border-right:0; border-bottom:1px solid #ecdcd2; } .sidebar, .metadata { padding:14px; } .brand { padding-bottom:12px; } .new-chat { margin-bottom:12px; } #session-list { display:flex; overflow:auto; } .session { min-width:150px; } .account { display:none; } .metadata { border-top:1px solid #ecdcd2; border-left:0; } header { padding:0 18px; } .thread { min-height:340px; padding:24px 18px; } .composer-area { padding:0 18px 16px; } }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar"><div class="brand"><span class="mark">D</span>DeepSeek</div><button class="new-chat" id="new-chat" type="button">＋ Новый чат</button><div class="sessions-label">Сессии</div><nav id="session-list" aria-label="Сессии"></nav><div class="account">Локальный интерфейс<br><span class="model">История хранится до перезапуска</span></div></aside>
    <main class="chat"><header><div><div id="chat-title">Новый чат</div><div class="model">DeepSeek V4 Flash</div></div><div id="status" data-kind="ready">● Готов к работе</div></header><section class="thread" id="thread" aria-live="polite"></section><div class="composer-area"><form class="composer" id="composer"><textarea id="message-input" aria-label="Сообщение" placeholder="Напишите сообщение…" required></textarea><button class="send" id="send" type="submit">Отправить</button></form><div class="hint">⌘ Enter или Ctrl Enter — отправить</div></div></main>
    <aside class="metadata"><div class="metadata-title"><h2>Метаданные</h2><span id="meta-status">ожидание</span></div><section class="meta-card"><div class="meta-label">User prompt</div><div class="meta-value" id="user-prompt">—</div></section><section class="meta-card"><div class="meta-label">System prompt</div><div class="meta-value" id="system-prompt">—</div></section><section class="meta-card"><div class="meta-label">Отправлено в API</div><pre class="meta-value" id="payload">—</pre></section><section class="meta-card"><div class="meta-label">Статус</div><div class="meta-value" id="api-status">—</div></section></aside>
  </div>
  <script>
    const state = { sessions: [], activeId: null, metadata: null };
    const elements = { sessions:document.querySelector("#session-list"), title:document.querySelector("#chat-title"), thread:document.querySelector("#thread"), input:document.querySelector("#message-input"), send:document.querySelector("#send"), status:document.querySelector("#status"), metaStatus:document.querySelector("#meta-status"), userPrompt:document.querySelector("#user-prompt"), systemPrompt:document.querySelector("#system-prompt"), payload:document.querySelector("#payload"), apiStatus:document.querySelector("#api-status") };
    async function api(url, options={}) { const response=await fetch(url,options); const body=await response.json().catch(()=>({error:"Сервер вернул некорректный ответ."})); return {response,body}; }
    function activeSession() { return state.sessions.find((session)=>session.id===state.activeId); }
    function setStatus(text, kind="ready") { elements.status.textContent=text; elements.status.dataset.kind=kind; }
    function renderSessions() { elements.sessions.replaceChildren(); state.sessions.forEach((session)=>{ const button=document.createElement("button"); button.type="button"; button.className="session"+(session.id===state.activeId?" active":""); const title=document.createElement("span"); title.textContent=session.title; const detail=document.createElement("span"); detail.className="session-date"; detail.textContent=session.messages.length+" сообщений"; button.append(title,detail); button.addEventListener("click",()=>{state.activeId=session.id; render(); elements.input.focus();}); elements.sessions.append(button); }); }
    function renderThread() { elements.thread.replaceChildren(); const session=activeSession(); elements.title.textContent=session?session.title:"Новый чат"; if(!session||!session.messages.length){const empty=document.createElement("div"); empty.className="bubble"; empty.textContent="Начните диалог — первое сообщение станет названием сессии."; elements.thread.append(empty); return;} session.messages.forEach((message)=>{const row=document.createElement("article"); row.className="message "+message.role; if(message.role==="assistant"){const avatar=document.createElement("div"); avatar.className="avatar"; avatar.textContent="D"; row.append(avatar);} const content=document.createElement("div"); content.className="bubble"; if(message.role==="assistant"){const author=document.createElement("div"); author.className="author"; author.textContent="DeepSeek"; content.append(author);} const text=document.createElement("div"); text.textContent=message.content; content.append(text); row.append(content); elements.thread.append(row);}); elements.thread.scrollTop=elements.thread.scrollHeight; }
    function renderMetadata() { const metadata=state.metadata; elements.userPrompt.textContent=metadata?metadata.userPrompt:"—"; elements.systemPrompt.textContent=metadata?metadata.systemPrompt:"—"; elements.payload.textContent=metadata?JSON.stringify(metadata.payload,null,2):"—"; elements.apiStatus.textContent=metadata?metadata.status.label:"—"; elements.metaStatus.textContent=metadata?(metadata.status.kind==="success"?"последний запрос":"ошибка"):"ожидание"; }
    function render() { renderSessions(); renderThread(); renderMetadata(); }
    function upsertSession(session) { const index=state.sessions.findIndex((item)=>item.id===session.id); if(index===-1) state.sessions.push(session); else state.sessions[index]=session; state.activeId=session.id; }
    async function loadSessions() { const {response,body}=await api("/api/sessions"); if(!response.ok) throw new Error(body.error||"Не удалось загрузить сессии."); state.sessions=body.sessions; state.activeId=state.sessions[0]?state.sessions[0].id:null; render(); }
    document.querySelector("#new-chat").addEventListener("click",async()=>{const {body}=await api("/api/sessions",{method:"POST"}); if(body.session){upsertSession(body.session); state.metadata=null; render(); elements.input.focus();}else setStatus(body.error||"Не удалось создать сессию.","error");});
    document.querySelector("#composer").addEventListener("submit",async(event)=>{event.preventDefault(); const text=elements.input.value.trim(); if(!text||!state.activeId)return; elements.send.disabled=true; setStatus("● Ожидаем ответ DeepSeek…"); try { const {response,body}=await api("/api/sessions/"+state.activeId+"/messages",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text})}); if(body.session)upsertSession(body.session); if(body.metadata)state.metadata=body.metadata; if(response.ok){elements.input.value="";setStatus("● Ответ получен");}else setStatus(body.error||"Не удалось получить ответ.","error"); render(); } catch(error) { setStatus("Не удалось соединиться с локальным сервером.","error"); } finally { elements.send.disabled=false; elements.input.focus(); }});
    elements.input.addEventListener("keydown",(event)=>{if((event.metaKey||event.ctrlKey)&&event.key==="Enter")document.querySelector("#composer").requestSubmit();});
    loadSessions().catch((error)=>setStatus(error.message,"error"));
  </script>
</body>
</html>
"""


class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {
            "session-1": {
                "id": "session-1",
                "title": "Новый чат",
                "messages": [],
                "settings": copy.deepcopy(DEFAULT_SETTINGS),
            }
        }
        self._next_id = 2

    def sessions(self):
        with self._lock:
            return copy.deepcopy(list(self._sessions.values()))

    def create(self):
        with self._lock:
            session_id = f"session-{self._next_id}"
            self._next_id += 1
            session = {
                "id": session_id,
                "title": "Новый чат",
                "messages": [],
                "settings": copy.deepcopy(DEFAULT_SETTINGS),
            }
            self._sessions[session_id] = session
            return copy.deepcopy(session)

    def update_settings(self, session_id, settings):
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            session["settings"] = copy.deepcopy(settings)
            return copy.deepcopy(session)

    def add_user_message(self, session_id, text):
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            if not session["messages"]:
                session["title"] = text[:40]
            session["messages"].append({"role": "user", "content": text})
            return copy.deepcopy(session)

    def add_assistant_message(self, session_id, text):
        with self._lock:
            session = self._sessions[session_id]
            session["messages"].append({"role": "assistant", "content": text})
            return copy.deepcopy(session)


def ask_deepseek(payload, **options):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("missing API key")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(model=payload["model"], messages=payload["messages"], **options)
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("empty API response")
    return response.choices[0].message.content


class ChatRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(200, PAGE)
        elif path == "/api/sessions":
            self._send_json(200, {"sessions": self.server.store.sessions()})
        elif path.startswith("/api/"):
            self._send_json(404, {"error": "Маршрут не найден."})
        else:
            self._send_html(404, "<h1>Страница не найдена</h1>")

    def do_POST(self):
        if not self._same_origin():
            self._send_json(403, {"error": "Запрос с другого источника запрещён."})
            return
        path = urlparse(self.path).path
        if path == "/api/sessions":
            self._send_json(201, {"session": self.server.store.create()})
            return
        parts = path.split("/")
        if len(parts) == 5 and parts[:3] == ["", "api", "sessions"] and parts[4] == "messages":
            self._handle_message(parts[3])
            return
        self._send_json(404, {"error": "Маршрут не найден."})

    def do_PUT(self):
        if not self._same_origin():
            self._send_json(403, {"error": "Запрос с другого источника запрещён."})
            return
        parts = urlparse(self.path).path.split("/")
        if len(parts) == 5 and parts[:3] == ["", "api", "sessions"] and parts[4] == "settings":
            self._handle_settings(parts[3])
            return
        self._send_json(404, {"error": "Маршрут не найден."})

    def _same_origin(self):
        origin = self.headers.get("Origin")
        return not origin or origin == f"http://{self.headers.get('Host')}"

    def _handle_message(self, session_id):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "Некорректный JSON."})
            return
        if "text" not in data or not isinstance(data["text"], str):
            self._send_json(400, {"error": "Поле text должно быть непустой строкой."})
            return
        text = data["text"].strip()
        if not text:
            self._send_json(400, {"error": "Пустое сообщение."})
            return
        session = self.server.store.add_user_message(session_id, text)
        if not session:
            self._send_json(404, {"error": "Сессия не найдена."})
            return
        settings = session["settings"]
        system_prompt = settings["systemPrompt"]
        options = {}
        if settings["format"] == "json":
            system_prompt = f"{system_prompt}\n\n{JSON_OUTPUT_INSTRUCTION}"
            options["response_format"] = {"type": "json_object"}
        if settings["maxTokens"] is not None:
            options["max_tokens"] = settings["maxTokens"]
        if settings["stop"]:
            options["stop"] = settings["stop"]
        payload = {"model": MODEL, "messages": [{"role": "system", "content": system_prompt}, *session["messages"]]}
        metadata = {
            "userPrompt": text,
            "systemPrompt": system_prompt,
            "payload": {**payload, **options},
            "status": {"kind": "success", "label": "200 OK"},
        }
        if not os.getenv("DEEPSEEK_API_KEY"):
            metadata["status"] = {"kind": "error", "label": "Ошибка API"}
            self._send_json(503, {"session": session, "error": "Не задан DEEPSEEK_API_KEY.", "metadata": metadata})
            return
        try:
            answer = self.server.ask_model(payload, **options)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("empty API response")
        except Exception:
            metadata["status"] = {"kind": "error", "label": "Ошибка API"}
            self._send_json(502, {"session": session, "error": "Не удалось получить ответ DeepSeek.", "metadata": metadata})
            return
        session = self.server.store.add_assistant_message(session_id, answer)
        self._send_json(200, {"session": session, "metadata": metadata})

    def _handle_settings(self, session_id):
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
        session = self.server.store.update_settings(session_id, settings)
        if not session:
            self._send_json(404, {"error": "Сессия не найдена."})
            return
        self._send_json(200, {"session": session})

    def _validate_settings(self, data):
        expected_fields = {"systemPrompt", "format", "maxTokens", "stop"}
        if not isinstance(data, dict) or set(data) != expected_fields:
            return None, "Настройки имеют неверный формат."
        system_prompt = data["systemPrompt"]
        response_format = data["format"]
        max_tokens = data["maxTokens"]
        stop = data["stop"]
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            return None, "System prompt не может быть пустым."
        if response_format not in {"text", "json"}:
            return None, "Формат ответа должен быть text или json."
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0):
            return None, "Максимум токенов должен быть положительным целым числом."
        if not isinstance(stop, str):
            return None, "Стоп-последовательность должна быть строкой."
        return {
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
        self.store = SessionStore()
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
