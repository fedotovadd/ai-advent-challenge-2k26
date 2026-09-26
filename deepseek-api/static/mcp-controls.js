(() => {
  const template = document.querySelector("#mcp-template");
  const toggle = document.querySelector(".view-toggle");
  const dayThreeView = document.querySelector("#day-three-view");
  if (!template || !toggle || !dayThreeView) return;

  const content = template.content.cloneNode(true);
  const viewButton = content.querySelector("#view-mcp");
  const view = content.querySelector("#mcp-view");
  toggle.append(viewButton);
  dayThreeView.after(view);

  const chatButton = document.querySelector("#view-chat");
  const dayThreeButton = document.querySelector("#view-day-three");
  const chatView = document.querySelector("#chat-view");
  const metadata = document.querySelector(".metadata");
  const app = document.querySelector(".app");
  const form = document.querySelector("#mcp-connect-form");
  const connect = document.querySelector("#mcp-connect");
  const status = document.querySelector("#mcp-status");
  const tools = document.querySelector("#mcp-tools");

  function leaveMcpView() {
    view.hidden = true;
    viewButton.setAttribute("aria-pressed", "false");
  }

  function showMcpView() {
    chatView.hidden = true;
    dayThreeView.hidden = true;
    metadata.hidden = true;
    app.classList.add("day-three-active");
    view.hidden = false;
    [chatButton, dayThreeButton, viewButton].forEach((button) => {
      button.setAttribute("aria-pressed", String(button === viewButton));
    });
  }

  viewButton.addEventListener("click", showMcpView);
  [chatButton, dayThreeButton].forEach((button) => button.addEventListener("click", leaveMcpView));

  function setStatus(message, kind) {
    status.textContent = message;
    if (kind) status.dataset.kind = kind;
    else delete status.dataset.kind;
  }

  let servers = [];
  let busy = false;
  const urlInput = document.querySelector("#mcp-url");
  window.McpControls = {connected: () => servers.some(server => server.status === "connected")};

  async function request(path, options) {
    const response = await fetch(path, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || "Не удалось выполнить запрос MCP.");
    return body;
  }

  function renderServers() {
    tools.replaceChildren();
    servers.forEach(server => {
      const card = document.createElement("article");
      card.className = "method-card";
      const title = document.createElement("h2");
      const scheduler = server.url === "http://127.0.0.1:8002/mcp";
      title.textContent = scheduler ? "Планировщик GitHub · День 18" : server.url === "http://127.0.0.1:8001/mcp" ? "GitHub — наш сервер" : "MCP-сервер";
      const address = document.createElement("p");
      address.className = "mcp-server-url";
      address.textContent = server.url;
      const state = document.createElement("p");
      state.textContent = server.status === "connected" ? "Доступен всем агентам" : "Отключён";
      const action = document.createElement("button");
      action.type = "button";
      action.disabled = busy;
      action.textContent = server.status === "connected" ? "Отключить" : "Подключить";
      action.addEventListener("click", () => changeConnection(server));
      const actions = document.createElement("div");
      actions.className = "mcp-server-actions";
      actions.append(action);
      card.append(title, address, state, actions);
      if (scheduler) {
        const summaries = document.createElement("a");
        summaries.className = "mcp-action-link";
        summaries.href = "http://127.0.0.1:8002/";
        summaries.target = "_blank";
        summaries.rel = "noopener noreferrer";
        summaries.textContent = "Открыть сводки по расписанию";
        const note = document.createElement("p");
        note.textContent = "Отключение MCP не останавливает фоновые задачи. Для отмены попросите агента остановить задание.";
        actions.append(summaries);
        card.append(note);
      }
      if (server.status === "connected" && !server.tools.length) {
        const empty = document.createElement("p");
        empty.textContent = "Сервер не вернул инструменты";
        card.append(empty);
      }
      server.tools.forEach(tool => {
        const details = document.createElement("details");
        const name = document.createElement("summary");
        name.textContent = tool.name;
        const description = document.createElement("p");
        description.textContent = tool.description;
        const schema = document.createElement("pre");
        schema.textContent = JSON.stringify(tool.inputSchema, null, 2);
        details.append(name, description, schema);
        card.append(details);
      });
      tools.append(card);
    });
    window.dispatchEvent(new Event("mcp-connections-changed"));
  }

  async function refresh() {
    const body = await request("/api/mcp/servers");
    servers = body.servers || [];
    renderServers();
  }

  async function changeConnection(server, url) {
    if (busy) return;
    busy = true;
    connect.disabled = true;
    connect.textContent = "Подключение…";
    renderServers();
    setStatus(server?.status === "connected" ? "Отключаем для всех агентов…" : "Подключаем сервер…");
    try {
      if (server?.status === "connected") {
        await request("/api/mcp/servers/" + server.id + "/disconnect", {method:"POST"});
      } else {
        await request("/api/mcp/servers/connect", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({url: url || server.url})});
      }
      await refresh();
      setStatus(server?.status === "connected" ? "Сервер отключён для всех агентов" : "Подключение установлено. Доступен всем агентам", "success");
    } catch (error) {
      setStatus(error.message || "Не удалось соединиться с локальным сервером.", "error");
      await refresh().catch(() => {});
    } finally {
      busy = false;
      connect.disabled = false;
      connect.textContent = "Подключить";
      renderServers();
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await changeConnection(null, urlInput.value.trim());
  });
  viewButton.addEventListener("click", () => { if (!busy) refresh().catch(error => setStatus(error.message, "error")); });
  refresh().catch(error => setStatus(error.message, "error"));
})();
