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

  function renderTools(items) {
    tools.replaceChildren();
    items.forEach((tool) => {
      const card = document.createElement("article");
      card.className = "method-card";
      const name = document.createElement("h2");
      name.textContent=tool.name;
      const description = document.createElement("p");
      description.className = "method-description";
      description.textContent=tool.description;
      card.append(name, description);
      tools.append(card);
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    connect.disabled = true;
    connect.textContent = "Подключение…";
    setStatus("Получаем список инструментов…");
    try {
      const response = await fetch("/api/mcp/tools", {method:"POST"});
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.error || "Не удалось получить список MCP-инструментов.");
      renderTools(Array.isArray(body.tools) ? body.tools : []);
      setStatus(body.tools?.length ? "Подключение установлено" : "Сервер не вернул инструменты");
    } catch (error) {
      setStatus(error.message || "Не удалось соединиться с локальным сервером.", "error");
    } finally {
      connect.disabled = false;
      connect.textContent = "Подключить";
    }
  });
})();
