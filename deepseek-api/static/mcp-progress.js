/* Streaming MCP call progress for normal chat turns. */
const McpProgress = (() => {
  const allowed = new Set(['mcp_tool_started', 'mcp_tool_completed', 'mcp_tool_failed', 'mcp_chat_completed', 'mcp_chat_failed']);

  async function consume(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', identity = null, seq = 0, terminal = null;
    function line(text) {
      if (!text.trim()) return;
      const event = JSON.parse(text);
      const key = [event.run_id, event.agent_id, event.branch_id].join(':');
      if (identity === null) identity = key;
      if (identity !== key || !Number.isInteger(event.seq) || !allowed.has(event.type)) throw new Error('Некорректный поток MCP.');
      if (event.seq <= seq) return;
      if (terminal) throw new Error('Событие после завершения MCP-запроса.');
      seq = event.seq;
      onEvent(event);
      if (event.type === 'mcp_chat_completed' || event.type === 'mcp_chat_failed') terminal = event;
    }
    try {
      while (true) {
        const chunk = await reader.read();
        buffer += decoder.decode(chunk.value || new Uint8Array(), {stream: !chunk.done});
        let end;
        while ((end = buffer.indexOf('\n')) >= 0) { line(buffer.slice(0, end)); buffer = buffer.slice(end + 1); }
        if (chunk.done) break;
        if (buffer.length > 2_000_000) throw new Error('Слишком большой ответ MCP.');
      }
      if (buffer.trim()) line(buffer);
      if (!terminal) throw new Error('Связь прервана; итог ответа неизвестен. Обновите страницу. Автоматического повтора нет.');
      return terminal;
    } finally { reader.releaseLock(); }
  }

  function serverLabel(url) {
    if (url === 'http://127.0.0.1:8001/mcp') return 'GitHub';
    if (url === 'http://127.0.0.1:8002/mcp') return 'Планировщик';
    if (url === 'http://127.0.0.1:8003/mcp') return 'Исследование GitHub';
    try { return new URL(url).host; } catch { return 'MCP-сервер'; }
  }

  function toolLabel(name) {
    return ({search:'поиск репозиториев', summarize:'подготовка обзора', saveToFile:'сохранение файла',
      get_repository:'загрузка репозитория', schedule_repository:'создание расписания',
      get_schedule_summary:'получение сводки расписания'})[name] || name;
  }

  function render(run) {
    const root = document.createElement('section');
    root.className = 'mcp-progress-card';
    root.setAttribute('aria-label', 'Вызовы MCP');
    const title = document.createElement('div');
    title.className = 'mcp-progress-title';
    title.textContent = run.finished ? 'Инструменты MCP · вызовы завершены' : 'Инструменты MCP · выполняется';
    root.append(title);
    for (const call of run.calls) {
      const row = document.createElement('div');
      row.className = 'mcp-progress-step';
      row.dataset.status = call.status;
      row.setAttribute('role', 'status');
      const prefix = call.status === 'running' ? '⏳' : call.status === 'success' ? '✓' : '✕';
      const suffix = call.status === 'running' ? 'выполняется…' : call.status === 'success' ? `готово · ${call.durationMs || 0} мс` : 'ошибка';
      row.textContent = `${prefix} ${serverLabel(call.serverUrl)} · ${toolLabel(call.name)} — ${suffix}`;
      root.append(row);
    }
    return root;
  }

  return {consume, render};
})();
