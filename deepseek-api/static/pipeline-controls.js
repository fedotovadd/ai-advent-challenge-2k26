/* Incremental research cards. All external values are rendered as text. */
const PipelineControls = (() => {
  const opened = new Set();
  async function consume(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', identity = null, seq = 0, terminal = null;
    function line(text) {
      if (!text.trim()) return;
      const event = JSON.parse(text);
      const key = [event.run_id, event.agent_id, event.branch_id].join(':');
      if (identity === null) identity = key;
      if (identity !== key || !Number.isInteger(event.seq)) throw new Error('Некорректный поток исследования.');
      if (event.seq <= seq) return;
      if (terminal) throw new Error('Событие после завершения исследования.');
      seq = event.seq;
      onEvent(event);
      if (event.type === 'pipeline_completed' || event.type === 'pipeline_failed') terminal = event;
    }
    try {
      while (true) {
        const chunk = await reader.read();
        buffer += decoder.decode(chunk.value || new Uint8Array(), {stream: !chunk.done});
        let end;
        while ((end = buffer.indexOf('\n')) >= 0) { line(buffer.slice(0, end)); buffer = buffer.slice(end + 1); }
        if (chunk.done) break;
        if (buffer.length > 2_000_000) throw new Error('Слишком большой ответ исследования.');
      }
      if (buffer.trim()) line(buffer);
      if (!terminal) throw new Error('Связь прервана; итог исследования неизвестен. Обновите страницу. Автоматического повтора нет.');
      return terminal;
    } finally { reader.releaseLock(); }
  }

  function stepLabel(step) {
    const data = step.result?.structuredContent;
    if (step.status === 'success') {
      if (step.name === 'search') return `✓ search — найдено ${data.items.length} репозитория`;
      if (step.name === 'summarize') return `✓ summarize — обзор по ${data.source_count} репозиториям подготовлен`;
      return '✓ saveToFile — файл сохранён';
    }
    if (step.status === 'error') return `✕ ${step.name} — ${step.error || 'Ошибка выполнения'}`;
    if (step.status === 'skipped') return `— ${step.name} — не выполнялся`;
    if (step.status === 'running') return `⏳ ${step.name} — ${{search:'ищу репозитории…', summarize:'готовлю обзор…', saveToFile:'сохраняю файл…'}[step.name]}`;
    return `○ ${step.name} — ожидает`;
  }
  function render(card) {
    const root = document.createElement('section');
    root.className = 'pipeline-card'; root.dataset.runId = card.run_id;
    root.setAttribute('aria-label', 'Выполнение исследования');
    const title = document.createElement('div'); title.className = 'pipeline-title'; title.textContent = 'Исследование · ' + card.query; root.append(title);
    if (card.interpretation) {
      const prep = document.createElement('div'); prep.className = 'pipeline-step-status'; prep.setAttribute('role', 'status');
      const intent = card.interpretation;
      prep.textContent = intent.status === 'running' ? '⏳ Понимаю запрос…' :
        intent.status === 'success' ? `✓ Запрос GitHub: ${intent.query} · до ${intent.limit} репозиториев` :
        '✕ Разбор запроса — ' + (intent.error || 'прерван');
      root.append(prep);
    }
    for (const step of card.steps) {
      const section = document.createElement('section'); section.className = 'pipeline-step'; section.dataset.status = step.status;
      const status = document.createElement('div'); status.className = 'pipeline-step-status'; status.setAttribute('role','status'); status.textContent = stepLabel(step); section.append(status);
      if (step.status !== 'pending' && step.status !== 'skipped') {
        const details = document.createElement('details'); const key = card.run_id + ':' + step.name;
        details.open = opened.has(key);
        details.addEventListener('toggle', () => { if(details.open) opened.add(key); else opened.delete(key); });
        const summary = document.createElement('summary'); summary.textContent = 'Входные данные / Результат'; details.append(summary);
        const info = document.createElement('div'); info.className = 'pipeline-info'; info.textContent = `${step.serverUrl || ''} · ${step.durationMs || 0} мс`; details.append(info);
        for (const [label, data] of [['Входные данные',step.arguments], ['Результат MCP', step.result]]) {
          if (data == null) continue;
          const heading = document.createElement('div'); heading.textContent = label;
          const pre = document.createElement('pre'); pre.textContent = JSON.stringify(data,null,2); details.append(heading,pre);
        }
        section.append(details);
      }
      root.append(section);
    }
    if (card.error) {const error=document.createElement('p'); error.className='pipeline-error'; error.textContent=card.error; root.append(error);}
    if (card.saved_file) { const file = document.createElement('p'); file.className='pipeline-file'; file.textContent='Файл: '+card.saved_file.path; root.append(file); }
    return root;
  }
  return {consume, render, stepLabel};
})();
if (typeof module !== 'undefined') module.exports = PipelineControls;
