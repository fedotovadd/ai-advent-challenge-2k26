/* Controls for per-agent memory and explicit dialogue branches. */
const ContextControls = (() => {
  const descriptions = {
    sliding_window: 'Хранятся только последние N сообщений. Текущее сообщение входит в N; удалённые сообщения не восстановятся при увеличении окна.',
    facts: 'Перед каждым ответом обновляется словарь фактов отдельным запросом. В основной запрос входят facts и последние N сообщений.',
    branching: 'В запрос входит полная история активной ветки. Сохраните точку и создайте от неё два варианта; продолжайте каждый независимо.',
    summary: 'Старая история сворачивается в summary пакетами по 10 сообщений. Последние 10 сообщений сохраняются без сжатия.'
  };
  function mount(deps) {
    const {activeAgent, state, scheduleSettingsSave, flushSettingsSave, api, replaceAgent, render, setChatStatus, syncSendButton} = deps;
    const panel = document.createElement('section');
    panel.id = 'context-controls';
    panel.className = 'context-controls';
    panel.innerHTML = `
      <h2>Управление контекстом · День 10</h2>
      <label class="setting-field">Стратегия
        <select id="context-strategy" aria-label="Стратегия контекста">
          <option value="sliding_window">Sliding Window — последние N</option>
          <option value="facts">Facts — факты + последние N</option>
          <option value="branching">Branching — ветки диалога</option>
          <option value="summary">Summary — сводка истории</option>
        </select>
      </label>
      <label class="setting-field" id="window-field">Размер окна N (сообщений)
        <input id="context-window" type="number" min="1" max="100" step="1" value="10">
      </label>
      <p id="context-description" class="context-help"></p>
      <div id="branch-controls" hidden>
        <label class="setting-field">Активная ветка<select id="active-branch"></select></label>
        <label class="setting-field">Название точки или новой ветки<input id="context-name" maxlength="80" placeholder="Например: MVP без оплаты"></label>
        <button id="create-checkpoint" type="button">Сохранить checkpoint</button>
        <label class="setting-field">Начать от точки<select id="checkpoint-select"></select></label>
        <button id="create-branch" type="button">Создать ветку от точки</button>
      </div>
      <div id="facts-panel" hidden><h3>Facts · ключ — значение</h3><dl id="facts-list"></dl><p id="facts-usage" class="context-help"></p></div>
      <details id="summary-panel" hidden><summary>Текущая сводка</summary><p id="summary-content" class="context-help"></p></details>
      <p id="context-cost" class="context-help"></p>`;
    document.querySelector('#settings-form').before(panel);
    const el = (id) => panel.querySelector('#' + id);
    let busy = false;
    const branchDrafts = {};
    const strategy = el('context-strategy'), windowInput = el('context-window');
    const pending = () => busy || Boolean(state.activeId && state.pendingAgentIds.has(state.activeId));
    strategy.addEventListener('change', () => { showMode(strategy.value); scheduleSettingsSave(0); });
    windowInput.addEventListener('input', () => scheduleSettingsSave());
    function showMode(mode) {
      el('context-description').textContent = descriptions[mode] || '';
      el('window-field').hidden = !['sliding_window', 'facts'].includes(mode);
      el('branch-controls').hidden = mode !== 'branching';
      el('facts-panel').hidden = mode !== 'facts';
      el('summary-panel').hidden = mode !== 'summary';
    }
    function options(select, items, selected) {
      select.replaceChildren(...Object.entries(items).map(([id, value]) => {
        const option = document.createElement('option');
        option.value = id; option.textContent = value.name; return option;
      }));
      if (selected && items[selected]) select.value = selected;
    }
    async function action(route, data) {
      const agent = activeAgent();
      if (!agent || pending()) return;
      const agentId = agent.id, branchId = agent.context.activeBranch;
      if (!await flushSettingsSave()) return;
      if (pending() || state.activeId !== agentId || activeAgent()?.context.activeBranch !== branchId) return;
      busy = true; state.contextBusy = true; refresh(); syncSendButton();
      try {
        const {response, body} = await api('/api/agents/' + agentId + '/' + route, {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
        });
        if (!AgentClientState.agentExists(state, agentId)) return;
        if (!response.ok) throw new Error(body.error || 'Не удалось изменить ветку.');
        if (route === 'switch-branch') {
          branchDrafts[agentId + ':' + agent.context.activeBranch] = state.drafts[agentId] || '';
          state.drafts[agentId] = branchDrafts[agentId + ':' + data.branchId] || '';
          delete state.chatStatuses[agentId]; delete state.probeAttempts[agentId];
        }
        replaceAgent(body.agent);
      } catch (error) { setChatStatus(agentId, error.message || 'Нет связи с сервером.'); }
      finally { busy = false; state.contextBusy = false; render(); }
    }
    el('create-checkpoint').addEventListener('click', () => action('checkpoints', {name: el('context-name').value || 'Точка ' + (Object.keys(activeAgent()?.context.checkpoints || {}).length + 1)}));
    el('create-branch').addEventListener('click', () => action('branches', {name: el('context-name').value || 'Ветка ' + Object.keys(activeAgent()?.context.branches || {}).length, checkpointId: el('checkpoint-select').value}));
    el('active-branch').addEventListener('change', () => action('switch-branch', {branchId: el('active-branch').value}));
    function refresh() {
      const agent = activeAgent();
      panel.querySelectorAll('input, select, button').forEach(control => control.disabled = !agent || pending());
      if (!agent) return;
      strategy.value = agent.settings.contextStrategy;
      windowInput.value = agent.settings.windowSize;
      showMode(strategy.value);
      const context = agent.context;
      options(el('active-branch'), context.branches, context.activeBranch);
      options(el('checkpoint-select'), context.checkpoints, el('checkpoint-select').value);
      el('create-branch').disabled = pending() || !Object.keys(context.checkpoints).length;
      const facts = el('facts-list'); facts.replaceChildren();
      for (const [key, value] of Object.entries(context.facts)) {
        const term = document.createElement('dt'), detail = document.createElement('dd');
        term.textContent = key; detail.textContent = value; facts.append(term, detail);
      }
      if (!Object.keys(context.facts).length) facts.textContent = 'Фактов пока нет.';
      const usage = context.factsUsage, summaryUsage = context.summaryUsage;
      el('facts-usage').textContent = `Обновлений: ${usage.calls} · токенов: ${usage.totalTokens} · $${usage.usd.toFixed(6)}. ` + (usage.last ? (usage.last.source === 'actual' ? 'Последний расход: usage API.' : 'Последний расход: приблизительная оценка.') : '');
      el('summary-content').textContent = context.summary || 'Сводка ещё не создавалась.';
      const allTokens = agent.metrics.totals.totalTokens + usage.totalTokens + summaryUsage.totalTokens;
      const allUsd = agent.metrics.totals.usd + usage.usd + summaryUsage.usd;
      el('context-cost').textContent = `Все вызовы в истории этой ветки, включая память: ${allTokens} токенов · $${allUsd.toFixed(6)}. Общий префикс учитывается в каждой ветке.`;
    }
    return {refresh, values: () => ({contextStrategy: strategy.value, windowSize: Number(windowInput.value)})};
  }
  return {mount};
})();
