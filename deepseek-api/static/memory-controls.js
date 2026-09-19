const MemoryControls = (() => {
  function mount(deps) {
    const {activeAgent, state, api, replaceAgent, render, setChatStatus, syncSendButton, flushSettingsSave} = deps;
    const panel = document.createElement('section');
    panel.id = 'memory-controls'; panel.className = 'context-controls';
    panel.innerHTML = `<h2>Память · День 11</h2>
      <p><strong>Краткосрочная память</strong><br><span id="memory-short"></span></p>
      <details open><summary>Рабочая память</summary>
        <label class="setting-field">Задача<input id="memory-task" maxlength="300"></label>
        <label class="setting-field">Данные JSON<textarea id="memory-working-data">{}</textarea></label>
        <button id="save-working" type="button">Сохранить рабочую память</button>
        <button id="clear-working" type="button">Очистить рабочую память</button></details>
      <details><summary>Долговременная память</summary>
        <label class="setting-field">Профиль JSON<textarea id="memory-profile">{}</textarea></label><button id="save-profile" type="button">Сохранить профиль</button>
        <label class="setting-field">Решения — по одному на строку<textarea id="memory-decisions"></textarea></label><button id="save-decisions" type="button">Сохранить решения</button>
        <label class="setting-field">Знания JSON<textarea id="memory-knowledge">{}</textarea></label><button id="save-knowledge" type="button">Сохранить знания</button></details>
      <details><summary>Последний prompt по слоям</summary><pre id="memory-trace" class="meta-value">—</pre></details>`;
    document.querySelector('#settings-form').before(panel);
    const el = id => panel.querySelector('#' + id);
    let busy = false;
    const parseObject = input => { const value = JSON.parse(input.value || '{}'); if (!value || Array.isArray(value) || typeof value !== 'object') throw new Error('Нужен JSON-объект.'); return value; };
    async function save(path, body) {
      const agent = activeAgent(); if (!agent || busy || state.pendingAgentIds.has(agent.id) || state.contextBusy) return;
      if (!await flushSettingsSave() || activeAgent()?.id !== agent.id) return;
      busy = true; refresh(); syncSendButton();
      try { const {response, body: result} = await api('/api/agents/' + agent.id + path, {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)}); if (!response.ok) throw new Error(result.error || 'Не удалось сохранить память.'); replaceAgent(result.agent); }
      catch (error) { setChatStatus(agent.id, error.message); }
      finally { busy = false; render(); syncSendButton(); }
    }
    el('save-working').addEventListener('click', () => { try { save('/memory/working', {task:el('memory-task').value.trim(), data:parseObject(el('memory-working-data'))}); } catch (error) { setChatStatus(activeAgent()?.id, error.message); } });
    el('clear-working').addEventListener('click', () => { if (window.confirm('Очистить только рабочую память? Долговременные данные сохранятся.')) save('/memory/working', {task:'', data:{}}); });
    el('save-profile').addEventListener('click', () => { try { save('/memory/long-term/profile', {entries:parseObject(el('memory-profile'))}); } catch (error) { setChatStatus(activeAgent()?.id, error.message); } });
    el('save-decisions').addEventListener('click', () => save('/memory/long-term/decisions', {entries:el('memory-decisions').value.split('\n').map(value=>value.trim()).filter(Boolean)}));
    el('save-knowledge').addEventListener('click', () => { try { save('/memory/long-term/knowledge', {entries:parseObject(el('memory-knowledge'))}); } catch (error) { setChatStatus(activeAgent()?.id, error.message); } });
    function refresh() { const agent=activeAgent(); panel.querySelectorAll('input,textarea,button').forEach(control=>control.disabled=!agent||busy||state.contextBusy||state.pendingAgentIds.has(agent.id)); if (!agent) return; const layers=agent.context.memoryLayers; el('memory-short').textContent=`${agent.messages.length} сообщений · ${agent.settings.contextStrategy}`; el('memory-task').value=layers.working.task; el('memory-working-data').value=JSON.stringify(layers.working.data,null,2); el('memory-profile').value=JSON.stringify(layers.longTerm.profile,null,2); el('memory-decisions').value=layers.longTerm.decisions.join('\n'); el('memory-knowledge').value=JSON.stringify(layers.longTerm.knowledge,null,2); el('memory-trace').textContent=agent.metadata?.memoryLayers ? JSON.stringify(agent.metadata.memoryLayers,null,2) : 'Пока нет отправленного prompt.'; }
    return {refresh};
  }
  return {mount};
})();
