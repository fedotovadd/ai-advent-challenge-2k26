const MemoryControls = (() => {
  function mount(deps) {
    const {activeAgent} = deps;
    const panel = document.createElement('section');
    panel.id = 'memory-controls'; panel.className = 'context-controls';
    panel.innerHTML = `<h2>Память</h2>
      <details open><summary>Краткосрочная память</summary><pre id="memory-short" class="meta-value">—</pre></details>
      <details open><summary>Рабочая память</summary><pre id="memory-working" class="meta-value">—</pre></details>
      <details><summary>Долговременная память</summary><pre id="memory-long-term" class="meta-value">—</pre></details>
      <details><summary>Последний prompt по слоям</summary><pre id="memory-trace" class="meta-value">—</pre></details>`;
    document.querySelector('#settings-form').before(panel);
    const el = id => panel.querySelector('#' + id);
    function refresh() { const agent=activeAgent(); if (!agent) { el('memory-short').textContent='—'; el('memory-working').textContent='—'; el('memory-long-term').textContent='—'; el('memory-trace').textContent='—'; return; } const layers=agent.context.memoryLayers; el('memory-short').textContent=`${agent.messages.length} сообщений · стратегия: ${agent.settings.contextStrategy}`; el('memory-working').textContent=JSON.stringify(layers.working,null,2); el('memory-long-term').textContent=JSON.stringify(layers.longTerm,null,2); el('memory-trace').textContent=agent.metadata?.memoryLayers ? JSON.stringify(agent.metadata.memoryLayers,null,2) : 'Пока нет отправленного prompt.'; }
    return {refresh};
  }
  return {mount};
})();
