const AgentClientState = (() => {
  function agentExists(state, agentId) {
    return state.agents.some((agent) => agent.id === agentId);
  }

  function withoutKey(values, key) {
    const result = {...(values || {})};
    delete result[key];
    return result;
  }

  function withoutAgentId(values, agentId) {
    const ids = values instanceof Set ? [...values] : values || [];
    const remaining = ids.filter((id) => id !== agentId);
    return values instanceof Set ? new Set(remaining) : remaining;
  }

  function removeAgentState(state, agentId) {
    const removedIndex = state.agents.findIndex((agent) => agent.id === agentId);
    if (removedIndex === -1) return state;
    const agents = state.agents.filter((agent) => agent.id !== agentId);
    const fallbackAgent = agents[removedIndex] || agents[removedIndex - 1] || null;
    const activeId = state.activeId === agentId ? (fallbackAgent ? fallbackAgent.id : null) : state.activeId;
    const clearsMetadata = state.metadataAgentId === agentId;
    return {
      ...state,
      agents,
      activeId,
      drafts: withoutKey(state.drafts, agentId),
      pendingAgentIds: withoutAgentId(state.pendingAgentIds, agentId),
      chatStatuses: withoutKey(state.chatStatuses, agentId),
      pendingSettings: state.pendingSettings?.agentId === agentId ? null : state.pendingSettings,
      metadata: clearsMetadata ? null : state.metadata,
      metadataAgentId: clearsMetadata ? null : state.metadataAgentId,
    };
  }

  return {agentExists, removeAgentState};
})();

if (typeof module !== "undefined") module.exports = AgentClientState;
