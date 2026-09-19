/* Global profile picker and editor. */
const ProfileControls = (() => {
  function mount({api}) {
    let data = {profiles: [], activeProfileId: null};
    let busy = false;
    const panel = document.createElement('section');
    panel.className = 'profile-controls';
    panel.innerHTML = `
      <h2>Профиль пользователя</h2>
      <div class="profile-picker">
        <button class="profile-picker-trigger" id="profile-picker-trigger" type="button" aria-expanded="false" aria-controls="profile-picker-menu">
          <span id="profile-picker-label">Загрузка…</span><span class="profile-picker-chevron" aria-hidden="true">⌄</span>
        </button>
        <div class="profile-picker-menu" id="profile-picker-menu" role="menu" hidden></div>
      </div>
      <p class="profile-status" id="profile-status" role="status"></p>`;
    const backdrop = document.createElement('div');
    backdrop.className = 'profile-editor-backdrop';
    backdrop.hidden = true;
    backdrop.innerHTML = `
      <section class="profile-editor" role="dialog" aria-modal="true" aria-labelledby="profile-editor-title">
        <div class="profile-editor-heading"><h2 id="profile-editor-title"></h2><button class="profile-editor-close" type="button" aria-label="Закрыть">×</button></div>
        <form id="profile-editor-form">
          <label>Имя<input name="name" maxlength="80" required></label>
          <label>Стиль ответа<textarea name="style" maxlength="500" required></textarea></label>
          <label>Формат ответа<textarea name="format" maxlength="500" required></textarea></label>
          <label>Ограничения<textarea name="constraints" maxlength="500" required></textarea></label>
          <p class="profile-editor-error" id="profile-editor-error" role="alert"></p>
          <div class="profile-editor-actions"><button class="profile-cancel" type="button">Отмена</button><button class="profile-save" type="submit">Сохранить</button></div>
        </form>
      </section>`;
    document.querySelector('#settings-form').before(panel);
    document.body.append(backdrop);
    const trigger = panel.querySelector('#profile-picker-trigger');
    const label = panel.querySelector('#profile-picker-label');
    const menu = panel.querySelector('#profile-picker-menu');
    const status = panel.querySelector('#profile-status');
    const form = backdrop.querySelector('#profile-editor-form');
    const title = backdrop.querySelector('#profile-editor-title');
    const error = backdrop.querySelector('#profile-editor-error');
    let editedProfile = null;

    function activeProfile() {
      return data.profiles.find((profile) => profile.id === data.activeProfileId) || null;
    }

    function closeMenu() {
      menu.hidden = true;
      trigger.setAttribute('aria-expanded', 'false');
    }

    function render() {
      const selected = activeProfile();
      label.textContent = selected ? selected.name : 'Выберите профиль';
      menu.replaceChildren();
      data.profiles.forEach((profile) => {
        const item = document.createElement('button');
        item.className = 'profile-menu-item' + (profile.id === data.activeProfileId ? ' selected' : '');
        item.type = 'button'; item.role = 'menuitem';
        const check = document.createElement('span'); check.className = 'profile-menu-check'; check.textContent = profile.id === data.activeProfileId ? '✓' : '';
        const name = document.createElement('span'); name.textContent = profile.name;
        item.append(check, name);
        item.addEventListener('click', () => activateAndEdit(profile));
        menu.append(item);
      });
      const create = document.createElement('button');
      create.className = 'profile-menu-create'; create.type = 'button'; create.role = 'menuitem'; create.textContent = '+  Новый профиль';
      create.addEventListener('click', () => openEditor(null));
      menu.append(create);
    }

    async function load() {
      const {response, body} = await api('/api/profiles');
      if (!response.ok) throw new Error(body.error || 'Не удалось загрузить профили.');
      data = body; render();
    }

    function openEditor(profile) {
      closeMenu(); status.textContent = ''; error.textContent = '';
      editedProfile = profile;
      title.textContent = profile ? 'Редактировать профиль' : 'Новый профиль';
      ['name', 'style', 'format', 'constraints'].forEach((key) => { form.elements[key].value = profile ? profile[key] : ''; });
      backdrop.hidden = false;
      form.elements.name.focus();
    }

    function closeEditor() {
      if (busy) return;
      backdrop.hidden = true; editedProfile = null; error.textContent = '';
    }

    async function activateAndEdit(profile) {
      if (busy) return;
      busy = true; status.textContent = '';
      try {
        const {response, body} = await api('/api/profiles/' + profile.id + '/activate', {method: 'POST'});
        if (!response.ok) throw new Error(body.error || 'Не удалось выбрать профиль.');
        data = {...data, activeProfileId: body.activeProfileId};
        render(); openEditor(data.profiles.find((item) => item.id === profile.id));
      } catch (requestError) { status.textContent = requestError.message || 'Нет связи с сервером.'; }
      finally { busy = false; }
    }

    trigger.addEventListener('click', () => {
      const open = menu.hidden; menu.hidden = !open;
      trigger.setAttribute('aria-expanded', String(open));
    });
    panel.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeMenu(); });
    backdrop.addEventListener('click', (event) => { if (event.target === backdrop) closeEditor(); });
    backdrop.querySelector('.profile-editor-close').addEventListener('click', closeEditor);
    backdrop.querySelector('.profile-cancel').addEventListener('click', closeEditor);
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && !backdrop.hidden) closeEditor(); });
    form.addEventListener('submit', async (event) => {
      event.preventDefault(); if (busy) return;
      const values = Object.fromEntries(new FormData(form).entries());
      busy = true; error.textContent = '';
      backdrop.querySelectorAll('button, input, textarea').forEach((control) => { control.disabled = true; });
      try {
        const route = editedProfile ? '/api/profiles/' + editedProfile.id : '/api/profiles';
        const {response, body} = await api(route, {method: editedProfile ? 'PUT' : 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(values)});
        if (!response.ok) throw new Error(body.error || 'Не удалось сохранить профиль.');
        await load();
        backdrop.hidden = true; editedProfile = null;
      } catch (requestError) { error.textContent = requestError.message || 'Нет связи с сервером.'; }
      finally {
        busy = false;
        backdrop.querySelectorAll('button, input, textarea').forEach((control) => { control.disabled = false; });
      }
    });
    load().catch((loadError) => { status.textContent = loadError.message || 'Нет связи с сервером.'; });
    return {refresh: load, flushProfileSave: async () => !busy};
  }
  return {mount};
})();
