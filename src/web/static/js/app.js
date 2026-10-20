// NoAssist – Hauptlogik der Oberfläche (Vanilla JS, ES-Module, kein Build-Schritt).
// Zuständig für: Seitenleiste und Chat-Liste, Provider-/Modellauswahl, Verlauf und
// Streaming, Eingabefeld, Theme sowie den Einstellungsdialog.

import { api, TurnStream } from './api.js';
import { h, icon, renderHistory, LiveTurnView, errorBubble, userMessage } from './render.js';

const $ = (id) => document.getElementById(id);

const els = {
  app: $('app'),
  main: document.querySelector('.main'),
  sidebar: $('sidebar'),
  sidebarClose: $('sidebar-close'),
  sidebarToggle: $('sidebar-toggle'),
  backdrop: $('backdrop'),
  newChat: $('new-chat'),
  chatList: $('chat-list'),
  openSettings: $('open-settings'),
  providerSelect: $('provider-select'),
  modelSelect: $('model-select'),
  reasoningSelect: $('reasoning-select'),
  providerDot: $('provider-dot'),
  themeToggle: $('theme-toggle'),
  messages: $('messages'),
  thread: $('thread'),
  heading: $('chat-heading'),
  composer: $('composer'),
  input: $('composer-input'),
  sendBtn: $('send-btn'),
  settings: $('settings-dialog'),
  settingsClose: $('settings-close'),
  tabs: $('settings-tabs'),
  confirm: $('confirm-dialog'),
  toasts: $('toasts'),
  announcer: $('announcer'),
};

const state = {
  settings: { theme: 'system', system_prompt: '', default_provider: null, default_model: null },
  providers: [],
  chats: [],
  chatId: null,
  chat: null,
  providerId: null,
  model: '', // '' = Standard des Providers (wird als null gesendet)
  reasoning: '', // '' = Standard des Modells (wird als null gesendet)
  busy: false,
  sending: false,
  turn: null, // laufender Turn im aktiven Chat
  finishedTurns: new Set(),
  // chatId -> {kind:'error'|'cancelled', error?, retry?, providerId?, userMessageId?} – genauere Fassung
  // des Turn-Endes für diese Sitzung; ohne Eintrag wird der Zustand aus dem Verlauf abgeleitet
  tails: new Map(),
  statusCache: new Map(), // providerId -> {at, status}
  modelsCache: new Map(), // providerId -> {at, data}
  openSeq: 0,
  modelSeq: 0,
};

const STORAGE = { theme: 'noassist.theme', draft: 'noassist.draft', sidebar: 'noassist.sidebar', reasoning: 'noassist.reasoning' };
const narrowMq = window.matchMedia('(max-width: 799.98px)');

// ---------- kleine Helfer ----------

function store(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* Speicher nicht verfügbar – egal */
  }
}

function load(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

let announceTimer = 0;
/** Kurze Meldung für Screenreader (aria-live). */
function announce(text) {
  clearTimeout(announceTimer);
  els.announcer.textContent = '';
  announceTimer = setTimeout(() => {
    els.announcer.textContent = text;
  }, 60);
}

function toast(text, kind = 'info') {
  const t = h('div', { class: 'toast', dataset: { kind } }, text);
  // Modale Dialoge liegen in der obersten Ebene – dort muss der Hinweis hinein, sonst ist er verdeckt
  const dialog = [...document.querySelectorAll('dialog[open]')].pop();
  let region = els.toasts;
  if (dialog) {
    region = dialog.querySelector(':scope > .toast-region');
    if (!region) {
      region = h('div', { class: 'toast-region' });
      dialog.append(region);
    }
  }
  region.append(t);
  announce(text);
  setTimeout(() => t.remove(), 4500);
}

let uidCounter = 0;
const uid = (prefix) => `${prefix}-${++uidCounter}`;

function providerById(id) {
  return state.providers.find((p) => p.id === id) || null;
}

const PROVIDER_TYPE_LABEL = {
  lmstudio: 'LM Studio',
  claude_agent: 'Claude (Agent SDK)',
  openai: 'OpenAI-kompatibel',
};

// ---------- Theme ----------

const THEMES = ['system', 'light', 'dark'];
const THEME_LABEL = { system: 'System', light: 'Hell', dark: 'Dunkel' };
const THEME_ICON = { system: 'monitor', light: 'sun', dark: 'moon' };

function applyTheme(theme) {
  const t = THEMES.includes(theme) ? theme : 'system';
  document.documentElement.dataset.theme = t;
  store(STORAGE.theme, t);
  const next = THEMES[(THEMES.indexOf(t) + 1) % THEMES.length];
  els.themeToggle.replaceChildren(icon(THEME_ICON[t]));
  els.themeToggle.setAttribute('aria-label', `Darstellung: ${THEME_LABEL[t]}. Wechseln zu ${THEME_LABEL[next]}`);
  els.themeToggle.title = `Darstellung: ${THEME_LABEL[t]}`;
  for (const radio of document.querySelectorAll('input[name="set-theme"]')) radio.checked = radio.value === t;
}

async function setTheme(theme) {
  applyTheme(theme);
  state.settings.theme = theme;
  try {
    const res = await api.putSettings({ theme });
    if (res && typeof res === 'object') state.settings = { ...state.settings, ...res };
  } catch (e) {
    toast(`Darstellung konnte nicht gespeichert werden: ${e.message}`, 'error');
  }
}

// ---------- Seitenleiste ----------

function setSidebar(open, { focus = false } = {}) {
  const narrow = narrowMq.matches;
  els.app.dataset.sidebar = open ? 'open' : 'closed';
  els.sidebar.inert = !open;
  // Unter 800 px ist die geöffnete Leiste modal: Hauptbereich ist dann nicht bedienbar
  els.main.inert = narrow && open;
  els.sidebarToggle.hidden = open;
  els.sidebarToggle.setAttribute('aria-expanded', String(open));
  if (!narrow) store(STORAGE.sidebar, open ? 'open' : 'closed');
  if (focus) (open ? els.newChat : els.sidebarToggle).focus();
}

function initSidebar() {
  const wideOpen = load(STORAGE.sidebar) !== 'closed';
  setSidebar(narrowMq.matches ? false : wideOpen);
  narrowMq.addEventListener('change', () => {
    setSidebar(narrowMq.matches ? false : load(STORAGE.sidebar) !== 'closed');
  });
  els.sidebarToggle.addEventListener('click', () => setSidebar(true, { focus: true }));
  els.sidebarClose.addEventListener('click', () => setSidebar(false, { focus: true }));
  els.backdrop.addEventListener('click', () => setSidebar(false, { focus: true }));
}

function closeSidebarIfNarrow() {
  if (narrowMq.matches && els.app.dataset.sidebar === 'open') setSidebar(false);
}

// ---------- Chat-Liste ----------

function chatGroups(chats) {
  const startToday = new Date();
  startToday.setHours(0, 0, 0, 0);
  const startYesterday = new Date(startToday);
  startYesterday.setDate(startYesterday.getDate() - 1);
  const groups = [
    { key: 'today', label: 'Heute', items: [] },
    { key: 'yesterday', label: 'Gestern', items: [] },
    { key: 'older', label: 'Älter', items: [] },
  ];
  for (const chat of chats) {
    const d = new Date(chat.updated_at || chat.created_at || 0);
    if (d >= startToday) groups[0].items.push(chat);
    else if (d >= startYesterday) groups[1].items.push(chat);
    else groups[2].items.push(chat);
  }
  return groups;
}

function renderChatList() {
  const nav = els.chatList;
  // Fokus innerhalb der Liste nach dem Neuaufbau wiederherstellen
  const focused = nav.contains(document.activeElement) ? document.activeElement : null;
  const focusId = focused?.closest('.chat-item')?.dataset.id;
  const focusRole = focused?.dataset.role;

  nav.replaceChildren();
  if (!state.chats.length) {
    nav.append(h('p', { class: 'chat-empty' }, 'Noch keine Chats.'));
    return;
  }
  for (const group of chatGroups(state.chats)) {
    if (!group.items.length) continue;
    const headId = `chat-group-${group.key}`;
    nav.append(
      h('h2', { class: 'chat-group-title', id: headId }, group.label),
      h('ul', { class: 'chat-group', 'aria-labelledby': headId }, group.items.map(chatItem)),
    );
  }
  if (focusId) {
    const item = nav.querySelector(`.chat-item[data-id="${CSS.escape(focusId)}"]`);
    item?.querySelector(`[data-role="${focusRole || 'link'}"]`)?.focus();
  }
}

function chatItem(chat) {
  const title = chat.title || 'Neuer Chat';
  const active = chat.id === state.chatId;
  const link = h(
    'a',
    {
      class: 'chat-link',
      href: `#chat=${encodeURIComponent(chat.id)}`,
      'aria-current': active ? 'page' : null,
      title,
      dataset: { role: 'link' },
    },
    title,
  );
  link.addEventListener('click', closeSidebarIfNarrow);
  const li = h('li', { class: `chat-item${active ? ' is-active' : ''}`, dataset: { id: chat.id } });
  const rename = h(
    'button',
    { type: 'button', class: 'icon-btn small', 'aria-label': `Chat „${title}“ umbenennen`, title: 'Umbenennen', dataset: { role: 'rename' } },
    icon('edit'),
  );
  const del = h(
    'button',
    { type: 'button', class: 'icon-btn small', 'aria-label': `Chat „${title}“ löschen`, title: 'Löschen', dataset: { role: 'delete' } },
    icon('trash'),
  );
  rename.addEventListener('click', () => startRename(li, chat));
  del.addEventListener('click', () => deleteChat(chat));
  li.append(link, h('div', { class: 'chat-actions' }, rename, del));
  return li;
}

function startRename(li, chat) {
  const input = h('input', { class: 'chat-rename', type: 'text', 'aria-label': 'Neuer Titel', maxlength: '200' });
  input.value = chat.title || '';
  li.replaceChildren(input);
  input.focus();
  input.select();
  let finished = false;
  const finish = async (save) => {
    if (finished) return;
    finished = true;
    const value = input.value.trim();
    if (save && value && value !== chat.title) {
      try {
        const res = await api.updateChat(chat.id, { title: value });
        chat.title = (res && res.title) || value;
        if (state.chat && state.chat.id === chat.id) setChatTitle(chat.title);
        announce('Chat umbenannt.');
      } catch (e) {
        toast(`Umbenennen fehlgeschlagen: ${e.message}`, 'error');
      }
    }
    renderChatList();
    els.chatList.querySelector(`.chat-item[data-id="${CSS.escape(chat.id)}"] [data-role="rename"]`)?.focus();
  };
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      finish(true);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      finish(false);
    }
  });
  input.addEventListener('blur', () => finish(true));
}

async function deleteChat(chat) {
  const ok = await confirmDialog({
    title: 'Chat löschen?',
    text: `„${chat.title || 'Neuer Chat'}“ wird endgültig gelöscht.`,
    okLabel: 'Löschen',
  });
  if (!ok) return;
  try {
    await api.deleteChat(chat.id);
  } catch (e) {
    toast(`Löschen fehlgeschlagen: ${e.message}`, 'error');
    return;
  }
  state.chats = state.chats.filter((c) => c.id !== chat.id);
  state.tails.delete(chat.id);
  if (state.chatId === chat.id) {
    setHash(null, { replace: true });
    showNewChat();
  }
  renderChatList();
  toast('Chat gelöscht.');
  els.newChat.focus();
}

async function refreshChatList() {
  try {
    state.chats = await api.listChats();
    renderChatList();
  } catch (e) {
    toast(`Chats konnten nicht geladen werden: ${e.message}`, 'error');
  }
}

// ---------- Bestätigungsdialog ----------

function confirmDialog({ title, text, okLabel = 'OK' }) {
  return new Promise((resolve) => {
    $('confirm-title').textContent = title;
    $('confirm-text').textContent = text;
    $('confirm-ok').textContent = okLabel;
    els.confirm.returnValue = '';
    els.confirm.addEventListener('close', () => resolve(els.confirm.returnValue === 'ok'), { once: true });
    els.confirm.showModal();
    // Bei destruktiven Aktionen startet der Fokus auf "Abbrechen"
    $('confirm-cancel').focus();
  });
}

// ---------- Routing über den URL-Hash (#chat=<id>) ----------

function hashChatId() {
  const m = location.hash.match(/^#chat=([^&]+)/);
  if (!m) return null;
  try {
    return decodeURIComponent(m[1]);
  } catch {
    return null;
  }
}

function setHash(id, { replace = false } = {}) {
  if (hashChatId() === id) return;
  const url = id ? `#chat=${encodeURIComponent(id)}` : location.pathname + location.search;
  if (replace) history.replaceState(null, '', url);
  else history.pushState(null, '', url);
}

async function route() {
  const id = hashChatId();
  if (id) {
    if (id !== state.chatId) await openChat(id);
  } else if (state.chatId !== null || !els.thread.hasChildNodes()) {
    showNewChat();
  }
}

// ---------- Provider und Modell (Kopfzeile) ----------

function enabledProviders() {
  return state.providers.filter((p) => p.enabled !== false);
}

function renderProviderSelect() {
  const sel = els.providerSelect;
  const list = enabledProviders();
  const opts = list.map((p) => h('option', { value: p.id }, p.name || p.id));
  if (state.providerId && !list.some((p) => p.id === state.providerId)) {
    const p = providerById(state.providerId);
    opts.push(h('option', { value: state.providerId }, p ? `${p.name} (deaktiviert)` : `${state.providerId} (nicht verfügbar)`));
  }
  if (!opts.length) opts.push(h('option', { value: '' }, 'Kein Provider'));
  sel.replaceChildren(...opts);
  sel.value = state.providerId || '';
}

function defaultModelFor(pid) {
  if (pid && pid === state.settings.default_provider && state.settings.default_model) return state.settings.default_model;
  return providerById(pid)?.default_model || '';
}

function defaultProviderId() {
  const list = enabledProviders();
  const def = state.settings.default_provider;
  if (def && list.some((p) => p.id === def)) return def;
  return list[0]?.id || null;
}

async function fetchModels(pid, { force = false } = {}) {
  const cached = state.modelsCache.get(pid);
  if (!force && cached && Date.now() - cached.at < 60_000) return cached.data;
  let data;
  try {
    data = await api.providerModels(pid);
  } catch (e) {
    data = { models: [], error: e.message };
  }
  if (!data.error) state.modelsCache.set(pid, { at: Date.now(), data });
  return data;
}

/** Modell-Auswahl füllen; der aktuelle Wert bleibt erhalten, auch wenn er nicht gelistet ist. */
function fillModelSelect(sel, pid, data, current, defaultLabel) {
  const p = providerById(pid);
  const def = p?.default_model;
  const opts = [h('option', { value: '' }, def ? `${defaultLabel} (${def})` : defaultLabel)];
  const seen = new Set(['']);
  for (const m of data.models || []) {
    if (!m || seen.has(m.id)) continue;
    seen.add(m.id);
    let label = m.label || m.id;
    if (m.loaded === false) label += ' (nicht geladen)';
    opts.push(h('option', { value: m.id, title: m.description || null }, label));
  }
  if (current && !seen.has(current)) opts.push(h('option', { value: current }, current));
  if (data.error) opts.push(h('option', { value: '__error', disabled: true }, 'Modellliste nicht verfügbar'));
  sel.replaceChildren(...opts);
  sel.value = current || '';
  sel.title = data.error ? `Modellliste nicht verfügbar: ${data.error}` : '';
}

async function loadHeaderModels(pid) {
  const seq = ++state.modelSeq;
  const sel = els.modelSelect;
  sel.replaceChildren(h('option', { value: state.model }, state.model || 'Modelle werden geladen …'));
  if (!pid) return;
  const data = await fetchModels(pid);
  if (seq !== state.modelSeq) return;
  fillModelSelect(sel, pid, data, state.model, 'Standard');
}

async function refreshStatus(pid, { force = false } = {}) {
  const dot = els.providerDot;
  const name = providerById(pid)?.name || pid || 'Provider';
  if (!pid) {
    setDot(dot, 'unknown', 'Status: kein Provider gewählt');
    return null;
  }
  const cached = state.statusCache.get(pid);
  let status;
  if (!force && cached && Date.now() - cached.at < 30_000) {
    status = cached.status;
  } else {
    setDot(dot, 'checking', `Status von ${name} wird geprüft`);
    try {
      status = await api.providerStatus(pid);
    } catch (e) {
      status = { ok: false, message: e.message };
    }
    state.statusCache.set(pid, { at: Date.now(), status });
  }
  if (state.providerId === pid) {
    const text = `${name}: ${status.ok ? 'bereit' : 'nicht bereit'}${status.message ? ` – ${status.message}` : ''}`;
    setDot(dot, status.ok ? 'ok' : 'error', text + (status.hint ? `. ${status.hint}` : ''));
  }
  return status;
}

function setDot(dot, stateName, label) {
  dot.dataset.state = stateName;
  dot.setAttribute('aria-label', label);
  dot.title = label;
}

// ---------- Reasoning-Stufe (Eingabe) ----------

const REASONING_LABEL = {
  off: 'Aus',
  minimal: 'Minimal',
  low: 'Niedrig',
  medium: 'Mittel',
  high: 'Hoch',
  xhigh: 'Sehr hoch',
  max: 'Maximal',
};

/** Stufen je Provider-Typ; leer = Provider unterstützt keine Reasoning-Stufe. */
const REASONING_LEVELS = {
  claude_agent: ['off', 'low', 'medium', 'high', 'xhigh', 'max'],
  openai: ['minimal', 'low', 'medium', 'high'],
};

function reasoningLevels(pid) {
  return REASONING_LEVELS[providerById(pid)?.type] || [];
}

function savedReasoning(pid) {
  try {
    return JSON.parse(load(STORAGE.reasoning) || '{}')[pid] || '';
  } catch {
    return '';
  }
}

function saveReasoning(pid, value) {
  let map = {};
  try {
    map = JSON.parse(load(STORAGE.reasoning) || '{}') || {};
  } catch {
    map = {};
  }
  if (value) map[pid] = value;
  else delete map[pid];
  store(STORAGE.reasoning, JSON.stringify(map));
}

function renderReasoningSelect() {
  const sel = els.reasoningSelect;
  const levels = reasoningLevels(state.providerId);
  if (!levels.includes(state.reasoning)) state.reasoning = '';
  sel.replaceChildren(
    h('option', { value: '' }, levels.length ? 'Reasoning: Standard' : 'Reasoning: –'),
    ...levels.map((l) => h('option', { value: l }, `Reasoning: ${REASONING_LABEL[l]}`)),
  );
  sel.value = state.reasoning;
  sel.hidden = !state.providerId;
  sel.disabled = state.busy || !levels.length;
  sel.title = levels.length ? 'Reasoning-Stufe' : 'Dieser Provider unterstützt keine Reasoning-Stufe.';
}

/** Provider/Modell setzen; persist=true speichert die Auswahl am aktiven Chat. */
async function selectProvider(pid, model, { persist = false } = {}) {
  state.providerId = pid;
  state.model = model ?? '';
  state.reasoning = pid ? savedReasoning(pid) : '';
  renderProviderSelect();
  renderReasoningSelect();
  refreshStatus(pid);
  const loading = loadHeaderModels(pid);
  if (persist) persistChatModel();
  await loading;
}

async function persistChatModel() {
  if (!state.chatId || state.busy) return;
  try {
    const res = await api.updateChat(state.chatId, { provider_id: state.providerId, model: state.model || null });
    if (state.chat && res) Object.assign(state.chat, { provider_id: res.provider_id, model: res.model });
  } catch (e) {
    toast(`Auswahl konnte nicht gespeichert werden: ${e.message}`, 'error');
  }
}

// ---------- Verlauf ----------

let stickToBottom = true;

function nearBottom() {
  const m = els.messages;
  return m.scrollHeight - m.scrollTop - m.clientHeight < 80;
}

function autoScroll() {
  if (stickToBottom) els.messages.scrollTop = els.messages.scrollHeight;
}

function scrollToBottom() {
  stickToBottom = true;
  autoScroll();
}

function setChatTitle(title) {
  const t = title || 'Neuer Chat';
  els.heading.textContent = t;
  document.title = state.chatId ? `${t} – NoAssist` : 'NoAssist';
}

function emptyState() {
  return h(
    'div',
    { class: 'empty-state' },
    h('h2', {}, 'Wie kann ich helfen?'),
    h('p', {}, 'Stelle eine Frage oder beschreibe eine Aufgabe.'),
  );
}

function detachTurn() {
  if (state.turn) {
    // Nur die Verbindung schließen – der Turn läuft auf dem Server weiter
    state.turn.stream.close();
    state.turn = null;
  }
  setBusy(false);
}

function showNewChat() {
  state.openSeq++;
  detachTurn();
  state.chatId = null;
  state.chat = null;
  setChatTitle(null);
  els.thread.replaceChildren(emptyState());
  renderChatList();
  // Für neue Chats gilt die Standardauswahl, außer der Nutzer hat bereits etwas gewählt
  if (!state.providerId) selectProvider(defaultProviderId(), defaultModelFor(defaultProviderId()));
}

async function openChat(id, { refresh = false } = {}) {
  const seq = ++state.openSeq;
  if (!refresh) detachTurn();
  state.chatId = id;
  renderChatList();
  let chat;
  try {
    chat = await api.getChat(id);
  } catch (e) {
    if (seq !== state.openSeq) return;
    if (e.status === 404) {
      toast('Dieser Chat existiert nicht mehr.', 'error');
      setHash(null, { replace: true });
      showNewChat();
      return;
    }
    els.thread.replaceChildren(
      errorBubble({ code: e.code, message: e.message }, { onRetry: () => openChat(id, { refresh: true }) }),
    );
    return;
  }
  if (seq !== state.openSeq) return;
  const keepScroll = refresh && !nearBottom() ? els.messages.scrollTop : null;
  state.chat = chat;
  setChatTitle(chat.title);
  const known = state.chats.find((c) => c.id === chat.id);
  if (known) Object.assign(known, { title: chat.title, updated_at: chat.updated_at });
  renderChatList();
  if (chat.provider_id && (chat.provider_id !== state.providerId || (chat.model || '') !== state.model)) {
    selectProvider(chat.provider_id, chat.model || '');
  }
  renderThread(chat);
  if (keepScroll !== null) {
    stickToBottom = false;
    els.messages.scrollTop = keepScroll;
  } else {
    scrollToBottom();
  }
}

function lastUserMessage(messages) {
  for (let i = (messages?.length || 0) - 1; i >= 0; i--) {
    if (messages[i].role === 'user') return messages[i];
  }
  return null;
}

/**
 * Endet der Verlauf ohne vollständige Antwort? Das ist der Fall, wenn zuletzt eine
 * User-Nachricht, ein Tool-Ergebnis, ein Tool-Aufruf ohne Ergebnis oder eine leere
 * Antwort gespeichert ist (Fehler, Abbruch oder Serverneustart mitten im Turn).
 */
function lacksAnswer(messages) {
  const last = messages?.[messages.length - 1];
  if (!last) return false;
  if (last.role === 'user' || last.role === 'tool') return true;
  if (last.role === 'assistant') return !String(last.content || '').trim() || (last.tool_calls || []).length > 0;
  return false;
}

function renderThread(chat) {
  let messages = chat.messages || [];
  const activeTurn = chat.active_turn_id && !state.finishedTurns.has(chat.active_turn_id) ? chat.active_turn_id : null;
  const lastUser = lastUserMessage(messages);
  // Läuft ein Turn, kommt alles nach der letzten User-Nachricht aus dem Event-Nachspielen
  if (activeTurn && lastUser) messages = messages.slice(0, messages.lastIndexOf(lastUser) + 1);

  els.thread.replaceChildren(renderHistory(messages));
  let tailShown = false;
  if (!activeTurn) {
    // Gemerkter Ausgang gilt nur, solange sich der gespeicherte Verlauf seitdem nicht geändert
    // hat – sonst wurde der Chat inzwischen (z. B. in einem anderen Fenster) weitergeführt
    const tail = state.tails.get(chat.id);
    const lastId = messages.at(-1)?.id ?? null;
    if (tail && tail.lastMessageId === undefined) tail.lastMessageId = lastId;
    const outdated = tail && ((tail.userMessageId && tail.userMessageId !== lastUser?.id) || tail.lastMessageId !== lastId);
    if (outdated) state.tails.delete(chat.id);
    else if (tail) {
      els.thread.append(tailNode(tail));
      tailShown = true;
    }
    if (!tailShown && lastUser && lacksAnswer(messages)) {
      els.thread.append(noAnswerNode(chat, lastUser));
      tailShown = true;
    }
  }
  if (!messages.length && !activeTurn && !tailShown) els.thread.append(emptyState());

  if (activeTurn) attachTurn(activeTurn, chat.id, { userMessageId: lastUser?.id, providerId: chat.provider_id });
  else setBusy(false);
}

function tailNode(tail) {
  if (tail.kind === 'cancelled') return h('p', { class: 'notice', dataset: { tail: '1' } }, 'Antwort abgebrochen.');
  const node = makeErrorBubble(tail.error, tail.retry, tail.providerId);
  node.dataset.tail = '1';
  return node;
}

/** Hinweis unter dem Verlauf, wenn auf die letzte User-Nachricht keine vollständige Antwort gespeichert ist. */
function noAnswerNode(chat, lastUser) {
  const node = makeErrorBubble(
    {
      code: 'no_answer',
      message: 'Zur letzten Nachricht ist keine vollständige Antwort gespeichert – etwa nach einem Fehler, einem Abbruch oder einem Neustart von NoAssist.',
    },
    { mode: 'turn', chatId: chat.id, userMessageId: lastUser.id },
    chat.provider_id,
  );
  node.dataset.tail = '1';
  return node;
}

/**
 * Fehlerblase mit passender Wiederholung:
 * - {mode:'turn', chatId, userMessageId}: Turn ist gescheitert, die User-Nachricht ist bereits
 *   gespeichert -> {"retry": true, "user_message_id"}, es entsteht keine neue User-Nachricht.
 * - {mode:'send', text}: Das Senden selbst ist gescheitert -> Text normal erneut senden.
 * providerId: Provider des gescheiterten Turns (Ziel von "Provider-Einstellungen").
 */
function makeErrorBubble(error, retry, providerId) {
  let onRetry = null;
  if (retry?.mode === 'turn') onRetry = () => retryTurn(retry.chatId, retry.userMessageId);
  else if (retry?.mode === 'send' && retry.text) onRetry = () => send(retry.text);
  return errorBubble(error, {
    onRetry,
    onSettings: () =>
      openSettings('providers', {
        providerId: providerId || state.providerId,
        // Bei abgelehnten Tools direkt auf den passenden Schalter
        focus: error?.code === 'tool_unsupported' ? 'supports_tools' : null,
      }),
  });
}

function clearTail(chatId) {
  state.tails.delete(chatId);
  for (const node of els.thread.querySelectorAll('[data-tail]')) node.remove();
}

// ---------- Streaming ----------

/** userMessageId: User-Nachricht, auf die der Turn antwortet (für "Erneut versuchen"). */
function attachTurn(turnId, chatId, { userMessageId = null, providerId = null } = {}) {
  const view = new LiveTurnView({ onChange: autoScroll });
  els.thread.append(view.el);
  const retry = { mode: 'turn', chatId, userMessageId };
  const turn = { id: turnId, chatId, providerId, view, retry, error: null, cancelling: false, reconnecting: false, stream: null };
  state.turn = turn;
  setBusy(true);
  connectTurn(turn, 0);
}

function connectTurn(turn, after) {
  turn.stream = new TurnStream(turn.id, {
    after,
    onEvent: (type, data) => onTurnEvent(turn, type, data),
    onGone: (lastSeq) => onTurnGone(turn, lastSeq),
    onConnection: (ok) => {
      if (state.turn !== turn) return;
      if (!ok) {
        turn.reconnecting = true;
        turn.view.status('Verbindung wird wiederhergestellt …');
      } else if (turn.reconnecting) {
        turn.reconnecting = false;
        turn.view.status('');
      }
    },
  });
  turn.stream.start();
}

function onTurnEvent(turn, type, d) {
  if (state.turn !== turn) return;
  const view = turn.view;
  switch (type) {
    case 'turn_start':
      if (d.provider_id) turn.providerId = d.provider_id;
      announce('Antwort wird erstellt.');
      break;
    case 'delta':
      view.delta(d.text);
      break;
    case 'reasoning':
      view.reasoning(d.text);
      break;
    case 'status':
      view.status(d.text);
      break;
    case 'tool_call':
      view.toolCall(d);
      break;
    case 'tool_result':
      view.toolResult(d);
      break;
    case 'message':
      view.message(d.message);
      break;
    case 'error': {
      turn.error = { code: d.code, message: d.message, hint: d.hint };
      view.error(makeErrorBubble(turn.error, turn.retry, turn.providerId));
      announce(`Fehler: ${d.message || d.code}`);
      break;
    }
    case 'done':
      finishTurn(turn, d.status || 'completed');
      break;
    default:
      break;
  }
}

async function finishTurn(turn, status) {
  turn.stream.close();
  turn.view.finish(status);
  state.finishedTurns.add(turn.id);
  if (state.turn === turn) state.turn = null;
  setBusy(false);
  const tailBase = { retry: turn.retry, providerId: turn.providerId, userMessageId: turn.retry.userMessageId };
  if (status === 'error') {
    const error = turn.error || { code: 'internal', message: 'Die Antwort wurde mit einem Fehler beendet.' };
    state.tails.set(turn.chatId, { ...tailBase, kind: 'error', error });
  } else if (status === 'cancelled') {
    state.tails.set(turn.chatId, { ...tailBase, kind: 'cancelled' });
  }
  announce(status === 'completed' ? 'Antwort fertig.' : status === 'cancelled' ? 'Antwort abgebrochen.' : 'Antwort mit Fehler beendet.');
  // Nach "done" zeigt der Client die gespeicherte Fassung
  if (state.chatId === turn.chatId) await openChat(turn.chatId, { refresh: true });
  refreshChatList();
}

/** Verbindung endgültig weg (z. B. Server neu gestartet): Zustand beim Server erfragen. */
async function onTurnGone(turn, lastSeq) {
  for (let attempt = 0; attempt < 15; attempt++) {
    if (state.turn !== turn) return;
    try {
      const chat = await api.getChat(turn.chatId);
      if (state.turn !== turn) return;
      if (chat.active_turn_id === turn.id) {
        // Turn läuft noch: ab der letzten bekannten seq neu verbinden
        connectTurn(turn, lastSeq);
      } else {
        // Turn ist beendet oder (z. B. durch einen Serverneustart) verloren. Der Ausgang ist
        // unbekannt – nicht "fertig" behaupten, sondern den gespeicherten Stand laden;
        // renderThread zeigt dann ggf. "Keine Antwort erhalten" mit "Erneut versuchen".
        turn.view.finish('lost');
        state.finishedTurns.add(turn.id);
        state.turn = null;
        setBusy(false);
        if (state.chatId === turn.chatId) await openChat(turn.chatId, { refresh: true });
        refreshChatList();
      }
      return;
    } catch {
      turn.view.status('Verbindung wird wiederhergestellt …');
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
  if (state.turn !== turn) return;
  turn.view.finish('error');
  turn.view.error(
    errorBubble(
      { code: 'stream_lost', message: 'Die Verbindung zur laufenden Antwort ist abgerissen.', hint: 'Prüfe, ob NoAssist noch läuft, und lade den Chat neu.' },
      { onRetry: () => openChat(turn.chatId, { refresh: true }) },
    ),
  );
  state.turn = null;
  setBusy(false);
}

async function cancelTurn() {
  const turn = state.turn;
  if (!turn || turn.cancelling) return;
  turn.cancelling = true;
  turn.view.status('Wird abgebrochen …');
  announce('Antwort wird abgebrochen.');
  try {
    await api.cancelTurn(turn.id);
  } catch (e) {
    turn.cancelling = false;
    turn.view.status('');
    toast(`Abbrechen fehlgeschlagen: ${e.message}`, 'error');
  }
}

// ---------- Eingabe und Senden ----------

function setBusy(busy) {
  state.busy = busy;
  const btn = els.sendBtn;
  btn.replaceChildren(icon(busy ? 'stop' : 'send', busy ? 'icon-stop' : ''));
  btn.setAttribute('aria-label', busy ? 'Antwort abbrechen' : 'Senden');
  btn.title = busy ? 'Antwort abbrechen (Esc)' : 'Senden (Enter)';
  els.providerSelect.disabled = busy;
  els.modelSelect.disabled = busy;
  renderReasoningSelect();
  updateSendState();
}

function updateSendState() {
  els.sendBtn.disabled = state.busy ? false : state.sending || !els.input.value.trim();
}

function autogrow() {
  const ta = els.input;
  ta.style.height = 'auto';
  ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`;
}

function makeTitle(text) {
  const line = text.replace(/\s+/g, ' ').trim();
  return line.length > 60 ? `${line.slice(0, 59)}…` : line;
}

function setDraft(value) {
  try {
    if (value) sessionStorage.setItem(STORAGE.draft, value);
    else sessionStorage.removeItem(STORAGE.draft);
  } catch {
    /* egal */
  }
}

async function send(resendText) {
  const fromInput = typeof resendText !== 'string';
  const text = (fromInput ? els.input.value : resendText).trim();
  if (!text || state.busy || state.sending) return;
  state.sending = true;
  updateSendState();
  if (fromInput) {
    els.input.value = '';
    autogrow();
    setDraft('');
  }
  let chatId = state.chatId;
  try {
    if (!chatId) {
      const chat = await api.createChat({ provider_id: state.providerId, model: state.model || null, title: makeTitle(text) });
      chatId = chat.id;
      state.openSeq++;
      state.chatId = chatId;
      state.chat = { ...chat, messages: [] };
      setHash(chatId);
      setChatTitle(chat.title);
      state.chats = [chat, ...state.chats.filter((c) => c.id !== chat.id)];
      renderChatList();
      els.thread.replaceChildren();
    }
    clearTail(chatId);
    els.thread.querySelector('.empty-state')?.remove();
    const providerId = state.providerId;
    const res = await api.sendMessage(chatId, { content: text, provider_id: providerId, model: state.model || null, reasoning: state.reasoning || null });
    if (state.chatId !== chatId) return;
    els.thread.append(userMessage(res.user_message || { role: 'user', content: text }));
    scrollToBottom();
    attachTurn(res.turn_id, chatId, { userMessageId: res.user_message?.id || null, providerId });
    refreshChatList();
  } catch (e) {
    if (fromInput && !els.input.value) {
      els.input.value = text;
      autogrow();
    }
    if (e.status === 409) {
      toast('In diesem Chat läuft bereits eine Antwort.', 'error');
      if (chatId) openChat(chatId, { refresh: true });
    } else {
      const error = { code: e.code === 'network' ? 'network' : e.code, message: e.message };
      els.thread.querySelector('.empty-state')?.remove();
      const node = makeErrorBubble(error, { mode: 'send', text });
      node.dataset.tail = '1';
      els.thread.append(node);
      announce(`Fehler: ${e.message}`);
      scrollToBottom();
    }
  } finally {
    state.sending = false;
    updateSendState();
  }
}

/**
 * Letzte gespeicherte User-Nachricht erneut ausführen, ohne sie neu anzulegen:
 * {"retry": true, "user_message_id": <die Nachricht, die der Nutzer wiederholen will>}.
 * Ist der Chat inzwischen weitergeführt worden, lehnt der Server mit 409 "retry_stale" ab.
 */
async function retryTurn(chatId, userMessageId) {
  if (!chatId || state.busy || state.sending) return;
  // Ohne bekannte ID: die letzte User-Nachricht der angezeigten Fassung
  const messageId = userMessageId || (state.chat?.id === chatId ? lastUserMessage(state.chat.messages)?.id : null) || null;
  const providerId = state.providerId;
  state.sending = true;
  updateSendState();
  try {
    clearTail(chatId);
    const body = { provider_id: providerId, model: state.model || null, reasoning: state.reasoning || null };
    if (messageId) body.user_message_id = messageId;
    const res = await api.retryMessage(chatId, body);
    if (state.chatId !== chatId) return;
    // Die User-Blase steht schon im Verlauf – nur die neue Antwort anhängen
    attachTurn(res.turn_id, chatId, { userMessageId: res.user_message?.id || messageId, providerId });
    scrollToBottom();
    refreshChatList();
  } catch (e) {
    if (e.code === 'retry_stale') {
      // Jemand (anderes Fenster) hat weitergeschrieben: aktuellen Stand zeigen statt Altes zu wiederholen
      state.tails.delete(chatId);
      toast('Der Chat wurde inzwischen weitergeführt – aktuelle Fassung geladen.');
      if (state.chatId === chatId) {
        await openChat(chatId, { refresh: true });
        scrollToBottom();
      }
      refreshChatList();
    } else if (e.status === 409) {
      toast('In diesem Chat läuft bereits eine Antwort.', 'error');
      openChat(chatId, { refresh: true });
    } else {
      const node = makeErrorBubble({ code: e.code, message: e.message }, { mode: 'turn', chatId, userMessageId: messageId }, providerId);
      node.dataset.tail = '1';
      els.thread.append(node);
      announce(`Fehler: ${e.message}`);
      scrollToBottom();
    }
  } finally {
    state.sending = false;
    updateSendState();
  }
}

function initComposer() {
  try {
    const draft = sessionStorage.getItem(STORAGE.draft);
    if (draft) els.input.value = draft;
  } catch {
    /* egal */
  }
  autogrow();
  updateSendState();
  els.input.addEventListener('input', () => {
    autogrow();
    updateSendState();
    setDraft(els.input.value);
  });
  els.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      if (!state.busy) send();
    }
  });
  els.composer.addEventListener('submit', (e) => {
    e.preventDefault();
    if (state.busy) cancelTurn();
    else send();
  });
  els.messages.addEventListener('scroll', () => {
    stickToBottom = nearBottom();
  }, { passive: true });
}

// ---------- globale Tastatur ----------

function initKeyboard() {
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || e.defaultPrevented) return;
    if (document.querySelector('dialog[open]')) return; // Dialoge schließen sich selbst
    if (narrowMq.matches && els.app.dataset.sidebar === 'open') {
      setSidebar(false, { focus: true });
      return;
    }
    if (state.busy) {
      e.preventDefault();
      cancelTurn();
    }
  });
}

// ---------- Einstellungsdialog ----------

const TABS = ['general', 'providers', 'tools', 'agents', 'roles'];

function selectTab(name, { focus = false } = {}) {
  for (const t of TABS) {
    const tab = $(`tab-${t}`);
    const panel = $(`panel-${t}`);
    const active = t === name;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    panel.hidden = !active;
    if (active && focus) tab.focus();
  }
  if (name === 'tools') renderToolsPanel();
  if (name === 'agents') renderAgentsPanel();
}

function initTabs() {
  for (const t of TABS) $(`tab-${t}`).addEventListener('click', () => selectTab(t));
  els.tabs.addEventListener('keydown', (e) => {
    const current = TABS.findIndex((t) => $(`tab-${t}`).getAttribute('aria-selected') === 'true');
    let next = null;
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') next = (current + 1) % TABS.length;
    else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') next = (current - 1 + TABS.length) % TABS.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = TABS.length - 1;
    if (next === null) return;
    e.preventDefault();
    selectTab(TABS[next], { focus: true });
  });
}

/** providerId: diese Provider-Karte aufklappen; focus: data-field des Elements, das den Fokus bekommt. */
async function openSettings(tab = 'general', { providerId = null, focus = null } = {}) {
  try {
    const [settings, providers] = await Promise.all([api.getSettings(), api.listProviders()]);
    state.settings = { ...state.settings, ...settings };
    state.providers = providers;
  } catch (e) {
    toast(`Einstellungen konnten nicht geladen werden: ${e.message}`, 'error');
  }
  renderGeneralPanel();
  renderProvidersPanel({ openId: providerId });
  $('panel-tools').replaceChildren();
  delete $('panel-tools').dataset.loaded;
  $('panel-agents').replaceChildren();
  delete $('panel-agents').dataset.loaded;
  agentsState = { agents: [], openId: null, creating: false };
  renderRolesPanel();
  selectTab(tab);
  if (!els.settings.open) els.settings.showModal();
  const card = tab === 'providers' && providerId ? $('panel-providers').querySelector(`[data-provider="${CSS.escape(providerId)}"]`) : null;
  const target = (focus && card?.querySelector(`[data-field="${CSS.escape(focus)}"]`)) || card?.querySelector('summary');
  if (target) {
    target.focus();
    target.scrollIntoView({ block: 'nearest' });
  } else {
    $(`tab-${tab}`).focus();
  }
}

function initSettings() {
  initTabs();
  els.openSettings.addEventListener('click', () => {
    closeSidebarIfNarrow();
    openSettings('general');
  });
  els.settingsClose.addEventListener('click', () => els.settings.close());
  // Klick auf den abgedunkelten Hintergrund schließt den Dialog
  els.settings.addEventListener('click', (e) => {
    if (e.target === els.settings) els.settings.close();
  });
  els.settings.addEventListener('close', async () => {
    // Änderungen an Providern in die Kopfzeile übernehmen
    try {
      state.providers = await api.listProviders();
    } catch {
      return;
    }
    state.statusCache.clear();
    state.modelsCache.clear();
    if (!state.providerId || !providerById(state.providerId)) {
      const pid = defaultProviderId();
      selectProvider(pid, defaultModelFor(pid), { persist: true });
    } else {
      selectProvider(state.providerId, state.model);
    }
  });
}

function field(labelText, control, help) {
  const id = control.id || uid('f');
  control.id = id;
  const helpEl = help ? h('p', { class: 'field-help', id: `${id}-help` }, help) : null;
  if (helpEl) control.setAttribute('aria-describedby', `${id}-help`);
  return h('div', { class: 'field' }, h('label', { for: id }, labelText), control, helpEl);
}

function checkboxField(labelText, checked) {
  const input = h('input', { type: 'checkbox' });
  input.checked = !!checked;
  return { input, el: h('label', { class: 'choice' }, input, labelText) };
}

/** Checkbox mit Hilfetext darunter; name landet in data-field (Ziel für openSettings({focus})). */
function toggleField(labelText, checked, help, name) {
  const input = h('input', { type: 'checkbox', id: uid('f'), dataset: name ? { field: name } : null });
  input.checked = !!checked;
  const helpEl = h('p', { class: 'field-help', id: `${input.id}-help` }, help);
  input.setAttribute('aria-describedby', helpEl.id);
  return { input, el: h('div', { class: 'field' }, h('label', { class: 'choice' }, input, labelText), helpEl) };
}

function radioGroup(legend, name, options, value) {
  const inputs = [];
  const row = h('div', { class: 'choice-row' });
  for (const [val, label] of options) {
    const input = h('input', { type: 'radio', name, value: val });
    input.checked = val === value;
    inputs.push(input);
    row.append(h('label', { class: 'choice' }, input, label));
  }
  return { el: h('fieldset', {}, h('legend', {}, legend), row), inputs, value: () => inputs.find((i) => i.checked)?.value };
}

function formStatus() {
  return h('span', { class: 'form-status', role: 'status', 'aria-live': 'polite' });
}

function setFormStatus(el, text, kind = 'info') {
  el.dataset.kind = kind;
  el.textContent = text;
}

function renderGeneralPanel() {
  const panel = $('panel-general');
  const s = state.settings;

  const theme = radioGroup('Darstellung', 'set-theme', [['system', 'System'], ['light', 'Hell'], ['dark', 'Dunkel']], s.theme || 'system');
  for (const input of theme.inputs) input.addEventListener('change', () => setTheme(input.value));

  const prompt = h('textarea', { class: 'textarea', rows: '6', id: uid('sysprompt') });
  prompt.value = s.system_prompt || '';

  const provSel = h('select', { class: 'select', id: uid('defprov') });
  const provOpts = state.providers.map((p) => h('option', { value: p.id }, `${p.name || p.id}${p.enabled === false ? ' (deaktiviert)' : ''}`));
  provSel.replaceChildren(...provOpts);
  provSel.value = s.default_provider || defaultProviderId() || '';

  const modelSel = h('select', { class: 'select', id: uid('defmodel') });
  let modelSeq = 0;
  const loadModels = async (current) => {
    const seq = ++modelSeq;
    modelSel.replaceChildren(h('option', { value: current || '' }, current || 'Modelle werden geladen …'));
    if (!provSel.value) return;
    const data = await fetchModels(provSel.value);
    if (seq !== modelSeq) return;
    fillModelSelect(modelSel, provSel.value, data, current, 'Standard des Providers');
  };
  loadModels(provSel.value === s.default_provider ? s.default_model || '' : '');
  provSel.addEventListener('change', () => loadModels(''));

  const status = formStatus();
  const form = h(
    'form',
    { novalidate: true },
    h('h3', {}, 'Allgemein'),
    theme.el,
    field('System-Prompt', prompt, 'Wird jedem Chat als Anweisung mitgegeben. Leer lassen für den Standard.'),
    field('Standard-Provider', provSel, 'Gilt für neue Chats.'),
    field('Standard-Modell', modelSel),
    h('div', { class: 'form-actions' }, h('button', { type: 'submit', class: 'btn btn-primary' }, 'Speichern'), status),
  );
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setFormStatus(status, 'Wird gespeichert …');
    try {
      const res = await api.putSettings({
        theme: theme.value() || 'system',
        system_prompt: prompt.value,
        default_provider: provSel.value || null,
        default_model: modelSel.value && modelSel.value !== '__error' ? modelSel.value : null,
      });
      if (res && typeof res === 'object') state.settings = { ...state.settings, ...res };
      setFormStatus(status, 'Gespeichert.');
    } catch (err) {
      setFormStatus(status, `Speichern fehlgeschlagen: ${err.message}`, 'error');
    }
  });
  panel.replaceChildren(form);
}

// --- Provider ---

const TEMPLATES = {
  openai: { label: 'OpenAI', name: 'OpenAI', base_url: 'https://api.openai.com/v1', help: 'API-Key erforderlich.' },
  ollama: { label: 'Ollama', name: 'Ollama', base_url: 'http://localhost:11434/v1', help: 'Läuft lokal, kein API-Key nötig.' },
  openrouter: { label: 'OpenRouter', name: 'OpenRouter', base_url: 'https://openrouter.ai/api/v1', help: 'API-Key erforderlich.' },
  lmstudio_v1: { label: 'LM Studio (/v1)', name: 'LM Studio (OpenAI-kompatibel)', base_url: 'http://localhost:1234/v1', help: 'OpenAI-kompatibler Endpunkt von LM Studio, kein API-Key nötig.' },
  custom: { label: 'Benutzerdefiniert', name: '', base_url: '', help: 'Beliebiger OpenAI-kompatibler Endpunkt.' },
};

function isLocalHost(hostname) {
  const host = hostname.replace(/^\[|\]$/g, '').toLowerCase();
  return host === 'localhost' || host.endsWith('.localhost') || host === '::1' || /^127(?:\.\d{1,3}){3}$/.test(host);
}

/** Warnung, wenn ein API-Key/Chat-Inhalt unverschlüsselt an einen fremden Host ginge. */
function urlWarning(value) {
  try {
    const u = new URL(value);
    if (u.protocol === 'http:' && !isLocalHost(u.hostname)) {
      return 'Unverschlüsselte Verbindung (http) zu einem nicht lokalen Host: API-Key und Chat-Inhalte werden im Klartext übertragen. Nutze https.';
    }
  } catch {
    /* ungültige URL – wird beim Speichern gemeldet */
  }
  return '';
}

function validUrl(value) {
  try {
    const u = new URL(value);
    return u.protocol === 'http:' || u.protocol === 'https:';
  } catch {
    return false;
  }
}

function urlField(labelText, value) {
  const input = h('input', { class: 'input', type: 'url', inputmode: 'url', spellcheck: 'false', autocomplete: 'off', placeholder: 'https://…/v1' });
  input.value = value || '';
  const warn = h('p', { class: 'warning', role: 'status', 'aria-live': 'polite' });
  const update = () => {
    const text = urlWarning(input.value.trim());
    warn.textContent = text;
    warn.hidden = !text;
  };
  input.addEventListener('input', update);
  update();
  const wrap = field(labelText, input);
  wrap.append(warn);
  return { input, el: wrap };
}

function keyField(hasKey) {
  const input = h('input', { class: 'input', type: 'password', autocomplete: 'new-password', spellcheck: 'false' });
  input.placeholder = hasKey ? 'Gespeichert – leer lassen, um ihn zu behalten' : 'Kein Key gespeichert';
  const wrap = field('API-Key', input, 'Der Key wird nie angezeigt. Leer lassen = unverändert.');
  let del = null;
  if (hasKey) {
    del = checkboxField('Gespeicherten Key löschen', false);
    del.input.addEventListener('change', () => {
      input.disabled = del.input.checked;
      if (del.input.checked) input.value = '';
    });
    wrap.append(del.el);
  }
  return {
    el: wrap,
    /** Wert für das Teilupdate: undefined = behalten, "" = löschen, sonst setzen */
    value: () => (del?.input.checked ? '' : input.value ? input.value : undefined),
    reset: () => {
      input.value = '';
    },
  };
}

function textField(labelText, value, attrs = {}, help) {
  const input = h('input', { class: 'input', type: 'text', spellcheck: 'false', autocomplete: 'off', ...attrs });
  input.value = value || '';
  return { input, el: field(labelText, input, help) };
}

// Wie im Backend: Name einer Umgebungsvariablen
const ENV_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;

/** Feld "API-Key aus Umgebungsvariable"; value() wirft bei ungültigem Namen. */
function envField(value) {
  const f = textField('API-Key aus Umgebungsvariable', value, { placeholder: 'z. B. OPENAI_API_KEY' }, 'Name der Variablen, die den Key enthält. Ein gespeicherter API-Key hat Vorrang.');
  return {
    el: f.el,
    value: () => {
      const v = f.input.value.trim();
      if (v && !ENV_NAME_RE.test(v)) throw new Error('Umgebungsvariable: nur Buchstaben, Ziffern und _, nicht mit einer Ziffer beginnend.');
      return v || null;
    },
  };
}

/** Feld "Modellliste" (eine ID pro Zeile). Einträge mit Label aus der Konfiguration bleiben erhalten, solange die Liste unverändert ist. */
function modelsField(models) {
  const lines = (list) => [...new Set(list.map((s) => s.trim()).filter(Boolean))];
  const initial = lines((models || []).map((m) => (typeof m === 'string' ? m : m?.id || '')));
  const input = h('textarea', { class: 'textarea textarea-short', rows: '3', spellcheck: 'false', autocomplete: 'off', placeholder: 'z. B. gpt-4o-mini' });
  input.value = initial.join('\n');
  const value = () => lines(input.value.split(/\r?\n/));
  return {
    el: field('Modellliste (optional, eine pro Zeile)', input, 'Leer lassen, um die Liste vom Server abzurufen. Sonst stehen nur diese Modelle zur Auswahl.'),
    value,
    changed: () => value().join('\n') !== initial.join('\n'),
  };
}

const TOOLS_HELP = 'Ausschalten, wenn Modell oder Server keine Tools unterstützen (Fehler „Tools werden nicht unterstützt“).';

function statusBox() {
  const line = h('p', { class: 'status-line' });
  const hint = h('p', { class: 'field-help' });
  const details = h('dl');
  const box = h('div', { class: 'status-box', role: 'status', 'aria-live': 'polite' }, line, hint, details);
  box.hidden = true;
  const show = (status, labels = {}) => {
    box.hidden = false;
    const dot = h('span', { class: 'status-dot', 'aria-hidden': 'true', dataset: { state: status.checking ? 'checking' : status.ok ? 'ok' : 'error' } });
    line.replaceChildren(dot, status.checking ? 'Status wird geprüft …' : status.message || (status.ok ? 'Bereit' : 'Nicht bereit'));
    hint.textContent = status.hint || '';
    hint.hidden = !status.hint;
    details.replaceChildren();
    for (const [key, label] of Object.entries(labels)) {
      const v = status.details?.[key];
      if (v === undefined || v === null || v === '') continue;
      details.append(h('dt', {}, label), h('dd', {}, typeof v === 'boolean' ? (v ? 'ja' : 'nein') : String(v)));
    }
    details.hidden = !details.childElementCount;
  };
  return { el: box, show };
}

async function checkStatus(pid, box, labels) {
  box.show({ checking: true });
  let status;
  try {
    status = await api.providerStatus(pid);
  } catch (e) {
    status = { ok: false, message: e.message };
  }
  state.statusCache.set(pid, { at: Date.now(), status });
  box.show(status, labels);
  return status;
}

const CLAUDE_DETAIL_LABELS = { logged_in: 'Angemeldet', auth_method: 'Methode', subscription_type: 'Abo', email: 'E-Mail' };

function providerCard(p, { open = false } = {}) {
  const details = h('details', { class: 'provider-card', dataset: { provider: p.id } });
  details.open = open;
  const summary = h(
    'summary',
    {},
    h('span', {}, p.name || p.id),
    (PROVIDER_TYPE_LABEL[p.type] || p.type) !== p.name ? h('span', { class: 'badge' }, PROVIDER_TYPE_LABEL[p.type] || p.type) : null,
    p.enabled === false ? h('span', { class: 'badge' }, 'deaktiviert') : null,
  );
  const body = h('div', { class: 'provider-body' });
  details.append(summary, body);

  const status = formStatus();
  const enabled = checkboxField('Aktiviert', p.enabled !== false);
  const box = statusBox();
  const labels = p.type === 'claude_agent' ? CLAUDE_DETAIL_LABELS : {};
  const recheck = h('button', { type: 'button', class: 'btn' }, p.type === 'claude_agent' ? 'Erneut prüfen' : 'Status prüfen');
  recheck.addEventListener('click', () => checkStatus(p.id, box, labels));

  const form = h('form', { novalidate: true });
  const actions = h('div', { class: 'form-actions' }, h('button', { type: 'submit', class: 'btn btn-primary' }, 'Speichern'), recheck);
  let collect = () => ({});

  if (p.type === 'openai') {
    const name = textField('Name', p.name);
    const url = urlField('Basis-URL', p.base_url);
    const key = keyField(!!p.has_api_key);
    const env = envField(p.api_key_env);
    const model = textField('Standard-Modell', p.default_model, { placeholder: 'z. B. gpt-4o-mini' }, 'Leer lassen, um das erste verfügbare Modell zu nutzen.');
    const models = modelsField(p.models);
    const tools = toggleField('Tools senden (Funktionsaufrufe)', p.supports_tools !== false, TOOLS_HELP, 'supports_tools');
    form.append(name.el, url.el, key.el, env.el, model.el, models.el, tools.el, h('div', { class: 'field' }, enabled.el));
    collect = () => {
      if (!name.input.value.trim()) throw new Error('Bitte einen Namen angeben.');
      if (!validUrl(url.input.value.trim())) throw new Error('Bitte eine gültige http(s)-URL angeben.');
      const data = {
        name: name.input.value.trim(),
        base_url: url.input.value.trim(),
        api_key_env: env.value(),
        default_model: model.input.value.trim() || null,
        supports_tools: tools.input.checked,
      };
      if (models.changed()) data.models = models.value();
      const k = key.value();
      if (k !== undefined) data.api_key = k;
      return data;
    };
  } else if (p.type === 'claude_agent') {
    const mode = radioGroup('Anmeldung', uid('authmode'), [['login', 'Claude-Code-Login (claude.ai)'], ['api_key', 'API-Key (Anthropic Console)']], p.auth_mode || 'login');
    const key = keyField(!!p.has_api_key);
    const model = textField('Standard-Modell', p.default_model, { placeholder: 'default' }, 'z. B. default, sonnet, opus oder haiku.');
    const loginBtn = h('button', { type: 'button', class: 'btn' }, 'Bei Claude anmelden');
    const loginInfo = h(
      'div',
      { class: 'info-box' },
      h('p', {}, '„Bei Claude anmelden“ öffnet eine Konsole mit „claude auth login“. Melde dich dort im Browser an und klicke danach auf „Erneut prüfen“.'),
      h('p', {}, 'Der claude.ai-Login ist nur für die eigene Nutzung auf diesem PC gedacht. Laut Anthropic dürfen Produkte Dritter den claude.ai-Login nicht anbieten – sollen andere NoAssist nutzen, verwende den Modus „API-Key“.'),
    );
    const loginRow = h('div', { class: 'form-actions' }, loginBtn);
    loginBtn.addEventListener('click', async () => {
      loginBtn.disabled = true;
      try {
        await api.claudeLogin();
        setFormStatus(status, 'Konsole zur Anmeldung wurde geöffnet. Danach „Erneut prüfen“ klicken.');
      } catch (e) {
        setFormStatus(status, `Anmeldung konnte nicht gestartet werden: ${e.message}`, 'error');
      } finally {
        loginBtn.disabled = false;
      }
    });
    const sync = () => {
      const isLogin = mode.value() === 'login';
      key.el.hidden = isLogin;
      loginInfo.hidden = !isLogin;
      loginRow.hidden = !isLogin;
    };
    for (const input of mode.inputs) input.addEventListener('change', sync);
    sync();
    form.append(h('div', { class: 'field' }, enabled.el), mode.el, loginInfo, key.el, model.el);
    body.append(box.el);
    collect = () => {
      const data = { auth_mode: mode.value() || 'login', default_model: model.input.value.trim() || null };
      const k = key.value();
      if (k !== undefined) data.api_key = k;
      return data;
    };
    actions.append(loginRow);
  } else {
    const model = textField('Standard-Modell', p.default_model, {}, 'Leer lassen für die Auswahl im Chat.');
    form.append(h('div', { class: 'field' }, enabled.el), model.el);
    body.append(box.el);
    collect = () => ({ default_model: model.input.value.trim() || null });
  }

  if (!p.builtin) {
    const del = h('button', { type: 'button', class: 'btn btn-danger' }, 'Löschen');
    del.addEventListener('click', async () => {
      const ok = await confirmDialog({ title: 'Provider löschen?', text: `„${p.name || p.id}“ wird aus der Konfiguration entfernt.`, okLabel: 'Löschen' });
      if (!ok) return;
      try {
        await api.deleteProvider(p.id);
        state.providers = state.providers.filter((x) => x.id !== p.id);
        details.remove();
        toast(`Provider „${p.name || p.id}“ gelöscht.`);
        $('tab-providers').focus();
      } catch (e) {
        setFormStatus(status, `Löschen fehlgeschlagen: ${e.message}`, 'error');
      }
    });
    actions.append(del);
  }
  if (p.type === 'openai') body.append(box.el);
  actions.append(status);
  form.append(actions);
  body.append(form);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    let data;
    try {
      data = { ...collect(), enabled: enabled.input.checked };
    } catch (err) {
      setFormStatus(status, err.message, 'error');
      return;
    }
    setFormStatus(status, 'Wird gespeichert …');
    try {
      const res = await api.updateProvider(p.id, data);
      const updated = res && typeof res === 'object' && res.id ? res : { ...p, ...data, has_api_key: data.api_key === undefined ? p.has_api_key : !!data.api_key };
      state.providers = state.providers.map((x) => (x.id === p.id ? updated : x));
      state.statusCache.delete(p.id);
      state.modelsCache.delete(p.id);
      const fresh = providerCard(updated, { open: true });
      details.replaceWith(fresh);
      const freshStatus = fresh.querySelector('.form-status');
      if (freshStatus) setFormStatus(freshStatus, 'Gespeichert.');
      fresh.querySelector('summary')?.focus();
    } catch (err) {
      setFormStatus(status, `Speichern fehlgeschlagen: ${err.message}`, 'error');
    }
  });

  // Status beim ersten Aufklappen laden (Claude/LM Studio sofort sichtbar, OpenAI auf Wunsch)
  let loaded = false;
  const maybeLoad = () => {
    if (details.open && !loaded && p.type !== 'openai') {
      loaded = true;
      checkStatus(p.id, box, labels);
    }
  };
  details.addEventListener('toggle', maybeLoad);
  if (open) queueMicrotask(maybeLoad);
  return details;
}

function addProviderForm() {
  const tplSel = h('select', { class: 'select', id: uid('tpl') });
  tplSel.replaceChildren(...Object.entries(TEMPLATES).map(([k, t]) => h('option', { value: k }, t.label)));
  const tplHelp = h('p', { class: 'field-help' });
  const name = textField('Name', '');
  const url = urlField('Basis-URL', '');
  const key = keyField(false);
  const env = envField('');
  const model = textField('Standard-Modell', '', { placeholder: 'optional' });
  const models = modelsField([]);
  const tools = toggleField('Tools senden (Funktionsaufrufe)', true, TOOLS_HELP, 'supports_tools');
  const status = formStatus();
  const applyTemplate = () => {
    const t = TEMPLATES[tplSel.value];
    name.input.value = t.name;
    url.input.value = t.base_url;
    url.input.dispatchEvent(new Event('input'));
    tplHelp.textContent = t.help;
  };
  tplSel.addEventListener('change', applyTemplate);
  const tplField = field('Vorlage', tplSel);
  tplField.append(tplHelp);
  const form = h(
    'form',
    { novalidate: true },
    h('h3', {}, 'Provider hinzufügen'),
    tplField,
    name.el,
    url.el,
    key.el,
    env.el,
    model.el,
    models.el,
    tools.el,
    h('div', { class: 'form-actions' }, h('button', { type: 'submit', class: 'btn btn-primary' }, 'Hinzufügen'), status),
  );
  applyTemplate();
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const n = name.input.value.trim();
    const u = url.input.value.trim();
    if (!n) return setFormStatus(status, 'Bitte einen Namen angeben.', 'error');
    if (!validUrl(u)) return setFormStatus(status, 'Bitte eine gültige http(s)-URL angeben.', 'error');
    const data = { type: 'openai', name: n, base_url: u, supports_tools: tools.input.checked };
    try {
      const envName = env.value();
      if (envName) data.api_key_env = envName;
    } catch (err) {
      return setFormStatus(status, err.message, 'error');
    }
    const k = key.value();
    if (k) data.api_key = k;
    if (model.input.value.trim()) data.default_model = model.input.value.trim();
    if (models.value().length) data.models = models.value();
    setFormStatus(status, 'Wird hinzugefügt …');
    try {
      const created = await api.addProvider(data);
      state.providers = [...state.providers, created];
      renderProvidersPanel({ openId: created.id });
      const s = $('panel-providers').querySelector(`[data-provider="${CSS.escape(created.id)}"] .form-status`);
      if (s) setFormStatus(s, `Provider „${created.name}“ hinzugefügt.`);
      $('panel-providers').querySelector(`[data-provider="${CSS.escape(created.id)}"] summary`)?.focus();
    } catch (err) {
      setFormStatus(status, `Hinzufügen fehlgeschlagen: ${err.message}`, 'error');
    }
  });
  return form;
}

function renderProvidersPanel({ openId = null } = {}) {
  const panel = $('panel-providers');
  panel.replaceChildren(
    h('h3', {}, 'Provider'),
    h('p', { class: 'panel-intro' }, 'API-Keys werden nur geschrieben, nie angezeigt. Sie liegen im Klartext in data/config.json auf diesem PC.'),
    h('div', {}, state.providers.map((p) => providerCard(p, { open: p.id === openId }))),
    addProviderForm(),
  );
}

// --- Tools ---

let toolsLoading = false;

async function renderToolsPanel() {
  const panel = $('panel-tools');
  if (toolsLoading || panel.dataset.loaded === '1') return;
  toolsLoading = true;
  panel.replaceChildren(h('h3', {}, 'Tools'), h('p', { class: 'panel-intro', role: 'status' }, 'Tools werden geladen …'));
  let data;
  try {
    data = await api.listTools();
  } catch (e) {
    toolsLoading = false;
    panel.replaceChildren(
      h('h3', {}, 'Tools'),
      errorBubble({ code: e.code, message: e.message }, { onRetry: () => renderToolsPanel() }),
    );
    return;
  }
  toolsLoading = false;
  const tools = data.tools || [];
  const servers = data.mcp_servers || [];
  const toolItem = (t) =>
    h(
      'li',
      {},
      h('div', { class: 'tool-line' }, h('code', {}, t.name), h('span', { class: 'badge' }, t.source && t.source !== 'local' ? t.source.replace(/^mcp:/, 'MCP: ') : 'lokal')),
      t.description ? h('p', {}, t.description) : null,
      t.parameters ? h('details', {}, h('summary', {}, 'Parameter'), h('pre', {}, JSON.stringify(t.parameters, null, 2))) : null,
    );
  const serverBlock = (s) => {
    const ok = s.status === 'ok' || s.status === 'running' || s.status === 'connected';
    const err = s.status === 'error' || s.status === 'failed';
    const names = (s.tools || []).map((t) => (typeof t === 'string' ? t : t?.name)).filter(Boolean);
    return h(
      'div',
      { class: 'mcp-server' },
      h('h4', {}, s.name, h('span', { class: 'badge', dataset: { kind: ok ? 'ok' : err ? 'error' : '' } }, ok ? 'verbunden' : err ? 'Fehler' : s.status || 'unbekannt')),
      s.error ? h('p', { class: 'mcp-error' }, s.error) : null,
      names.length
        ? h('ul', { class: 'tool-list' }, names.map((n) => h('li', {}, h('code', {}, n))))
        : h('p', { class: 'panel-intro' }, 'Keine Tools gemeldet.'),
    );
  };
  panel.replaceChildren(
    h('h3', {}, 'Tools'),
    h('p', { class: 'panel-intro' }, 'Nur Anzeige. Tools laufen automatisch; jeder Aufruf erscheint im Chat.'),
    h(
      'p',
      { class: 'info-box' },
      'MCP-Server werden in data/config.json (Abschnitt „mcp_servers“) eingetragen. Änderungen dort wirken erst nach einem Neustart von NoAssist.',
    ),
    tools.length ? h('ul', { class: 'tool-list' }, tools.map(toolItem)) : h('p', { class: 'panel-intro' }, 'Keine Tools verfügbar.'),
    h('h3', {}, 'MCP-Server'),
    servers.length ? h('div', {}, servers.map(serverBlock)) : h('p', { class: 'panel-intro' }, 'Keine MCP-Server konfiguriert.'),
  );
  panel.dataset.loaded = '1';
}

// --- Agenten ---

const MAX_UPLOAD_BYTES = 30 * 1024 * 1024;

function readFileBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1] || '');
    reader.onerror = () => reject(new Error(`„${file.name}“ konnte nicht gelesen werden.`));
    reader.readAsDataURL(file);
  });
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Ablagefläche mit Dateiauswahl; onFiles(FileList-Array) wird bei Drop oder Auswahl aufgerufen. */
function dropZone({ label, hint, accept, multiple, onFiles }) {
  const input = h('input', { type: 'file', class: 'visually-hidden', accept, multiple: multiple || null, tabindex: '-1' });
  const button = h('button', { type: 'button', class: 'btn' }, label);
  const zone = h('div', { class: 'dropzone' }, h('p', {}, hint), button, input);
  button.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    const files = [...input.files];
    input.value = '';
    if (files.length) onFiles(files);
  });
  let depth = 0;
  const setOver = (on) => zone.classList.toggle('is-over', on);
  zone.addEventListener('dragenter', (e) => {
    e.preventDefault();
    depth++;
    setOver(true);
  });
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  zone.addEventListener('dragleave', () => {
    depth = Math.max(0, depth - 1);
    if (!depth) setOver(false);
  });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    setOver(false);
    const files = [...(e.dataTransfer?.files || [])];
    if (files.length) onFiles(multiple ? files : files.slice(0, 1));
  });
  return zone;
}

function agentFields(agent) {
  const name = textField('Name', agent?.name, { maxlength: '100' });
  const description = h('textarea', { class: 'textarea textarea-short', rows: '3', maxlength: '2000' });
  description.value = agent?.description || '';
  const prompt = h('textarea', { class: 'textarea textarea-prompt', rows: '10', spellcheck: 'false' });
  prompt.value = agent?.prompt || '';
  return {
    name,
    description,
    prompt,
    els: [
      name.el,
      field('Beschreibung', description),
      field('Prompt', prompt, 'Anweisung des Agenten (Markdown).'),
    ],
    value: () => {
      if (!name.input.value.trim()) throw new Error('Bitte einen Namen angeben.');
      return { name: name.input.value.trim(), description: description.value, prompt: prompt.value };
    },
  };
}

function agentCard(agent, { open = false, onChange, onDelete }) {
  const details = h('details', { class: 'provider-card', dataset: { agent: agent.id } });
  details.open = open;
  details.append(
    h(
      'summary',
      {},
      h('span', {}, agent.name || agent.id),
      agent.assets.length ? h('span', { class: 'badge' }, `${agent.assets.length} ${agent.assets.length === 1 ? 'Asset' : 'Assets'}`) : null,
    ),
  );
  const body = h('div', { class: 'provider-body' });
  details.append(body);

  const status = formStatus();
  const fields = agentFields(agent);
  const form = h('form', { novalidate: true }, ...fields.els);
  const del = h('button', { type: 'button', class: 'btn btn-danger' }, 'Agent löschen');
  form.append(h('div', { class: 'form-actions' }, h('button', { type: 'submit', class: 'btn btn-primary' }, 'Speichern'), del, status));
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    let data;
    try {
      data = fields.value();
    } catch (err) {
      setFormStatus(status, err.message, 'error');
      return;
    }
    setFormStatus(status, 'Wird gespeichert …');
    try {
      onChange(await api.updateAgent(agent.id, data), { open: true });
      toast('Agent gespeichert.');
    } catch (err) {
      setFormStatus(status, `Speichern fehlgeschlagen: ${err.message}`, 'error');
    }
  });
  del.addEventListener('click', async () => {
    const ok = await confirmDialog({ title: 'Agent löschen?', text: `„${agent.name || agent.id}“ wird mit allen Assets gelöscht.`, okLabel: 'Löschen' });
    if (!ok) return;
    try {
      await api.deleteAgent(agent.id);
      onDelete(agent);
    } catch (err) {
      setFormStatus(status, `Löschen fehlgeschlagen: ${err.message}`, 'error');
    }
  });

  // Assets
  const assetStatus = formStatus();
  const list = h('ul', { class: 'asset-list' });
  if (!agent.assets.length) list.append(h('li', { class: 'field-help' }, 'Keine Assets.'));
  for (const asset of agent.assets) {
    const rm = h('button', { type: 'button', class: 'btn btn-danger btn-small', 'aria-label': `Asset ${asset.name} löschen` }, 'Löschen');
    rm.addEventListener('click', async () => {
      const ok = await confirmDialog({ title: 'Asset löschen?', text: `„${asset.name}“ wird entfernt.`, okLabel: 'Löschen' });
      if (!ok) return;
      try {
        onChange(await api.deleteAgentAsset(agent.id, asset.name), { open: true });
      } catch (err) {
        setFormStatus(assetStatus, `Löschen fehlgeschlagen: ${err.message}`, 'error');
      }
    });
    list.append(
      h(
        'li',
        {},
        h('a', { href: api.agentAssetUrl(agent.id, asset.name), download: asset.name }, asset.name),
        h('span', { class: 'field-help' }, formatSize(asset.size)),
        rm,
      ),
    );
  }
  const upload = dropZone({
    label: 'Assets hochladen',
    hint: 'Dateien hierher ziehen oder auswählen.',
    multiple: true,
    onFiles: async (files) => {
      setFormStatus(assetStatus, 'Wird hochgeladen …');
      try {
        const big = files.find((f) => f.size > MAX_UPLOAD_BYTES);
        if (big) throw new Error(`„${big.name}“ ist zu groß.`);
        const payload = await Promise.all(files.map(async (f) => ({ name: f.name, data: await readFileBase64(f) })));
        onChange(await api.addAgentAssets(agent.id, payload), { open: true });
        toast(`${files.length} ${files.length === 1 ? 'Asset' : 'Assets'} hochgeladen.`);
      } catch (err) {
        setFormStatus(assetStatus, `Hochladen fehlgeschlagen: ${err.message}`, 'error');
      }
    },
  });
  body.append(form, h('h4', {}, 'Assets'), list, upload, assetStatus);
  return details;
}

let agentsState = { agents: [], openId: null, creating: false };
let agentsRender = 0;

async function renderAgentsPanel({ force = false } = {}) {
  const panel = $('panel-agents');
  if (!force && panel.dataset.loaded === '1') return;
  const seq = ++agentsRender;
  if (!panel.firstChild) panel.replaceChildren(h('h3', {}, 'Agents'), h('p', { class: 'panel-intro', role: 'status' }, 'Agents werden geladen …'));
  try {
    agentsState.agents = await api.listAgents();
  } catch (e) {
    if (seq !== agentsRender) return;
    panel.replaceChildren(h('h3', {}, 'Agents'), errorBubble({ code: e.code, message: e.message }, { onRetry: () => renderAgentsPanel({ force: true }) }));
    return;
  }
  if (seq !== agentsRender) return;
  panel.dataset.loaded = '1';
  drawAgents();
}

function drawAgents() {
  const panel = $('panel-agents');
  const { agents, openId } = agentsState;
  const status = formStatus();

  const importZone = dropZone({
    label: 'Datei auswählen',
    hint: 'Markdown-Datei (.md) oder ZIP (.zip mit .md und Assets) hierher ziehen, um einen Agenten zu erstellen.',
    accept: '.md,.markdown,.zip',
    multiple: false,
    onFiles: async ([file]) => {
      setFormStatus(status, 'Wird importiert …');
      try {
        if (file.size > MAX_UPLOAD_BYTES) throw new Error(`„${file.name}“ ist zu groß.`);
        const created = await api.importAgent({ filename: file.name, data: await readFileBase64(file) });
        agentsState.openId = created.id;
        toast(`Agent „${created.name}“ erstellt.`);
        await renderAgentsPanel({ force: true });
        panel.querySelector(`[data-agent="${CSS.escape(created.id)}"] summary`)?.focus();
      } catch (err) {
        setFormStatus(status, `Import fehlgeschlagen: ${err.message}`, 'error');
      }
    },
  });

  const newBtn = h('button', { type: 'button', class: 'btn btn-primary' }, 'Neuer Agent');
  const createBox = h('div', { class: 'agent-create' });
  createBox.hidden = !agentsState.creating;
  const showCreate = () => {
    const fields = agentFields(null);
    const createStatus = formStatus();
    const cancel = h('button', { type: 'button', class: 'btn' }, 'Abbrechen');
    const form = h(
      'form',
      { novalidate: true },
      h('h4', {}, 'Neuer Agent'),
      ...fields.els,
      h('div', { class: 'form-actions' }, h('button', { type: 'submit', class: 'btn btn-primary' }, 'Erstellen'), cancel, createStatus),
    );
    cancel.addEventListener('click', () => {
      agentsState.creating = false;
      createBox.hidden = true;
      newBtn.focus();
    });
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      let data;
      try {
        data = fields.value();
      } catch (err) {
        setFormStatus(createStatus, err.message, 'error');
        return;
      }
      setFormStatus(createStatus, 'Wird erstellt …');
      try {
        const created = await api.createAgent(data);
        agentsState.creating = false;
        agentsState.openId = created.id;
        toast(`Agent „${created.name}“ erstellt.`);
        await renderAgentsPanel({ force: true });
        panel.querySelector(`[data-agent="${CSS.escape(created.id)}"] summary`)?.focus();
      } catch (err) {
        setFormStatus(createStatus, `Erstellen fehlgeschlagen: ${err.message}`, 'error');
      }
    });
    createBox.replaceChildren(form);
    createBox.hidden = false;
    agentsState.creating = true;
    fields.name.input.focus();
  };
  newBtn.addEventListener('click', showCreate);

  const replaceCard = (stale, updated, { open }) => {
    const card = panel.querySelector(`[data-agent="${CSS.escape(updated.id)}"]`) || stale;
    const idx = agentsState.agents.findIndex((a) => a.id === updated.id);
    if (idx >= 0) agentsState.agents[idx] = updated;
    agentsState.openId = open ? updated.id : agentsState.openId;
    const fresh = makeCard(updated, open);
    card.replaceWith(fresh);
    if (document.activeElement === document.body || !panel.contains(document.activeElement)) fresh.querySelector('summary')?.focus();
  };
  const makeCard = (agent, open) => {
    const card = agentCard(agent, {
      open,
      onChange: (updated, opts) => replaceCard(card, updated, opts),
      onDelete: (gone) => {
        agentsState.agents = agentsState.agents.filter((a) => a.id !== gone.id);
        if (agentsState.openId === gone.id) agentsState.openId = null;
        toast(`Agent „${gone.name || gone.id}“ gelöscht.`);
        drawAgents();
        $('tab-agents').focus();
      },
    });
    return card;
  };

  panel.replaceChildren(
    h('h3', {}, 'Agents'),
    h('p', { class: 'panel-intro' }, 'Vorgefertigte Agenten mit Name, Beschreibung, Prompt und Assets. Sie liegen als Ordner in data/agents.'),
    h('div', { class: 'form-actions' }, newBtn),
    createBox,
    importZone,
    status,
    agents.length
      ? h('div', { class: 'agent-list' }, agents.map((a) => makeCard(a, a.id === openId)))
      : h('p', { class: 'panel-intro' }, 'Noch keine Agenten vorhanden.'),
  );
  if (agentsState.creating && !createBox.firstChild) showCreate();
}

// --- Agent-Modelle (Provider und Modell je Agent-Art) ---

const AGENT_ROLE_DEFS = [
  { key: 'agent', label: 'Agenten', help: 'Vorgefertigte Agenten, die als Subagent oder Assistent Agent laufen. Hat Vorrang vor der Einstellung der jeweiligen Art.' },
  { key: 'subagent', label: 'Subagenten', help: 'Vom Chat gestartete Subagenten (create_subagent).' },
  { key: 'assistant', label: 'Assistent Agents', help: 'Von Subagenten gestartete Hilfsagenten (create_assistant_agent). Sie können keine weiteren Agenten starten.' },
];

function roleRow(def, settings) {
  const provSel = h('select', { class: 'select', id: uid('rprov') });
  const modelSel = h('select', { class: 'select', id: uid('rmodel') });
  provSel.replaceChildren(
    h('option', { value: '' }, 'Standard-Provider'),
    ...state.providers.map((p) => h('option', { value: p.id }, `${p.name || p.id}${p.enabled === false ? ' (deaktiviert)' : ''}`)),
  );
  provSel.value = settings[`${def.key}_provider`] || '';
  let seq = 0;
  const loadModels = async (current) => {
    const mine = ++seq;
    const pid = provSel.value;
    modelSel.disabled = !pid;
    if (!pid) {
      modelSel.replaceChildren(h('option', { value: '' }, 'Standardmodell'));
      return;
    }
    modelSel.replaceChildren(h('option', { value: current || '' }, current || 'Modelle werden geladen …'));
    const data = await fetchModels(pid);
    if (mine !== seq) return;
    fillModelSelect(modelSel, pid, data, current, 'Standard des Providers');
  };
  loadModels(settings[`${def.key}_model`] || '');
  provSel.addEventListener('change', () => loadModels(''));
  const fieldset = h(
    'fieldset',
    { class: 'role-row' },
    h('legend', {}, def.label),
    h('p', { class: 'field-help' }, def.help),
    field('Provider', provSel),
    field('Modell (optional)', modelSel),
  );
  return {
    el: fieldset,
    value: () => ({
      [`${def.key}_provider`]: provSel.value || null,
      [`${def.key}_model`]: provSel.value && modelSel.value && modelSel.value !== '__error' ? modelSel.value : null,
    }),
  };
}

function renderRolesPanel() {
  const panel = $('panel-roles');
  const s = state.settings;
  const rows = AGENT_ROLE_DEFS.map((def) => roleRow(def, s));
  const status = formStatus();
  const form = h(
    'form',
    { novalidate: true },
    h('h3', {}, 'Agent-Modelle'),
    h(
      'p',
      { class: 'panel-intro' },
      'Provider und Modell für Agenten, Subagenten und Assistent Agents. Ohne Auswahl gilt der Standard-Provider. Ein im Tool-Aufruf angegebener Provider hat immer Vorrang.',
    ),
    ...rows.map((r) => r.el),
    h('div', { class: 'form-actions' }, h('button', { type: 'submit', class: 'btn btn-primary' }, 'Speichern'), status),
  );
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setFormStatus(status, 'Wird gespeichert …');
    try {
      const res = await api.putSettings(Object.assign({}, ...rows.map((r) => r.value())));
      if (res && typeof res === 'object') state.settings = { ...state.settings, ...res };
      setFormStatus(status, 'Gespeichert.');
    } catch (err) {
      setFormStatus(status, `Speichern fehlgeschlagen: ${err.message}`, 'error');
    }
  });
  panel.replaceChildren(form);
}

// ---------- Start ----------

async function init() {
  applyTheme(load(STORAGE.theme) || 'system');
  initSidebar();
  initComposer();
  initKeyboard();
  initSettings();

  els.themeToggle.addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme;
    setTheme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]);
  });
  els.newChat.addEventListener('click', () => {
    closeSidebarIfNarrow();
    setHash(null);
    showNewChat();
    els.input.focus();
  });
  els.providerSelect.addEventListener('change', () => {
    const pid = els.providerSelect.value;
    selectProvider(pid, defaultModelFor(pid), { persist: true });
  });
  els.reasoningSelect.addEventListener('change', () => {
    state.reasoning = els.reasoningSelect.value;
    saveReasoning(state.providerId, state.reasoning);
  });
  els.modelSelect.addEventListener('change', () => {
    if (els.modelSelect.value === '__error') return;
    state.model = els.modelSelect.value;
    persistChatModel();
  });
  window.addEventListener('hashchange', route);
  window.addEventListener('popstate', route);

  try {
    const [settings, providers] = await Promise.all([api.getSettings(), api.listProviders()]);
    state.settings = { ...state.settings, ...settings };
    state.providers = providers;
  } catch (e) {
    els.thread.replaceChildren(
      errorBubble({ code: e.code, message: e.message, hint: 'Läuft der NoAssist-Server?' }, { onRetry: () => location.reload() }),
    );
    return;
  }
  applyTheme(state.settings.theme);
  const pid = defaultProviderId();
  selectProvider(pid, defaultModelFor(pid));
  refreshChatList();
  await route();
}

init();
