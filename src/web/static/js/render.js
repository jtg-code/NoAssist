// Darstellung von Nachrichten: Markdown (marked + DOMPurify), Tool-Karten, Fehlerblasen.
// Regel: Fremde Daten (Modell, Tools, Server) landen nur als Text-Knoten im DOM oder
// laufen durch DOMPurify. Es wird nirgends innerHTML mit ungeprüften Daten gesetzt.

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Element bauen: h('p', {class: 'x', onclick: fn}, 'Text', kindElement) */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
    else if (key === 'text') el.textContent = value;
    else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, String(value));
  }
  append(el, children);
  return el;
}

function append(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    parent.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/** Icon aus dem Sprite in index.html (dekorativ, aria-hidden) */
export function icon(name, extraClass = '') {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', `icon ${extraClass}`.trim());
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  const use = document.createElementNS(SVG_NS, 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.append(use);
  return svg;
}

// ---------- Markdown ----------

function escapeHtml(s) {
  return String(s ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

// Erlaubt sind nur http(s)-, mailto- und Anker-Links; alles andere entfernt DOMPurify.
const SAFE_URI = /^(?:(?:https?|mailto):|#)/i;

const PURIFY_CONFIG = {
  RETURN_DOM_FRAGMENT: true,
  // Keine Medien/Formulare/eingebetteten Inhalte: verhindert externe Requests und Phishing-Formulare
  FORBID_TAGS: [
    'style', 'img', 'picture', 'source', 'video', 'audio', 'track', 'iframe', 'frame', 'object', 'embed',
    'form', 'input', 'button', 'select', 'textarea', 'option', 'svg', 'math', 'link', 'meta', 'base',
  ],
  FORBID_ATTR: ['style', 'srcset', 'background', 'poster', 'formaction', 'action'],
  ALLOW_DATA_ATTR: false,
  ALLOWED_URI_REGEXP: SAFE_URI,
};

let mdReady = false;

function setupMarkdown() {
  if (mdReady) return true;
  const { marked, DOMPurify } = window;
  if (!marked || !DOMPurify || typeof DOMPurify.sanitize !== 'function') return false;

  marked.use({
    gfm: true,
    breaks: false,
    async: false,
    renderer: {
      // Rohes HTML aus Modellausgaben wird als Text gezeigt, nicht interpretiert
      html(token) {
        return escapeHtml(token.text ?? token.raw ?? '');
      },
      // Bilder werden nie geladen (keine externen Requests) – stattdessen ein Link
      image(token) {
        const label = `Bild: ${token.text || token.href || ''}`;
        return `<a href="${escapeHtml(token.href || '')}">${escapeHtml(label)}</a>`;
      },
      // Aufgabenlisten ohne Formularelemente
      checkbox(token) {
        return token.checked ? '[x] ' : '[ ] ';
      },
    },
  });

  // Alle Links in neuem Fenster, ohne Zugriff auf window.opener und ohne Referrer
  DOMPurify.addHook('afterSanitizeAttributes', (node) => {
    if (node.nodeName === 'A') {
      if (node.hasAttribute('href')) {
        node.setAttribute('target', '_blank');
        node.setAttribute('rel', 'noopener noreferrer');
      } else {
        node.removeAttribute('target');
      }
    }
  });
  mdReady = true;
  return true;
}

/** Markdown-Text zu einem bereinigten DocumentFragment (Fallback: reiner Text). */
export function markdownFragment(text) {
  const src = String(text ?? '');
  if (!setupMarkdown()) {
    // Bibliotheken fehlen: sicher als reinen Text zeigen
    const frag = document.createDocumentFragment();
    frag.append(h('p', { class: 'plain-text' }, src));
    return frag;
  }
  const html = window.marked.parse(src);
  const frag = window.DOMPurify.sanitize(html, PURIFY_CONFIG);
  enhanceCodeBlocks(frag);
  return frag;
}

/** Inhalt eines Containers durch gerendertes Markdown ersetzen. */
export function renderMarkdownInto(container, text) {
  container.replaceChildren(markdownFragment(text));
}

function enhanceCodeBlocks(root) {
  for (const pre of root.querySelectorAll('pre')) {
    const wrap = h('div', { class: 'code-block' });
    pre.replaceWith(wrap);
    const btn = h('button', { type: 'button', class: 'code-copy', 'aria-label': 'Code kopieren' }, 'Kopieren');
    btn.addEventListener('click', () => copyText(pre.textContent || '', btn));
    wrap.append(pre, btn);
  }
}

/** Text in die Zwischenablage kopieren und am Button kurz bestätigen. */
export async function copyText(text, button) {
  let ok = false;
  try {
    await navigator.clipboard.writeText(text);
    ok = true;
  } catch {
    ok = false;
  }
  if (!button) return ok;
  const label = button.getAttribute('aria-label');
  const original = button.textContent;
  const msg = ok ? 'Kopiert' : 'Kopieren nicht möglich';
  if (original && original.trim()) button.textContent = msg;
  button.setAttribute('aria-label', msg);
  setTimeout(() => {
    if (original && original.trim()) button.textContent = original;
    if (label) button.setAttribute('aria-label', label);
  }, 1500);
  return ok;
}

// ---------- Tool-Karten ----------

function formatArgs(args) {
  if (args === undefined || args === null) return '';
  if (typeof args === 'string') return args;
  try {
    return JSON.stringify(args, null, 2);
  } catch {
    return String(args);
  }
}

function formatDuration(ms) {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) return '';
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1).replace('.', ',')} s`;
}

const TOOL_STATE_LABEL = {
  running: 'läuft …',
  ok: 'fertig',
  error: 'Fehler',
  cancelled: 'abgebrochen',
  missing: 'ohne Ergebnis',
};

/** Aufklappbare Karte für einen Tool-Aufruf (Argumente, Ergebnis, Status). */
export class ToolCard {
  constructor({ name, source, args, state = 'running' }) {
    this.name = name || 'Tool';
    this.stateEl = h('span', { class: 'tool-state' });
    this.summaryText = h('span', { class: 'visually-hidden' });
    const sourceLabel = source && source !== 'local' ? source.replace(/^mcp:/, 'MCP: ') : null;
    this.argsPre = h('pre', {}, formatArgs(args) || '{}');
    this.resultPre = h('pre', {}, '');
    this.resultHead = h('h4', {}, 'Ergebnis');
    this.note = h('p', { class: 'tool-note' });
    this.resultHead.hidden = true;
    this.resultPre.hidden = true;
    this.note.hidden = true;
    this.el = h(
      'details',
      { class: 'tool-card' },
      h(
        'summary',
        {},
        h('span', { class: 'tool-icon' }, icon('tool')),
        h('span', { class: 'visually-hidden' }, 'Tool-Aufruf '),
        h('span', { class: 'tool-name' }, this.name),
        sourceLabel ? h('span', { class: 'tool-source' }, sourceLabel) : null,
        this.stateEl,
      ),
      h('div', { class: 'tool-body' }, h('h4', {}, 'Argumente'), this.argsPre, this.resultHead, this.resultPre, this.note),
    );
    this.setState(state);
  }

  setState(state, durationMs) {
    this.state = state;
    this.el.dataset.state = state;
    const parts = [];
    if (state === 'running') parts.push(h('span', { class: 'spinner', 'aria-hidden': 'true' }));
    let label = TOOL_STATE_LABEL[state] || state;
    const dur = formatDuration(durationMs);
    if (dur && (state === 'ok' || state === 'error')) label += ` (${dur})`;
    parts.push(label);
    this.stateEl.replaceChildren(...parts);
  }

  setArgs(args) {
    this.argsPre.textContent = formatArgs(args) || '{}';
  }

  /** Live-Ausgabe eines laufenden Tools anhängen; die Karte klappt beim ersten Stück auf. */
  appendOutput(text, stream) {
    if (this.state !== 'running' || !text) return;
    if (!this.live) {
      this.live = true;
      this.liveText = '';
      this.resultHead.hidden = false;
      this.resultPre.hidden = false;
      this.resultHead.textContent = 'Ausgabe (live)';
      this.el.open = true;
    }
    this.liveText += text;
    // Nur das Ende halten: ein Dauerläufer soll den Browser nicht füllen
    if (this.liveText.length > 200_000) this.liveText = this.liveText.slice(-150_000);
    const pre = this.resultPre;
    const atEnd = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
    pre.textContent = this.liveText;
    pre.dataset.stream = stream === 'stderr' ? 'stderr' : 'stdout';
    if (atEnd) pre.scrollTop = pre.scrollHeight;
  }

  /** Ergebnis immer als Text (nie als HTML) anzeigen. */
  setResult({ result, isError, truncated, durationMs }) {
    this.resultHead.hidden = false;
    this.resultHead.textContent = 'Ergebnis';
    this.live = false;
    this.resultPre.hidden = false;
    this.resultPre.textContent = typeof result === 'string' ? result : formatArgs(result);
    if (truncated) {
      this.note.hidden = false;
      this.note.textContent = 'Ergebnis für die Anzeige gekürzt.';
    }
    this.setState(isError ? 'error' : 'ok', durationMs);
  }
}

// ---------- Fehlerblase ----------

const ERROR_TITLES = {
  auth_required: 'Anmeldung erforderlich',
  cli_missing: 'Claude-CLI nicht gefunden',
  provider_unreachable: 'Provider nicht erreichbar',
  model_not_found: 'Modell nicht gefunden',
  rate_limit: 'Nutzungslimit erreicht',
  bad_request: 'Ungültige Anfrage',
  tool_unsupported: 'Tools werden nicht unterstützt',
  internal: 'Interner Fehler',
  network: 'Keine Verbindung zum Server',
  busy: 'Antwort läuft bereits',
  stream_lost: 'Verbindung zur Antwort verloren',
  no_answer: 'Keine Antwort erhalten',
};

const SETTINGS_CODES = new Set(['auth_required', 'cli_missing', 'provider_unreachable', 'model_not_found', 'tool_unsupported']);

/** Fehlerblase mit Hinweis und Aktionen ("Erneut versuchen", ggf. "Einstellungen öffnen"). */
export function errorBubble(err, { onRetry, onSettings } = {}) {
  const code = err.code || 'error';
  const title = ERROR_TITLES[code] || 'Fehler';
  const actions = h('div', { class: 'error-actions' });
  if (onRetry) {
    actions.append(h('button', { type: 'button', class: 'btn', onclick: onRetry }, icon('retry'), 'Erneut versuchen'));
  }
  if (onSettings && SETTINGS_CODES.has(code)) {
    actions.append(h('button', { type: 'button', class: 'btn btn-ghost', onclick: onSettings }, icon('settings'), 'Provider-Einstellungen'));
  }
  return h(
    'div',
    { class: 'error-bubble', dataset: { code } },
    h('p', { class: 'error-title' }, icon('alert'), title),
    err.message ? h('p', { class: 'error-message' }, err.message) : null,
    err.hint ? h('p', { class: 'error-hint' }, err.hint) : null,
    actions.childElementCount ? actions : null,
  );
}

// ---------- Nachrichten ----------

export function userMessage(msg) {
  return h(
    'article',
    { class: 'msg msg-user', dataset: { id: msg.id || '' } },
    h('h2', { class: 'visually-hidden' }, 'Du'),
    h('div', { class: 'bubble-user' }, msg.content || ''),
  );
}

function assistantArticle() {
  return h('article', { class: 'msg msg-assistant' }, h('h2', { class: 'visually-hidden' }, 'Assistent'));
}

function copyAction(getText) {
  const btn = h('button', { type: 'button', class: 'icon-btn small', 'aria-label': 'Antwort kopieren', title: 'Antwort kopieren' }, icon('copy'));
  btn.addEventListener('click', () => copyText(getText(), btn));
  return h('div', { class: 'msg-actions' }, btn);
}

/**
 * Gespeicherten Verlauf darstellen. Aufeinanderfolgende Assistenten- und Tool-Nachrichten
 * bilden einen Antwortblock; Tool-Ergebnisse werden der Karte mit passender tool_call_id zugeordnet.
 */
export function renderHistory(messages) {
  const frag = document.createDocumentFragment();
  let block = null;
  let cards = new Map();
  let texts = [];

  const closeBlock = () => {
    if (!block) return;
    for (const card of cards.values()) {
      if (card.state === 'running') card.setState('missing');
    }
    if (texts.length) {
      const all = texts.join('\n\n');
      block.append(copyAction(() => all));
    }
    frag.append(block);
    block = null;
    cards = new Map();
    texts = [];
  };

  for (const msg of messages || []) {
    if (msg.role === 'user') {
      closeBlock();
      frag.append(userMessage(msg));
      continue;
    }
    if (!block) block = assistantArticle();
    if (msg.role === 'assistant') {
      if (msg.content) {
        const md = h('div', { class: 'md' });
        renderMarkdownInto(md, msg.content);
        block.append(md);
        texts.push(msg.content);
      }
      for (const tc of msg.tool_calls || []) {
        const card = new ToolCard({ name: tc.name, args: tc.arguments });
        cards.set(tc.id, card);
        block.append(card.el);
      }
    } else if (msg.role === 'tool') {
      let card = cards.get(msg.tool_call_id);
      if (!card) {
        card = new ToolCard({ name: msg.name || 'Tool', args: null });
        block.append(card.el);
      }
      card.setResult({ result: msg.content || '', isError: !!msg.is_error });
      cards.delete(msg.tool_call_id);
    }
  }
  closeBlock();
  return frag;
}

/** Leerraum vereinheitlichen, damit Zeilenumbrüche/Endleerzeichen den Vergleich nicht stören. */
function squash(text) {
  return String(text ?? '').replace(/\s+/g, ' ').trim();
}

/** Ist `full` derselbe Text wie `shown` (oder dessen Fortsetzung)? */
function continuesText(shown, full) {
  const a = squash(shown);
  const b = squash(full);
  return !!a && (a === b || b.startsWith(a));
}

/**
 * Live-Ansicht eines laufenden Turns, gespeist aus den SSE-Events.
 * onChange() wird nach jeder sichtbaren Änderung aufgerufen (für Auto-Scroll).
 */
export class LiveTurnView {
  constructor({ onChange } = {}) {
    this.onChange = onChange || (() => {});
    this.el = assistantArticle();
    this.el.setAttribute('aria-busy', 'true');
    this.segment = null; // aktueller Textabschnitt (div.md)
    this.lastClosed = null; // zuletzt abgeschlossener Abschnitt, auf den noch keine message kam
    this.text = '';
    this.texts = [];
    this.cards = new Map();
    this.reasoningBody = null;
    this.statusText = h('span', {});
    this.indicator = h(
      'div',
      { class: 'msg-status' },
      h('span', { class: 'typing', 'aria-hidden': 'true' }, h('span'), h('span'), h('span')),
      this.statusText,
    );
    this.el.append(this.indicator);
    this._raf = 0;
    this.finished = false;
  }

  _place(node) {
    this.el.insertBefore(node, this.indicator);
  }

  _scheduleRender() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      if (this.segment) renderMarkdownInto(this.segment, this.text);
      this.onChange();
    });
  }

  _closeSegment(finalText) {
    if (!this.segment) return;
    if (this._raf) {
      cancelAnimationFrame(this._raf);
      this._raf = 0;
    }
    const text = finalText ?? this.text;
    renderMarkdownInto(this.segment, text);
    if (text) this.texts.push(text);
    // Merken für message(): Manche Server (z. B. LM Studio) führen Tools schon während des
    // Streamings aus – dann schließt tool_call den Abschnitt, bevor message denselben Text liefert.
    this.lastClosed = text ? { el: this.segment, text, index: this.texts.length - 1 } : null;
    this.segment = null;
    this.text = '';
  }

  _updateIndicator() {
    // Während Text einläuft, ist der Text selbst die Rückmeldung
    this.indicator.hidden = this.finished || (!!this.segment && !this.statusText.textContent);
  }

  delta(text) {
    if (!text) return;
    if (!this.segment) {
      this.segment = h('div', { class: 'md' });
      this._place(this.segment);
    }
    this.statusText.textContent = '';
    this.text += text;
    this._updateIndicator();
    this._scheduleRender();
  }

  reasoning(text) {
    if (!text) return;
    if (!this.reasoningBody) {
      this.reasoningBody = h('div', { class: 'reasoning-body' });
      this._place(h('details', { class: 'reasoning' }, h('summary', {}, 'Gedankengang'), this.reasoningBody));
    }
    this.reasoningBody.append(document.createTextNode(text));
    this.onChange();
  }

  status(text) {
    this.statusText.textContent = text || '';
    this._updateIndicator();
    this.onChange();
  }

  toolCall(d) {
    this._closeSegment();
    let card = this.cards.get(d.call_id);
    if (!card) {
      card = new ToolCard({ name: d.name, source: d.source, args: d.arguments });
      this.cards.set(d.call_id, card);
      this._place(card.el);
    } else {
      card.setArgs(d.arguments);
    }
    this._updateIndicator();
    this.onChange();
  }

  toolOutput(d) {
    const card = this.cards.get(d.call_id);
    if (!card) return;
    card.appendOutput(d.text, d.stream);
    this.onChange();
  }

  toolResult(d) {
    let card = this.cards.get(d.call_id);
    if (!card) {
      card = new ToolCard({ name: d.name, args: null });
      this.cards.set(d.call_id, card);
      this._place(card.el);
    }
    card.setResult({ result: d.result, isError: !!d.is_error, truncated: !!d.truncated, durationMs: d.duration_ms });
    this.onChange();
  }

  /** Fertige Nachricht: beim Assistenten ist der gespeicherte Text maßgeblich. */
  message(msg) {
    if (!msg || msg.role !== 'assistant') return;
    const content = msg.content || '';
    const last = this.lastClosed;
    if (this.segment) {
      this._closeSegment(content || this.text);
    } else if (content && last && continuesText(last.text, content)) {
      // Text wurde schon gestreamt und vom tool_call abgeschlossen: nicht doppelt zeigen,
      // nur die gespeicherte (maßgebliche) Fassung in den vorhandenen Abschnitt übernehmen
      if (content !== last.text) {
        renderMarkdownInto(last.el, content);
        this.texts[last.index] = content;
      }
    } else if (content) {
      const md = h('div', { class: 'md' });
      this._place(md);
      renderMarkdownInto(md, content);
      this.texts.push(content);
    }
    // Jede message gehört zu höchstens einem Abschnitt
    this.lastClosed = null;
    this._updateIndicator();
    this.onChange();
  }

  error(node) {
    this._closeSegment();
    this._place(node);
    this.onChange();
  }

  /**
   * Turn beendet: offene Karten markieren, Anzeige abschließen.
   * status: 'completed' | 'cancelled' | 'error' | 'lost' (Ausgang unbekannt, z. B. nach Serverneustart)
   */
  finish(status) {
    this._closeSegment();
    this.finished = true;
    for (const card of this.cards.values()) {
      if (card.state === 'running') card.setState(status === 'completed' || status === 'lost' ? 'missing' : 'cancelled');
    }
    if (status === 'cancelled') this._place(h('p', { class: 'notice' }, 'Antwort abgebrochen.'));
    this.indicator.remove();
    this.el.removeAttribute('aria-busy');
    this.onChange();
  }
}
