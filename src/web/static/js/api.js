// API-Zugriff auf den lokalen NoAssist-Server (Schema siehe Plan, Abschnitt "API-Schema").
// Alle Anfragen gehen an denselben Ursprung; außer GET wird immer JSON gesendet
// (der Server verlangt Content-Type: application/json, sonst 415).

/** Fehler einer API-Anfrage mit HTTP-Status und Fehlercode aus {"error":{code,message}}. */
export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

async function request(method, path, body) {
  const opts = { method, headers: { Accept: 'application/json' }, cache: 'no-store' };
  if (method !== 'GET') {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body ?? {});
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new ApiError(0, 'network', 'Der NoAssist-Server ist nicht erreichbar.');
  }
  if (res.status === 204) return null;
  let data = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!res.ok) {
    const err = data && typeof data === 'object' ? data.error : null;
    throw new ApiError(
      res.status,
      (err && err.code) || `http_${res.status}`,
      (err && err.message) || `Anfrage fehlgeschlagen (HTTP ${res.status}).`,
    );
  }
  return data;
}

const enc = encodeURIComponent;

/** Liste aus einer Antwort lesen (nackte Liste oder {key:[...]}) */
function asList(data, key) {
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data[key])) return data[key];
  return [];
}

export const api = {
  health: () => request('GET', '/api/health'),

  getSettings: () => request('GET', '/api/settings'),
  putSettings: (data) => request('PUT', '/api/settings', data),

  listProviders: async () => asList(await request('GET', '/api/providers'), 'providers'),
  addProvider: (data) => request('POST', '/api/providers', data),
  updateProvider: (id, data) => request('PUT', `/api/providers/${enc(id)}`, data),
  deleteProvider: (id) => request('DELETE', `/api/providers/${enc(id)}`),
  providerModels: (id) => request('GET', `/api/providers/${enc(id)}/models`),
  providerStatus: (id) => request('GET', `/api/providers/${enc(id)}/status`),
  claudeLogin: () => request('POST', '/api/providers/claude/login', {}),

  listTools: () => request('GET', '/api/tools'),
  // Browser eines Chats: {active, visible, url}; showBrowser öffnet ihn als Fenster (visible=false versteckt ihn)
  browserState: (chatId) => request('GET', `/api/chats/${enc(chatId)}/browser`),
  showBrowser: (chatId, visible = true) => request('POST', `/api/chats/${enc(chatId)}/browser/show`, { visible }),
  // Subagenten des Chats chatId (bzw. der Assistent Agents, wenn chatId selbst ein Subagent ist)
  listSubagents: async (chatId) => asList(await request('GET', `/api/subagents?chat_id=${enc(chatId || '')}`), 'subagents'),

  listAgents: async () => asList(await request('GET', '/api/agents'), 'agents'),
  createAgent: (data) => request('POST', '/api/agents', data),
  // file: {filename, data (Base64)} – .md oder .zip (mit .md und Assets)
  importAgent: (file) => request('POST', '/api/agents/import', file),
  updateAgent: (id, data) => request('PUT', `/api/agents/${enc(id)}`, data),
  deleteAgent: (id) => request('DELETE', `/api/agents/${enc(id)}`),
  // files: [{name, data (Base64)}]; gleichnamige Assets werden ersetzt
  addAgentAssets: (id, files) => request('POST', `/api/agents/${enc(id)}/assets`, { files }),
  deleteAgentAsset: (id, name) => request('DELETE', `/api/agents/${enc(id)}/assets/${enc(name)}`),
  agentAssetUrl: (id, name) => `/api/agents/${enc(id)}/assets/${enc(name)}`,

  listChats: async () => asList(await request('GET', '/api/chats'), 'chats'),
  createChat: (data) => request('POST', '/api/chats', data),
  getChat: (id) => request('GET', `/api/chats/${enc(id)}`),
  updateChat: (id, data) => request('PATCH', `/api/chats/${enc(id)}`, data),
  deleteChat: (id) => request('DELETE', `/api/chats/${enc(id)}`),
  sendMessage: (chatId, data) => request('POST', `/api/chats/${enc(chatId)}/messages`, data),
  // Letzte gespeicherte User-Nachricht erneut ausführen, ohne eine neue anzulegen.
  // data: {provider_id, model, user_message_id} – user_message_id = die Nachricht, die der Client
  // wiederholen will. Antwort wie sendMessage: 202 {turn_id, user_message}; 409 "retry_stale",
  // wenn der Chat inzwischen weitergeführt wurde, 409 "busy", wenn gerade ein Turn läuft.
  retryMessage: (chatId, data = {}) => request('POST', `/api/chats/${enc(chatId)}/messages`, { ...data, retry: true }),
  cancelTurn: (turnId) => request('POST', `/api/turns/${enc(turnId)}/cancel`, {}),
};

/** Alle Event-Typen eines Turns (SSE "event:"-Feld). */
export const TURN_EVENTS = [
  'turn_start', 'delta', 'reasoning', 'status', 'tool_call', 'tool_output', 'tool_result', 'message', 'error', 'done',
];

/**
 * Live-Verbindung zu einem Turn per EventSource.
 *
 * - Startet mit ?after=<seq>; beim automatischen Wiederverbinden schickt der Browser
 *   selbst Last-Event-ID (der Server liefert dann nur neuere Events).
 * - Events mit seq <= zuletzt gesehener seq werden verworfen (doppelte Zustellung).
 * - Nach "done" wird die Verbindung geschlossen, damit EventSource nicht neu verbindet.
 * - Bricht die Verbindung endgültig ab (z. B. 404, weil der Server neu gestartet hat),
 *   wird onGone(lastSeq) aufgerufen; der Aufrufer entscheidet, wie es weitergeht.
 */
export class TurnStream {
  constructor(turnId, { after = 0, onEvent, onGone, onConnection } = {}) {
    this.turnId = turnId;
    this.lastSeq = after;
    this.onEvent = onEvent || (() => {});
    this.onGone = onGone || (() => {});
    this.onConnection = onConnection || (() => {});
    this.source = null;
    this.closed = false;
  }

  start() {
    if (this.closed) return;
    const url = `/api/turns/${enc(this.turnId)}/events?after=${this.lastSeq}`;
    const es = new EventSource(url);
    this.source = es;
    for (const type of TURN_EVENTS) {
      es.addEventListener(type, (e) => this._handle(type, e));
    }
    es.onopen = () => this.onConnection(true);
    es.onerror = () => {
      if (this.closed) return;
      if (es.readyState === EventSource.CLOSED) {
        // Endgültig: Browser verbindet nicht mehr selbst neu
        this.source = null;
        this.onGone(this.lastSeq);
      } else {
        // Browser verbindet selbst neu (retry aus dem Stream, Last-Event-ID wird mitgeschickt)
        this.onConnection(false);
      }
    };
  }

  _handle(type, e) {
    if (this.closed) return;
    // Achtung: EventSource feuert bei Verbindungsproblemen selbst ein "error"-Event (kein
    // MessageEvent). Nur echte SSE-Events mit "event: error" sind Turn-Fehler.
    if (!(e instanceof MessageEvent)) return;
    const seq = Number.parseInt(e.lastEventId, 10);
    if (Number.isFinite(seq)) {
      if (seq <= this.lastSeq) return;
      this.lastSeq = seq;
    }
    let data = {};
    try {
      data = e.data ? JSON.parse(e.data) : {};
    } catch {
      data = {};
    }
    if (type === 'done') this.close();
    this.onEvent(type, data);
  }

  close() {
    this.closed = true;
    if (this.source) {
      this.source.close();
      this.source = null;
    }
  }
}
