// Live-Reload im Dev-Modus (Vertrag: Plan, "Dev-Events").
// - /api/health liefert dev=true  -> EventSource auf /api/dev/events, sonst passiert nichts.
// - "hello" {boot_id}: erste boot_id merken; kommt nach dem Wiederverbinden eine andere
//   (Server neu gestartet), wird die Seite neu geladen. Der aktive Chat bleibt über den
//   URL-Hash (#chat=<id>) erhalten.
// - "reload" {files, css_only}: css_only=true tauscht nur die Stylesheets aus
//   (Cache-Buster), sonst location.reload().

let bootId = null;
let source = null;
let retryTimer = 0;

function reloadPage() {
  if (source) source.close();
  location.reload();
}

function swapStylesheets() {
  const stamp = Date.now().toString(36);
  for (const link of document.querySelectorAll('link[rel="stylesheet"]')) {
    const url = new URL(link.href, location.href);
    if (url.origin !== location.origin) continue;
    url.searchParams.set('v', stamp);
    // Neues <link> erst nach dem Laden tauschen, damit die Seite nicht kurz ungestylt ist
    const next = link.cloneNode();
    next.href = url.pathname + url.search;
    next.addEventListener('load', () => link.remove(), { once: true });
    next.addEventListener('error', () => next.remove(), { once: true });
    link.after(next);
  }
}

function parse(e) {
  try {
    return e.data ? JSON.parse(e.data) : {};
  } catch {
    return {};
  }
}

function connect() {
  clearTimeout(retryTimer);
  source = new EventSource('/api/dev/events');
  source.addEventListener('hello', (e) => {
    const id = parse(e).boot_id ?? null;
    if (bootId !== null && id !== null && id !== bootId) {
      reloadPage();
      return;
    }
    if (id !== null) bootId = id;
  });
  source.addEventListener('reload', (e) => {
    const data = parse(e);
    if (data.css_only) swapStylesheets();
    else reloadPage();
  });
  source.onerror = () => {
    // Bei Netzfehlern verbindet EventSource selbst neu. Ist die Verbindung endgültig
    // geschlossen (z. B. Server liefert kurz 404/502 beim Neustart), selbst neu versuchen.
    if (source.readyState === EventSource.CLOSED) {
      source = null;
      retryTimer = setTimeout(connect, 2000);
    }
  };
}

async function start() {
  try {
    const res = await fetch('/api/health', { cache: 'no-store', headers: { Accept: 'application/json' } });
    if (!res.ok) return;
    const health = await res.json();
    if (health && health.dev === true) {
      if (health.boot_id) bootId = health.boot_id;
      connect();
    }
  } catch {
    /* Server nicht erreichbar: kein Dev-Reload */
  }
}

start();
