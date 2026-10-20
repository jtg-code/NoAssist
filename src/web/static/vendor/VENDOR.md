# Fremdbibliotheken (Vendor)

Beide Dateien liegen unverändert aus den offiziellen npm-Paketen im Repo. Zur Laufzeit
werden keine CDNs und kein Node benötigt; `index.html` lädt sie als klassische Skripte
(UMD), sie stellen die Globals `marked` und `DOMPurify` bereit. Verwendet werden sie nur
in `js/render.js`.

| Datei | Paket | Version | Quelle im Paket | Lizenz | SHA-256 der Datei |
|---|---|---|---|---|---|
| `marked.min.js` | [marked](https://github.com/markedjs/marked) | 18.1.1 | `lib/marked.umd.js` (bereits minifiziert, nur umbenannt) | MIT | `b82b9334406d8bdd16b6236c08de1cb65b2a859873ca9503d9cc0b292ca3c12a` |
| `purify.min.js` | [dompurify](https://github.com/cure53/DOMPurify) | 3.4.17 | `dist/purify.min.js` | MPL-2.0 OR Apache-2.0 (genutzt: Apache-2.0) | `cd37898f9782adf54820c413191b5691803f0f137c799d6edcfd38523b4c2c31` |

## Herkunft und Prüfung

Bezogen am 2026-10-19 mit `npm pack marked@18.1.1 dompurify@3.4.17`. Die Tarballs
stimmen mit der Registry-Integrität (`npm view <paket>@<version> dist.integrity`) überein:

| Tarball | npm-Integrität (SHA-512) | SHA-256 |
|---|---|---|
| `marked-18.1.1.tgz` | `sha512-zEGgrtUG8crk+TrMacJSVABU4DYDDkS8uqgU3ohrn/2fW9C2AEfUrO3m6IhpIpQpl7YcVpiXXC9cSMWDANT3dg==` | `c05b3cff890d35e1208805a01fd0501571d09a81e725de0559b10bdfae90adad` |
| `dompurify-3.4.17.tgz` | `sha512-B/pSlYVHQ58kqISD/yDjuJ1yfzI34m1gF4lAod2bwSRhiN4K5Cl3H8Af7hqB2PmTL911tdEudnnXn5FnhRoz0w==` | `056cbdae276897208b88c21d1d779adfa09ad7be488a2f510a28e900d3bbffe7` |

Lizenzen: `package.json` von marked nennt `MIT` (Copyright-Hinweis steht im Dateikopf),
DOMPurify nennt `(MPL-2.0 OR Apache-2.0)` (Lizenzkommentar steht im Dateikopf). Beide sind
mit der Lizenz von NoAssist vereinbar. Die Lizenzkommentare in den Dateien bleiben erhalten.

Hinweis: Beide Dateien enden mit einem `sourceMappingURL`-Kommentar. Die `.map`-Dateien
liegen bewusst nicht im Repo; nur geöffnete Entwicklerwerkzeuge fragen sie (lokal) an.

## Aktualisieren

1. In einem leeren Ordner außerhalb des Repos `npm pack marked@<v> dompurify@<v>` ausführen.
2. Integrität mit `npm view <paket>@<v> dist.integrity` vergleichen.
3. `package/lib/marked.umd.js` als `marked.min.js` und `package/dist/purify.min.js` als
   `purify.min.js` hierher kopieren.
4. Versionen, Lizenzen und SHA-256 (`certutil -hashfile <datei> SHA256`) in dieser Datei
   nachtragen und die Darstellung (XSS-Test in `js/render.js`) erneut prüfen.
