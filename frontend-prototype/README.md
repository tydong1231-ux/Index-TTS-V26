# VoiceBook Studio Frontend

The confirmed two-level interaction model is connected to the FastAPI backend:

1. **Project workspace** — chapters, project-wide roles and voices, batch jobs, settings, and export state stay on one page.
2. **Chapter workspace** — source, role split, voice references, synthesis, and QA stay on one child page.

The frontend calls `/api/health` on startup. When the backend is unavailable, it switches to built-in demonstration data rather than rendering an unusable page.

## Run

From the repository root:

```bash
pip install -r app/requirements-audiobook.txt
```

Windows:

```bat
start_audiobook_studio.bat
```

macOS/Linux:

```bash
sh start_audiobook_studio.sh
```

Then open `http://127.0.0.1:7861`.

## Repository layout

- `index.html` — small production loader served by FastAPI.
- `chunks/` — generated compressed runtime containing the complete HTML, CSS, and JavaScript UI.
- `source/api.js` — readable backend API client.
- `source/app-core.js` — state, backend initialization, and data mapping.
- `source/app-ui.js` — project and chapter rendering.
- `source/app-actions.js` — event handlers, batch jobs, and chapter actions.
- `source/style-*.css` — readable design-system, project, chapter, overlay, and responsive styles.

The runtime bundle is generated from these source modules and the semantic page template. The split runtime avoids a single oversized GitHub Contents API write while keeping the served application self-contained.
