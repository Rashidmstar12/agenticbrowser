# Agentic Browser

A Chromium-based browser automation agent built on [Playwright](https://playwright.dev/python/).  
It ships with two ways to use it — a **local interactive terminal** and a **REST API server** — and solves the biggest practical problem with existing browser agents: **hallucination in workflow planning**.

---

## Table of Contents

1. [What makes this different](#1-what-makes-this-different)
2. [Architecture overview](#2-architecture-overview)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Quick start](#5-quick-start)
   - [Local interactive terminal](#local-interactive-terminal)
   - [API server](#api-server)
6. [Local runner — full reference](#6-local-runner--full-reference)
   - [Command-line flags](#command-line-flags)
   - [REPL commands — browser](#repl-commands--browser)
   - [REPL commands — system tools](#repl-commands--system-tools)
   - [REPL commands — task planner](#repl-commands--task-planner)
   - [JSON task files](#json-task-files)
7. [API server — full reference](#7-api-server--full-reference)
   - [Session management](#session-management)
   - [Navigation](#navigation)
   - [Element interaction](#element-interaction)
   - [Information extraction](#information-extraction)
   - [Popup handling](#popup-handling)
   - [Screenshots](#screenshots)
   - [Wait helpers](#wait-helpers)
   - [System tools — file I/O](#system-tools--file-io)
   - [System tools — code execution](#system-tools--code-execution)
   - [Task planner](#task-planner)
8. [Task planner — how it works](#8-task-planner--how-it-works)
   - [Built-in workflow templates](#built-in-workflow-templates)
   - [LLM backends](#llm-backends)
   - [Step schema reference](#step-schema-reference)
   - [Variable substitution](#variable-substitution)
9. [System tools — workspace & safety](#9-system-tools--workspace--safety)
10. [LLM configuration](#10-llm-configuration)
11. [Environment variables](#11-environment-variables)
12. [Example workflows](#12-example-workflows)
13. [Troubleshooting](#13-troubleshooting)
14. [Project structure](#14-project-structure)

---

## 1. What makes this different

| Problem with existing agents | How this tool fixes it |
|---|---|
| LLM hallucinates extra/wrong steps for simple tasks like "go to Google and search X" | **Deterministic workflow templates** handle common tasks with zero LLM calls |
| Wrong CSS selectors for well-known sites | **Hardcoded verified selectors** for Google, Bing, DuckDuckGo, YouTube, Wikipedia |
| No schema enforcement — LLM can invent actions that don't exist | Every step is **validated against a strict schema** before execution |
| Browser-only — can't save results to disk | **System tools layer**: file read/write, Python execution, shell commands |
| Workflows broken by cookie banners / modals | **Auto popup dismissal** on every page load + manual override |
| Collect data with no way to store it | **`{{last}}` variable** pipes page text directly into file-write steps |

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────┐
│                   Your code / CLI                   │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │      TaskPlanner      │  intent → validated steps → execution
         │  (task_planner.py)    │
         └──────┬──────────┬─────┘
                │          │
    ┌───────────▼──┐   ┌───▼──────────┐
    │ BrowserAgent │   │ SystemTools  │
    │(browser_agent│   │(system_tools │
    │    .py)      │   │    .py)      │
    │  Playwright/ │   │ File I/O,    │
    │  Chromium    │   │ Python exec, │
    └──────────────┘   │ Shell exec   │
                       └──────────────┘

Two entry points:
  local_runner.py  →  interactive REPL + JSON task file runner
  api_server.py    →  FastAPI REST server (port 8000)
```

---

## 3. Requirements

- **Python 3.11+**
- **Chromium** (installed automatically by Playwright)
- Optional: **OpenAI API key** or a running **Ollama** instance for the LLM planner

---

## 4. Installation

```bash
# 1. Clone the repository
git clone https://github.com/Rashidmstar12/agenticbrowser.git
cd agenticbrowser

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Playwright's Chromium browser
playwright install chromium
```

> **Tip — Docker / headless servers**: If you're running on a server without a display, the browser starts in headless mode by default (no changes needed). To install the system-level dependencies Playwright needs on Ubuntu/Debian:
> ```bash
> playwright install-deps chromium
> ```

---

## 5. Quick start

### Local interactive terminal

```bash
# Start the interactive REPL (headless browser)
python local_runner.py

# Show the browser window
python local_runner.py --no-headless

# Run a single natural-language task and exit
python local_runner.py --intent "go to google and search python asyncio"

# Run a JSON task file
python local_runner.py --task examples/search_and_save.json

# Slow down every action by 500 ms (great for debugging)
python local_runner.py --no-headless --slow-mo 500
```

Once the REPL starts you'll see `browser>`. Type `help` for a full list of commands.

```
browser> navigate https://example.com
browser> text
browser> screenshot result.png
browser> task go to google and search best Python libraries
browser> quit
```

### API server

```bash
# Start the API server on port 8000
python api_server.py

# Custom host/port
python api_server.py --host 127.0.0.1 --port 9000

# Auto-reload on code changes (development)
python api_server.py --reload
```

Once running, open **http://localhost:8000/docs** for the interactive Swagger UI.

Basic usage:
```bash
# 1. Start a browser session
curl -s -X POST http://localhost:8000/session/start \
  -H "Content-Type: application/json" \
  -d '{"headless": true}'

# 2. Run a natural-language task
curl -s -X POST http://localhost:8000/task/run \
  -H "Content-Type: application/json" \
  -d '{"intent": "go to google and search python tutorials"}'

# 3. Stop the session
curl -s -X POST http://localhost:8000/session/stop
```

---

## 6. Local runner — full reference

### Command-line flags

| Flag | Default | Description |
|---|---|---|
| `--no-headless` | off | Show the browser window |
| `--slow-mo MS` | `0` | Slow every action by MS milliseconds |
| `--no-auto-popups` | off | Disable automatic popup dismissal |
| `--intent TEXT` | — | Run a single natural-language task and exit |
| `--task FILE` | — | Run a JSON task file and exit |
| `--cmd TEXT` | — | Run a single low-level command and exit |
| `--workspace DIR` | `./workspace` | Directory for file I/O operations |
| `--proxy URL` | — | Proxy server, e.g. `http://user:pass@host:port` or `socks5://host:port` |

### REPL commands — browser

| Command | Example | Description |
|---|---|---|
| `navigate <url>` | `navigate https://example.com` | Go to a URL |
| `click <selector>` | `click button#submit` | Click an element |
| `type <selector> <text>` | `type input[name=q] hello` | Type text (fires key events) |
| `fill <selector> <value>` | `fill #email user@example.com` | Fill an input field (fast) |
| `press <key>` | `press Enter` | Press a keyboard key |
| `hover <selector>` | `hover nav a.menu` | Hover over an element |
| `scroll [y] [x]` | `scroll 500` | Scroll by pixels |
| `scroll_to <selector>` | `scroll_to footer` | Scroll element into view |
| `close_popups` | `close_popups` | Dismiss cookie banners / modals |
| `screenshot [path] [full]` | `screenshot out.png true` | Capture a screenshot |
| `text [selector]` | `text h1` | Print inner text (default: body) |
| `html [selector]` | `html #content` | Print inner HTML |
| `attr <selector> <attr>` | `attr a[class=logo] href` | Get an element attribute |
| `query <selector>` | `query a` | List all matching elements |
| `eval <js>` | `eval document.title` | Run JavaScript |
| `info` | `info` | Show current URL, title, viewport |
| `wait <selector>` | `wait #results` | Wait for an element to appear |
| `wait_state [state]` | `wait_state networkidle` | Wait for page load state |
| `help` | `help` | List all commands |
| `quit` | `quit` | Exit |

### REPL commands — system tools

All file paths are relative to the **workspace directory** (default `./workspace/`).

| Command | Example | Description |
|---|---|---|
| `write_file <path> <content>` | `write_file out.txt Hello world` | Write a file (overwrite) |
| `append_file <path> <content>` | `append_file log.txt new line` | Append to a file |
| `read_file <path>` | `read_file results.txt` | Read and print a file |
| `list_dir [path]` | `list_dir .` | List workspace directory |
| `run_python <code>` | `run_python print(1+1)` | Execute Python code |
| `run_shell <command>` | `run_shell ls -la` | Execute a shell command |

### REPL commands — task planner

| Command | Example | Description |
|---|---|---|
| `task <intent>` | `task go to google and search asyncio` | Plan + execute a natural-language task |
| `task_plan <intent>` | `task_plan search python on youtube` | Preview the plan without executing |

### JSON task files

A task file is a JSON array of step objects. Each step has an `"action"` key plus the required parameters for that action.

```json
[
  { "action": "navigate",      "url": "https://news.ycombinator.com" },
  { "action": "close_popups" },
  { "action": "wait_state",    "state": "networkidle" },
  { "action": "get_text",      "selector": "body" },
  { "action": "write_file",    "path": "hn_front_page.txt", "content": "{{last}}" },
  { "action": "screenshot",    "path": "hn.png", "full_page": true }
]
```

Run it:
```bash
python local_runner.py --task my_task.json
```

See the **[Step schema reference](#step-schema-reference)** for all available actions and their parameters.

---

## 7. API server — full reference

All endpoints accept and return JSON. Start a session before calling any browser/task endpoint.

Interactive API docs: **http://localhost:8000/docs**

### Session management

| Method | Path | Description |
|---|---|---|
| `POST` | `/session/start` | Start a browser session |
| `POST` | `/session/stop` | Stop the session and free resources |
| `GET` | `/session/status` | Get current URL, title, viewport |

**`POST /session/start` body:**
```json
{
  "headless": true,
  "slow_mo": 0,
  "auto_close_popups": true,
  "default_timeout": 30000
}
```

### Navigation

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/navigate` | `{"url": "...", "wait_until": "domcontentloaded"}` | Navigate to a URL |

### Element interaction

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/click` | `{"selector": "...", "timeout": null}` | Click an element |
| `POST` | `/fill` | `{"selector": "...", "value": "..."}` | Fill an input |
| `POST` | `/type` | `{"selector": "...", "text": "...", "clear_first": true}` | Type text |
| `POST` | `/press_key` | `{"key": "Enter"}` | Press a key |
| `POST` | `/hover` | `{"selector": "..."}` | Hover over element |
| `POST` | `/select_option` | `{"selector": "...", "value": "..."}` | Select a dropdown option |
| `POST` | `/scroll` | `{"x": 0, "y": 500}` | Scroll by pixels |
| `POST` | `/scroll_to_element` | `{"selector": "..."}` | Scroll element into view |

### Information extraction

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/page/info` | — | URL, title, viewport |
| `POST` | `/page/text` | `{"selector": "body"}` | Inner text of element |
| `POST` | `/page/html` | `{"selector": "body"}` | Inner HTML of element |
| `POST` | `/page/attribute` | `{"selector": "...", "attribute": "href"}` | Element attribute value |
| `POST` | `/page/query_all` | `{"selector": "a"}` | All matching elements (text + href) |
| `POST` | `/evaluate` | `{"script": "document.title"}` | Run JavaScript |

### Popup handling

| Method | Path | Description |
|---|---|---|
| `POST` | `/popups/close` | Dismiss cookie banners, modals, dialogs |

### Screenshots

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/screenshot` | `{"path": null, "full_page": false, "as_base64": true}` | Capture screenshot |

When `path` is `null` the image is returned as a base64-encoded string in the response.

### Wait helpers

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/wait/selector` | `{"selector": "...", "timeout": null}` | Wait for element to appear |
| `POST` | `/wait/load_state` | `{"state": "networkidle"}` | Wait for page load state |

### System tools — file I/O

All paths are relative to the server's **workspace directory** (`./workspace/` by default).

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/system/write_file` | `{"path": "out.txt", "content": "...", "mode": "w"}` | Write a file |
| `POST` | `/system/append_file` | `{"path": "log.txt", "content": "..."}` | Append to a file |
| `POST` | `/system/read_file` | `{"path": "out.txt"}` | Read a file |
| `POST` | `/system/list_dir` | `{"path": "."}` | List directory |
| `POST` | `/system/make_dir` | `{"path": "subdir"}` | Create a directory |
| `POST` | `/system/delete_file` | `{"path": "old.txt"}` | Delete file or directory |
| `GET` | `/system/info` | — | Workspace path and file counts |

### System tools — code execution

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/system/run_python` | `{"code": "print(1+1)", "timeout": 30}` | Run Python snippet |
| `POST` | `/system/run_shell` | `{"command": "ls -la", "timeout": 30}` | Run shell command |

Both return `{"stdout": "...", "stderr": "...", "exit_code": 0, "success": true}`.

### Task planner

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/task/run` | `{"intent": "...", "stop_on_error": true}` | Plan + execute in one call |
| `POST` | `/task/plan` | `{"intent": "..."}` | Preview steps without executing |
| `POST` | `/task/execute` | `{"steps": [...], "stop_on_error": true}` | Execute a pre-built step list |
| `GET` | `/task/schema` | — | Full step schema (all valid actions) |

**`POST /task/run` — example:**
```bash
curl -s -X POST http://localhost:8000/task/run \
  -H "Content-Type: application/json" \
  -d '{"intent": "collect text from https://example.com and save to example.txt"}'
```

---

## 8. Task planner — how it works

The planner converts a natural-language intent into a **minimal, validated, deterministic sequence of steps** and executes them.

```
intent  ──►  template match?  ──yes──►  hardcoded steps  ──►  validate  ──►  execute
                │ no
                ▼
           LLM call (constrained prompt, JSON-only output)
                │
                ▼
           validate against STEP_SCHEMA
                │
                ▼
              execute
```

### Built-in workflow templates

These intents are handled with **zero LLM calls** using hardcoded, verified steps.

| Intent pattern | What it does |
|---|---|
| `go to google and search <query>` | Navigates to Google, fills the search box, presses Enter |
| `search <query> on google` | Same |
| `google <query>` | Same |
| `search <query> on bing` | Bing search |
| `search <query> on duckduckgo` | DuckDuckGo search |
| `search <query> on youtube` | YouTube search |
| `search <query> on wikipedia` | Wikipedia search |
| `open / go to / navigate to / visit <url>` | Navigate and dismiss popups |
| `collect text from <url> and save to <file>` | Navigate → get page text → write to file |

All patterns are case-insensitive and handle common phrasing variations.

### LLM backends

When no template matches, the planner calls an LLM with a **tightly constrained prompt** that:
- Outputs **only** a JSON array (no prose, no markdown)
- Only uses action names from the schema
- Has a step cap of 12 to prevent runaway plans
- Injects hardcoded selectors for popular sites so the LLM never guesses them

Configure an LLM backend via environment variables (see [LLM configuration](#10-llm-configuration)).

### Step schema reference

Every step must have an `"action"` field. The full list:

#### Browser actions

| Action | Required params | Optional params | Description |
|---|---|---|---|
| `navigate` | `url` | `wait_until` (default: `domcontentloaded`) | Go to a URL |
| `click` | `selector` | `timeout` | Click an element |
| `fill` | `selector`, `value` | — | Fill an input field |
| `type` | `selector`, `text` | `clear_first` (default: `true`) | Type text |
| `press` | `key` | — | Press a key |
| `hover` | `selector` | — | Hover over element |
| `select_option` | `selector`, `value` | — | Choose a `<select>` option |
| `scroll` | — | `x` (0), `y` (500) | Scroll by pixels |
| `close_popups` | — | — | Dismiss overlays |
| `wait_selector` | `selector` | `timeout` | Wait for element |
| `wait_state` | — | `state` (default: `networkidle`) | Wait for load state |
| `screenshot` | — | `path`, `full_page` | Capture screenshot |
| `evaluate` | `script` | — | Run JavaScript |
| `get_text` | — | `selector` (default: `body`) | Get page text → stored as `{{last}}` |

#### System actions

| Action | Required params | Optional params | Description |
|---|---|---|---|
| `write_file` | `path`, `content` | `mode` (default: `w`) | Write file in workspace |
| `append_file` | `path`, `content` | — | Append to file |
| `read_file` | `path` | — | Read file |
| `list_dir` | — | `path` (default: `.`) | List directory |
| `run_python` | `code` | `timeout` (default: 30) | Execute Python code |
| `run_shell` | `command` | `timeout` (default: 30) | Execute shell command |

### Variable substitution

Use `{{last}}` in any string field of a step to reference the **text result of the previous step**.

```json
[
  { "action": "get_text",   "selector": "#results" },
  { "action": "write_file", "path": "results.txt", "content": "{{last}}" }
]
```

Only `{{last}}` is supported; it always refers to the most recent step that produced text output.

---

## 9. System tools — workspace & safety

All file operations are **confined to the workspace directory** (default `./workspace/`). Attempts to escape via `../` or absolute paths (e.g. `/etc/passwd`) raise a `PathTraversalError` and are blocked.

Code execution runs in a **subprocess** with a configurable timeout. The working directory is set to the workspace, so relative file paths work as expected.

Output from `run_python` and `run_shell` is capped at 1 MB to protect against runaway commands.

---

## 10. LLM configuration

The task planner auto-detects the available backend in this order:

1. **OpenAI** — if `OPENAI_API_KEY` is set
2. **Ollama** — if `OLLAMA_HOST` is set or Ollama is running on `localhost:11434`
3. **Templates only** — if no LLM is available (common tasks still work fine)

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # optional, default: gpt-4o-mini
```

### Ollama (local, free)

1. Install Ollama: https://ollama.com
2. Pull a model: `ollama pull llama3`
3. Start the server: `ollama serve`
4. Set the env var (if not on localhost): `export OLLAMA_HOST=http://my-server:11434`
5. Optionally set the model: `export OLLAMA_MODEL=llama3`

---

## 11. Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (enables OpenAI planning) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model to use |
| `BROWSER_WORKSPACE` | `./workspace` | Root directory for all file operations |
| `BROWSER_API_KEY` | — | When set, all API server requests must include `X-API-Key: <value>` header (or `?api_key=<value>` for WebSocket connections). Omit to run without authentication (local/trusted-network use only). |
| `BROWSER_CORS_ORIGINS` | — | Comma-separated list of allowed CORS origins, e.g. `https://app.example.com,https://dev.example.com`. Use `*` for development only. When unset, no CORS headers are added (safest default). |

---

## 12. Example workflows

### Search Google and save results

**REPL:**
```
browser> task go to google and search python asyncio
```

**JSON task file:**
```json
[
  { "action": "navigate",      "url": "https://www.google.com" },
  { "action": "close_popups" },
  { "action": "wait_selector", "selector": "textarea[name='q']" },
  { "action": "fill",          "selector": "textarea[name='q']", "value": "python asyncio" },
  { "action": "press",         "key": "Enter" },
  { "action": "wait_state",    "state": "networkidle" },
  { "action": "get_text",      "selector": "#search" },
  { "action": "write_file",    "path": "google_results.txt", "content": "{{last}}" }
]
```

### Scrape a page and save it

**API:**
```bash
curl -X POST http://localhost:8000/task/run \
  -H "Content-Type: application/json" \
  -d '{"intent": "collect text from https://news.ycombinator.com and save to hn.txt"}'
```

**JSON task file:**
```json
[
  { "action": "navigate",   "url": "https://news.ycombinator.com" },
  { "action": "close_popups" },
  { "action": "wait_state", "state": "networkidle" },
  { "action": "screenshot", "path": "hn.png", "full_page": true },
  { "action": "get_text",   "selector": "body" },
  { "action": "write_file", "path": "hn.txt", "content": "{{last}}" }
]
```

### Run Python on collected data

```json
[
  { "action": "navigate",    "url": "https://example.com/data.csv" },
  { "action": "get_text",    "selector": "pre" },
  { "action": "write_file",  "path": "data.csv", "content": "{{last}}" },
  { "action": "run_python",  "code": "import csv, json\nwith open('data.csv') as f:\n  rows = list(csv.DictReader(f))\nprint(json.dumps(rows[:5], indent=2))" }
]
```

### Login to a site

```json
[
  { "action": "navigate",      "url": "https://app.example.com/login" },
  { "action": "close_popups" },
  { "action": "fill",          "selector": "input[name='email']",    "value": "user@example.com" },
  { "action": "fill",          "selector": "input[name='password']", "value": "mysecretpassword" },
  { "action": "click",         "selector": "button[type='submit']" },
  { "action": "wait_state",    "state": "networkidle" },
  { "action": "screenshot",    "path": "after_login.png" }
]
```

### Preview a plan without running it

```bash
curl -X POST http://localhost:8000/task/plan \
  -H "Content-Type: application/json" \
  -d '{"intent": "search machine learning on youtube"}'
```

Response:
```json
{
  "intent": "search machine learning on youtube",
  "steps": [
    {"action": "navigate",      "url": "https://www.youtube.com", "wait_until": "domcontentloaded"},
    {"action": "close_popups"},
    {"action": "wait_selector", "selector": "input#search", "timeout": null},
    {"action": "fill",          "selector": "input#search", "value": "machine learning"},
    {"action": "press",         "key": "Enter"},
    {"action": "wait_state",    "state": "networkidle"}
  ],
  "count": 6
}
```

---

## 13. Troubleshooting

### `playwright install chromium` fails

On Linux servers, install the system dependencies first:
```bash
playwright install-deps chromium
playwright install chromium
```

### `Error: Browser session is not active`

You must start a session before calling browser/task endpoints:
```bash
curl -X POST http://localhost:8000/session/start -H "Content-Type: application/json" -d '{}'
```

### `No template matched … and no LLM is configured`

Your intent doesn't match a built-in template and no LLM is set up.  
Either:
- Rephrase to match a template (e.g. `"go to google and search X"`)
- Set `OPENAI_API_KEY` for OpenAI, or start Ollama for a local LLM

### `networkidle` timeout on modern web apps

Sites using WebSockets (Slack, Notion, Figma, etc.) never reach `networkidle`.  
Use `"wait_until": "load"` or `"wait_until": "domcontentloaded"` instead:
```json
{ "action": "navigate", "url": "https://app.example.com", "wait_until": "load" }
```

### Popup not dismissed

If `close_popups` can't dismiss a popup automatically:
1. Take a screenshot to see what's on screen: `screenshot debug.png`
2. Use `eval` to find the element: `eval document.querySelector('[class*="modal"]')?.outerHTML`
3. Click it directly: `click .your-close-button-selector`

### `PathTraversalError` when writing files

File paths must stay inside the workspace. Use relative paths only:
```
✓  write_file results/output.txt ...
✗  write_file /etc/passwd ...
✗  write_file ../../outside.txt ...
```

---

## 14. Project structure

```
agenticbrowser/
├── browser_agent.py   # Core BrowserAgent class (Playwright/Chromium)
│                      #   Navigation, click, type, fill, scroll,
│                      #   screenshot, popup handling, JS execution
│
├── system_tools.py    # SystemTools class
│                      #   File read/write, Python execution, shell commands
│                      #   All confined to a safe workspace directory
│
├── task_planner.py    # TaskPlanner class
│                      #   Template matching (zero-LLM common tasks)
│                      #   Constrained LLM planning (OpenAI / Ollama)
│                      #   Step schema validation
│                      #   Variable substitution ({{last}})
│                      #   Execution engine (browser + system steps)
│
├── api_server.py      # FastAPI REST server
│                      #   All browser/system/task endpoints
│                      #   Swagger UI at /docs
│
├── local_runner.py    # Interactive terminal + JSON task file runner
│                      #   REPL with all commands
│                      #   --intent / --task / --cmd / --workspace CLI flags
│
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 15. Futuristic features

These features extend the core with smart extraction, assertions, wait helpers, multi-tab support, cookie persistence, step-level retry, and execution logging.

### Smart extraction actions

| Action | Required | Optional | Description |
|--------|----------|----------|-------------|
| `extract_links` | — | `selector` ("a"), `limit` (100) | Return all `<a>` hrefs as `[{text, href}]`. Stored as `{{last}}` (JSON string). |
| `extract_table` | — | `selector` ("table"), `table_index` (0) | Return HTML table rows as `[{header: value}]`. |

**Example — scrape all links from a page:**
```json
[
  { "action": "navigate",       "url": "https://news.ycombinator.com" },
  { "action": "extract_links",  "selector": "a.storylink", "limit": 30 },
  { "action": "write_file",     "path": "hn_links.json", "content": "{{last}}" }
]
```

**API:** `POST /page/extract_links`, `POST /page/extract_table`

---

### Assertion actions

Assertions **fail the task** (raise an error) when the condition is not met. Use them to verify page state before proceeding to the next step.

| Action | Required | Optional | Description |
|--------|----------|----------|-------------|
| `assert_text` | `text` | `selector` ("body"), `case_sensitive` (false) | Fail if text is not found in element. |
| `assert_url` | `pattern` | — | Fail if current URL does not match regex pattern. |

**Example — confirm login succeeded:**
```json
[
  { "action": "navigate",    "url": "https://app.example.com/login" },
  { "action": "fill",        "selector": "input[name='email']",    "value": "user@example.com" },
  { "action": "fill",        "selector": "input[name='password']", "value": "secret" },
  { "action": "click",       "selector": "button[type='submit']" },
  { "action": "assert_url",  "pattern": "/dashboard" },
  { "action": "assert_text", "text": "Welcome back" }
]
```

**API:** `POST /assert/text`, `POST /assert/url`

---

### Wait for dynamic content

| Action | Required | Optional | Description |
|--------|----------|----------|-------------|
| `wait_text` | `text` | `selector` ("body"), `timeout` (ms) | Poll until the text appears in the element (good for async/SPA content). |

**API:** `POST /wait/text`

---

### Cookie persistence

Save and restore browser sessions across runs.

| Action | Required | Description |
|--------|----------|-------------|
| `save_cookies` | `path` | Serialize all cookies to a JSON file in the workspace. |
| `load_cookies` | `path` | Restore cookies from a workspace JSON file into the browser context. |

**Example — reuse a logged-in session:**
```json
[
  { "action": "load_cookies", "path": "session.json" },
  { "action": "navigate",     "url": "https://app.example.com/dashboard" },
  { "action": "assert_url",   "pattern": "/dashboard" }
]
```

**API:** `POST /cookies/save`, `POST /cookies/load`

---

### Multi-tab management

| Action | Required | Optional | Description |
|--------|----------|----------|-------------|
| `new_tab` | — | `url` | Open a new tab (optionally navigate). Makes it the active tab. |
| `switch_tab` | `index` | — | Switch the active tab by zero-based index. |
| `close_tab` | — | `index` (active tab) | Close a tab. |
| `list_tabs` | — | — | Return info about all open tabs. |

**Example — open two tabs in parallel:**
```json
[
  { "action": "navigate",    "url": "https://example.com" },
  { "action": "new_tab",     "url": "https://news.ycombinator.com" },
  { "action": "get_text",    "selector": ".itemlist" },
  { "action": "switch_tab",  "index": 0 },
  { "action": "get_text",    "selector": "body" }
]
```

**REPL:** `new_tab [url]`, `switch_tab <index>`, `close_tab [index]`, `list_tabs`

**API:** `POST /tabs/new`, `POST /tabs/switch`, `POST /tabs/close`, `GET /tabs/list`

---

### Per-step retry

Any step can declare `retry` (number of extra attempts) and `retry_delay` (seconds between attempts). The step is only marked as failed after all attempts are exhausted.

```json
[
  { "action": "click", "selector": "#dynamic-button", "retry": 3, "retry_delay": 2.0 }
]
```

This is useful for elements that appear asynchronously, flaky network conditions, or timing-sensitive interactions.

---

### Execution logging

Pass `log_path` to `/task/run` or `TaskPlanner.run()` to save the full execution log (steps, results, timestamp) as a JSON file in the workspace.

**API:**
```bash
curl -X POST http://localhost:8000/task/run \
  -H "Content-Type: application/json" \
  -d '{"intent": "search python on google", "log_path": "logs/run.json"}'
```

**Python:**
```python
planner.run("search python on google", agent, log_path="logs/run.json")
```

---

### `last_result` in Python scripts

When a `run_python` step follows a step that produced text output (e.g. `get_text`, `read_file`, `extract_links`), the previous output is automatically injected as a Python variable named `last_result`:

```json
[
  { "action": "get_text",    "selector": "#results" },
  { "action": "run_python",  "code": "words = last_result.split(); print(len(words), 'words')" }
]
```

---

### `--intent` and `--workspace` CLI flags

```bash
# Run a task directly from the command line and exit
python local_runner.py --intent "go to google and search python tutorials"

# Use a custom workspace directory for all file operations
python local_runner.py --workspace /tmp/myworkspace --intent "collect text from https://example.com and save to page.txt"

# BROWSER_WORKSPACE environment variable is also honoured
export BROWSER_WORKSPACE=/data/workspace
python local_runner.py --intent "..."
```
