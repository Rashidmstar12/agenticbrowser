# agenticbrowser

**agenticbrowser** is a Python library that gives AI agents a simple, structured
interface for controlling a web browser.  It wraps
[Playwright](https://playwright.dev/python/) and exposes every browser action
as a plain Python call **or** as a dictionary-driven `run_action()` call that
is easy to integrate with language-model tool-use frameworks.

---

## Features

| Category | Actions |
|----------|---------|
| **Navigation** | `navigate`, `go_back`, `go_forward`, `reload` |
| **Interaction** | `click`, `type_text`, `press_key`, `hover`, `select_option` |
| **Scrolling** | `scroll`, `scroll_to_bottom`, `scroll_to_top`, `scroll_element_into_view` |
| **Extraction** | `get_text`, `get_html`, `get_url`, `get_title` |
| **Querying** | `find_elements`, `find_links`, `find_buttons`, `find_inputs` |
| **Screenshots** | `screenshot`, `screenshot_base64` |
| **JavaScript** | `evaluate` |
| **Waiting** | `wait_for_selector`, `wait_for_load`, `wait_for_network_idle` |
| **Cookies** | `get_cookies`, `set_cookie`, `clear_cookies` |

All actions return structured result objects (`ActionResult`, `NavigateResult`,
`TextResult`, …) that are easy to inspect programmatically.

---

## Installation

```bash
pip install agenticbrowser
playwright install chromium   # or: playwright install --with-deps
```

---

## Quick-start

### Context manager (recommended)

```python
from agenticbrowser import Browser

with Browser() as browser:
    result = browser.navigate("https://example.com")
    print(result.title)           # "Example Domain"

    text = browser.get_text()
    print(text.text[:200])

    links = browser.find_links()
    for link in links.elements:
        print(link.href, link.text)
```

### Agent-style action dispatch

```python
from agenticbrowser.agent import AgentBrowser

with AgentBrowser() as agent:
    agent.run_action({"action": "navigate", "url": "https://example.com"})
    agent.run_action({"action": "click", "selector": "a.read-more"})
    result = agent.run_action({"action": "get_text"})
    print(result.text)
```

### Custom browser options

```python
from agenticbrowser import Browser, BrowserOptions

opts = BrowserOptions(
    browser_type="firefox",
    headless=False,
    viewport_width=1920,
    viewport_height=1080,
    timeout=60_000,
)

with Browser(opts) as browser:
    browser.navigate("https://example.com")
    browser.screenshot("page.png", full_page=True)
```

### Screenshot → base64 (for multimodal LLMs)

```python
with Browser() as browser:
    browser.navigate("https://example.com")
    b64 = browser.screenshot_base64()
    # pass b64 directly to an LLM vision API
```

---

## CLI

```bash
# Navigate and print the page title + URL
agenticbrowser navigate https://example.com

# Save a full-page screenshot
agenticbrowser screenshot https://example.com page.png --full-page

# Dump page text to stdout
agenticbrowser text https://example.com

# List all links (as JSON)
agenticbrowser links https://example.com --json

# Execute a sequence of actions from a JSON file
agenticbrowser run actions.json
```

**`actions.json`** example:

```json
[
  {"action": "navigate", "url": "https://example.com"},
  {"action": "find_links"},
  {"action": "screenshot", "path": "result.png"}
]
```

Run `agenticbrowser --help` or `agenticbrowser <command> --help` for full
option details.

---

## Supported action names (for `AgentBrowser.run_action`)

| Action | Required params | Optional params |
|--------|-----------------|-----------------|
| `navigate` | `url` | `wait_until` |
| `back` | — | — |
| `forward` | — | — |
| `reload` | — | — |
| `click` | `selector` | `timeout` |
| `type` | `selector`, `text` | `clear`, `timeout` |
| `press` | `key` | `selector` |
| `hover` | `selector` | — |
| `select` | `selector`, `value` | — |
| `scroll` | — | `x`, `y` |
| `scroll_bottom` | — | — |
| `scroll_top` | — | — |
| `get_text` | — | `selector` |
| `get_html` | — | `selector` |
| `get_url` | — | — |
| `get_title` | — | — |
| `find` | `selector` | — |
| `find_links` | — | — |
| `find_buttons` | — | — |
| `find_inputs` | — | — |
| `screenshot` | — | `path`, `full_page` |
| `screenshot_b64` | — | — |
| `wait_for` | `selector` | `state`, `timeout` |
| `evaluate` | `script` | — |

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Run tests
pytest
```

---

## License

MIT
