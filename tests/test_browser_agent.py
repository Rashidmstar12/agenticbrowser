import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from browser_agent import STEP_SCHEMA, BrowserAgent


@pytest.fixture
def agent():
    a = BrowserAgent(headless=True)
    mock_page = MagicMock()
    mock_context = MagicMock()
    mock_browser = MagicMock()
    mock_playwright = MagicMock()

    # Async methods on page
    mock_page.goto = AsyncMock()
    mock_page.click = AsyncMock()
    mock_page.fill = AsyncMock()
    mock_page.type = AsyncMock()
    mock_page.press = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=None)
    mock_page.eval_on_selector = AsyncMock()
    mock_page.screenshot = AsyncMock()
    mock_page.inner_text = AsyncMock(return_value="some text")
    mock_page.get_attribute = AsyncMock(return_value="attr_value")
    mock_page.title = AsyncMock(return_value="Page Title")
    mock_page.hover = AsyncMock()
    mock_page.dblclick = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.select_option = AsyncMock()
    mock_page.check = AsyncMock()
    mock_page.uncheck = AsyncMock()
    mock_page.set_input_files = AsyncMock()
    mock_page.url = "https://example.com"

    # Async methods on context
    mock_context.cookies = AsyncMock(return_value=[{"name": "c", "value": "v"}])
    mock_context.add_cookies = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=MagicMock(url="https://new-tab.com"))

    # Async methods on browser
    mock_browser.close = AsyncMock()

    # Async methods on playwright
    mock_playwright.stop = AsyncMock()

    a.page = mock_page
    a.context = mock_context
    a.browser = mock_browser
    a._playwright = mock_playwright
    return a


# ── Instantiation ─────────────────────────────────────────────────────────────

def test_agent_headless_default():
    a = BrowserAgent()
    assert a.headless is True


def test_agent_headless_false():
    a = BrowserAgent(headless=False)
    assert a.headless is False


def test_agent_initial_state():
    a = BrowserAgent()
    assert a._tabs == []
    assert a.browser is None
    assert a.page is None
    assert a.context is None
    assert a._playwright is None


# ── start / stop ──────────────────────────────────────────────────────────────

async def test_start():
    with patch("browser_agent.async_playwright") as mock_ap:
        mock_pw = AsyncMock()
        mock_ap.return_value.start = AsyncMock(return_value=mock_pw)
        mock_browser = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_context = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_page = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        a = BrowserAgent()
        await a.start()
        assert a.browser == mock_browser
        assert a.page == mock_page
        assert a.context == mock_context


async def test_stop(agent):
    await agent.stop()
    agent.browser.close.assert_called_once()
    agent._playwright.stop.assert_called_once()


async def test_start_headless_false():
    with patch("browser_agent.async_playwright") as mock_ap:
        mock_pw = AsyncMock()
        mock_ap.return_value.start = AsyncMock(return_value=mock_pw)
        mock_browser = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_context = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_page = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        a = BrowserAgent(headless=False)
        await a.start()
        mock_pw.chromium.launch.assert_called_with(headless=False)


# ── execute_step: basic actions ───────────────────────────────────────────────

async def test_navigate(agent):
    result = await agent.execute_step({"action": "navigate", "url": "https://example.com"})
    assert result["success"] is True
    agent.page.goto.assert_called_with("https://example.com")


async def test_click(agent):
    result = await agent.execute_step({"action": "click", "selector": "#btn"})
    assert result["success"] is True
    agent.page.click.assert_called_with("#btn")


async def test_fill(agent):
    result = await agent.execute_step({"action": "fill", "selector": "#inp", "value": "hello"})
    assert result["success"] is True
    agent.page.fill.assert_called_with("#inp", "hello")


async def test_type_text(agent):
    result = await agent.execute_step({"action": "type_text", "selector": "#inp", "text": "world"})
    assert result["success"] is True
    agent.page.type.assert_called_with("#inp", "world")


async def test_clear(agent):
    result = await agent.execute_step({"action": "clear", "selector": "#inp"})
    assert result["success"] is True
    agent.page.fill.assert_called_with("#inp", "")


async def test_press_key(agent):
    result = await agent.execute_step({"action": "press_key", "key": "Enter"})
    assert result["success"] is True
    agent.page.press.assert_called_with("body", "Enter")


async def test_press_key_with_selector(agent):
    result = await agent.execute_step({"action": "press_key", "selector": "#inp", "key": "Tab"})
    assert result["success"] is True
    agent.page.press.assert_called_with("#inp", "Tab")


async def test_scroll(agent):
    result = await agent.execute_step({"action": "scroll", "x": 0, "y": 200})
    assert result["success"] is True
    agent.page.evaluate.assert_called_once()


async def test_scroll_to_element(agent):
    result = await agent.execute_step({"action": "scroll_to_element", "selector": "#el"})
    assert result["success"] is True
    agent.page.eval_on_selector.assert_called_with("#el", "el => el.scrollIntoView()")


async def test_screenshot(agent):
    result = await agent.execute_step({"action": "screenshot", "path": "out.png"})
    assert result["success"] is True
    agent.page.screenshot.assert_called_with(path="out.png")


async def test_screenshot_default_path(agent):
    result = await agent.execute_step({"action": "screenshot"})
    assert result["success"] is True
    agent.page.screenshot.assert_called_with(path="screenshot.png")


async def test_get_text(agent):
    agent.page.inner_text = AsyncMock(return_value="hello text")
    result = await agent.execute_step({"action": "get_text", "selector": "#el"})
    assert result["success"] is True
    assert result["text"] == "hello text"


async def test_get_attribute(agent):
    agent.page.get_attribute = AsyncMock(return_value="my_attr")
    result = await agent.execute_step({"action": "get_attribute", "selector": "#el", "name": "href"})
    assert result["success"] is True
    assert result["attribute"] == "my_attr"


async def test_get_url(agent):
    agent.page.url = "https://test.com"
    result = await agent.execute_step({"action": "get_url"})
    assert result["success"] is True
    assert result["url"] == "https://test.com"


async def test_get_title(agent):
    agent.page.title = AsyncMock(return_value="My Title")
    result = await agent.execute_step({"action": "get_title"})
    assert result["success"] is True
    assert result["title"] == "My Title"


async def test_hover(agent):
    result = await agent.execute_step({"action": "hover", "selector": "#el"})
    assert result["success"] is True
    agent.page.hover.assert_called_with("#el")


async def test_double_click(agent):
    result = await agent.execute_step({"action": "double_click", "selector": "#el"})
    assert result["success"] is True
    agent.page.dblclick.assert_called_with("#el")


async def test_right_click(agent):
    result = await agent.execute_step({"action": "right_click", "selector": "#el"})
    assert result["success"] is True
    agent.page.click.assert_called_with("#el", button="right")


async def test_wait_for_selector(agent):
    result = await agent.execute_step({"action": "wait_for_selector", "selector": "#el"})
    assert result["success"] is True
    agent.page.wait_for_selector.assert_called_with("#el")


async def test_wait_text(agent):
    result = await agent.execute_step({"action": "wait_text", "text": "hello"})
    assert result["success"] is True
    agent.page.wait_for_selector.assert_called_with("text=hello")


async def test_select_option(agent):
    result = await agent.execute_step({"action": "select_option", "selector": "#sel", "value": "opt1"})
    assert result["success"] is True
    agent.page.select_option.assert_called_with("#sel", "opt1")


async def test_check(agent):
    result = await agent.execute_step({"action": "check", "selector": "#chk"})
    assert result["success"] is True
    agent.page.check.assert_called_with("#chk")


async def test_uncheck(agent):
    result = await agent.execute_step({"action": "uncheck", "selector": "#chk"})
    assert result["success"] is True
    agent.page.uncheck.assert_called_with("#chk")


async def test_upload_file(agent):
    result = await agent.execute_step({"action": "upload_file", "selector": "#file", "path": "test.txt"})
    assert result["success"] is True
    agent.page.set_input_files.assert_called_with("#file", "test.txt")


async def test_extract_links(agent):
    agent.page.evaluate = AsyncMock(return_value=["https://a.com", "https://b.com"])
    result = await agent.execute_step({"action": "extract_links"})
    assert result["success"] is True
    assert result["links"] == ["https://a.com", "https://b.com"]


async def test_extract_table(agent):
    agent.page.evaluate = AsyncMock(return_value=[["h1", "h2"], ["v1", "v2"]])
    result = await agent.execute_step({"action": "extract_table"})
    assert result["success"] is True
    assert result["rows"] == [["h1", "h2"], ["v1", "v2"]]


async def test_assert_text_pass(agent):
    agent.page.inner_text = AsyncMock(return_value="hello world")
    result = await agent.execute_step({"action": "assert_text", "selector": "#el", "expected": "hello"})
    assert result["success"] is True


async def test_assert_text_fail(agent):
    agent.page.inner_text = AsyncMock(return_value="goodbye")
    result = await agent.execute_step({"action": "assert_text", "selector": "#el", "expected": "hello"})
    assert result["success"] is False
    assert "error" in result


async def test_assert_url_pass(agent):
    agent.page.url = "https://example.com"
    result = await agent.execute_step({"action": "assert_url", "expected": "https://example.com"})
    assert result["success"] is True


async def test_assert_url_fail(agent):
    agent.page.url = "https://example.com"
    result = await agent.execute_step({"action": "assert_url", "expected": "https://other.com"})
    assert result["success"] is False


async def test_save_cookies(agent, tmp_path):
    agent.context.cookies = AsyncMock(return_value=[{"name": "a", "value": "1"}])
    cookie_file = str(tmp_path / "cookies.json")
    result = await agent.execute_step({"action": "save_cookies", "path": cookie_file})
    assert result["success"] is True
    with open(cookie_file) as f:
        cookies = json.load(f)
    assert cookies == [{"name": "a", "value": "1"}]


async def test_load_cookies(agent, tmp_path):
    cookie_file = tmp_path / "cookies.json"
    cookies = [{"name": "a", "value": "1"}]
    cookie_file.write_text(json.dumps(cookies))
    result = await agent.execute_step({"action": "load_cookies", "path": str(cookie_file)})
    assert result["success"] is True
    agent.context.add_cookies.assert_called_with(cookies)


# ── Tab management ────────────────────────────────────────────────────────────

async def test_new_tab(agent):
    mock_new_page = MagicMock(url="https://new.com")
    agent.context.new_page = AsyncMock(return_value=mock_new_page)
    result = await agent.execute_step({"action": "new_tab"})
    assert result["success"] is True
    assert mock_new_page in agent._tabs


async def test_switch_tab(agent):
    mock_tab = MagicMock(url="https://tab1.com")
    agent._tabs = [mock_tab]
    result = await agent.execute_step({"action": "switch_tab", "index": 0})
    assert result["success"] is True
    assert agent.page == mock_tab


async def test_close_tab(agent):
    mock_tab = MagicMock()
    mock_tab.close = AsyncMock()
    agent._tabs = [mock_tab]
    result = await agent.execute_step({"action": "close_tab", "index": 0})
    assert result["success"] is True
    mock_tab.close.assert_called_once()


async def test_list_tabs(agent):
    tab1 = MagicMock(url="https://tab1.com")
    tab2 = MagicMock(url="https://tab2.com")
    agent._tabs = [tab1, tab2]
    result = await agent.execute_step({"action": "list_tabs"})
    assert result["success"] is True
    assert result["tabs"] == ["https://tab1.com", "https://tab2.com"]


async def test_list_tabs_empty(agent):
    agent._tabs = []
    result = await agent.execute_step({"action": "list_tabs"})
    assert result["success"] is True
    assert result["tabs"] == []


# ── Error handling ────────────────────────────────────────────────────────────

async def test_unknown_action(agent):
    result = await agent.execute_step({"action": "fly_to_moon"})
    assert result["success"] is False
    assert "fly_to_moon" in result["error"]


async def test_exception_returns_failure(agent):
    agent.page.goto = AsyncMock(side_effect=RuntimeError("network error"))
    result = await agent.execute_step({"action": "navigate", "url": "https://example.com"})
    assert result["success"] is False
    assert "network error" in result["error"]


# ── run_task ──────────────────────────────────────────────────────────────────

async def test_run_task_empty(agent):
    results = await agent.run_task([])
    assert results == []


async def test_run_task_single(agent):
    results = await agent.run_task([{"action": "navigate", "url": "https://example.com"}])
    assert len(results) == 1
    assert results[0]["success"] is True


async def test_run_task_multiple(agent):
    results = await agent.run_task([
        {"action": "navigate", "url": "https://example.com"},
        {"action": "click", "selector": "#btn"},
    ])
    assert len(results) == 2
    assert all(r["success"] for r in results)


async def test_run_task_continues_after_failure(agent):
    agent.page.goto = AsyncMock(side_effect=RuntimeError("fail"))
    results = await agent.run_task([
        {"action": "navigate", "url": "https://example.com"},
        {"action": "click", "selector": "#btn"},
    ])
    assert len(results) == 2
    assert results[0]["success"] is False
    assert results[1]["success"] is True


# ── STEP_SCHEMA coverage ──────────────────────────────────────────────────────

def test_step_schema_count():
    assert len(STEP_SCHEMA) == 32
