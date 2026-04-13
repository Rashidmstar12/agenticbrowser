"""
Tests for Tier-1 features:
  T1-1  Enhanced self-healing selectors  (browser_agent._generate_dynamic_fallbacks
                                           + resolve_selector dynamic tier)
  T1-2  Vision / screenshot grounding    (task_planner.TaskPlanner.vision_plan +
                                           _call_vision_openai)
  T1-3  Per-request session isolation    (api_server X-Agent-Id / ?agent_id routing)
  T1-4  Vision API routes                (POST /task/vision_plan, POST /task/vision_run)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

import api_server
from api_server import app
from browser_agent import BrowserAgent, _generate_dynamic_fallbacks
from task_planner import (
    StepValidationError,
    TaskPlanner,
    _call_vision_anthropic,
    _call_vision_gemini,
    _call_vision_ollama,
    _call_vision_openai,
)

# ---------------------------------------------------------------------------
# Helpers shared across test sections
# ---------------------------------------------------------------------------

def _make_agent(**kwargs) -> tuple[BrowserAgent, MagicMock, MagicMock]:
    """Return (agent, mock_page, mock_context) with Playwright fully mocked."""
    agent = BrowserAgent(**kwargs)
    page = MagicMock()
    page.url = "https://example.com"
    page.title.return_value = "Test"
    page.query_selector.return_value = None  # nothing found by default
    locator_mock = MagicMock()
    locator_mock.count.return_value = 0
    page.locator.return_value = locator_mock
    context = MagicMock()
    agent._page = page
    agent._context = context
    agent._pages = [page]
    return agent, page, context


# ---------------------------------------------------------------------------
# T1-1  _generate_dynamic_fallbacks
# ---------------------------------------------------------------------------

class TestGenerateDynamicFallbacks:
    def test_empty_selector_returns_empty(self):
        assert _generate_dynamic_fallbacks("body") == []

    def test_aria_label_generates_label_and_text(self):
        result = _generate_dynamic_fallbacks('[aria-label="Search"]')
        assert "label=Search" in result
        assert "text=Search" in result

    def test_placeholder_generates_placeholder(self):
        result = _generate_dynamic_fallbacks('[placeholder="Enter name"]')
        assert "placeholder=Enter name" in result

    def test_input_name_generates_label_and_placeholder(self):
        result = _generate_dynamic_fallbacks("input[name='q']")
        assert "label=q" in result
        assert "placeholder=q" in result

    def test_textarea_name_generates_label_and_placeholder(self):
        result = _generate_dynamic_fallbacks("textarea[name='message']")
        assert "label=message" in result
        assert "placeholder=message" in result

    def test_data_testid_generates_text(self):
        result = _generate_dynamic_fallbacks('[data-testid="submit-btn"]')
        assert "text=submit-btn" in result

    def test_title_attr_generates_title(self):
        result = _generate_dynamic_fallbacks('[title="Close dialog"]')
        assert "title=Close dialog" in result

    def test_class_selector_generates_class_contains(self):
        result = _generate_dynamic_fallbacks(".btn.submit-button")
        assert any("[class*='submit-button']" in r or "[class*='btn']" in r for r in result)

    def test_multiple_hints_combined(self):
        sel = "input[aria-label='Email'][placeholder='you@example.com']"
        result = _generate_dynamic_fallbacks(sel)
        assert "label=Email" in result
        assert "placeholder=you@example.com" in result

    def test_no_duplicates_for_simple_selector(self):
        result = _generate_dynamic_fallbacks("#login-btn")
        # No attributes → only possible class match (there are none)
        assert result == []


# ---------------------------------------------------------------------------
# T1-1  resolve_selector dynamic fallback tier
# ---------------------------------------------------------------------------

class TestResolveSelectorDynamicFallback:
    def test_primary_found_returns_immediately(self):
        agent, page, _ = _make_agent()
        el = MagicMock()
        page.query_selector.side_effect = lambda sel: el if sel == "input#q" else None
        result = agent.resolve_selector("input#q")
        assert result == "input#q"

    def test_falls_back_to_dynamic_label(self):
        """When CSS selector fails, a label= fallback derived from aria-label is used."""
        agent, page, _ = _make_agent()
        # CSS selector fails; semantic locator succeeds
        page.query_selector.return_value = None
        locator_with_match = MagicMock()
        locator_with_match.count.return_value = 1

        def _locator_side(sel):
            if sel == "label=Search":
                return locator_with_match
            fallback = MagicMock()
            fallback.count.return_value = 0
            return fallback

        page.locator.side_effect = _locator_side
        result = agent.resolve_selector('[aria-label="Search"]')
        assert result == "label=Search"

    def test_falls_back_to_dynamic_placeholder(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = None
        locator_match = MagicMock()
        locator_match.count.return_value = 1

        def _locator(sel):
            if sel == "placeholder=Search query":
                return locator_match
            m = MagicMock()
            m.count.return_value = 0
            return m

        page.locator.side_effect = _locator
        result = agent.resolve_selector("[placeholder='Search query']")
        assert result == "placeholder=Search query"

    def test_error_lists_dynamic_fallbacks_in_message(self):
        """ValueError message must include the dynamic fallbacks that were tried."""
        agent, page, _ = _make_agent()
        page.query_selector.return_value = None
        locator_mock = MagicMock()
        locator_mock.count.return_value = 0
        page.locator.return_value = locator_mock
        with pytest.raises(ValueError) as exc_info:
            agent.resolve_selector('[aria-label="Nope"]')
        msg = str(exc_info.value)
        assert "label=Nope" in msg or "text=Nope" in msg

    def test_selector_with_no_dynamic_fallbacks_raises_cleanly(self):
        agent, page, _ = _make_agent()
        page.query_selector.return_value = None
        with pytest.raises(ValueError, match="No element found"):
            agent.resolve_selector("#nonexistent-id")

    def test_dynamic_class_fallback_used(self):
        agent, page, _ = _make_agent()
        # CSS fails; class*= partial match succeeds
        def _qs(sel):
            if sel == "[class*='submit-btn']":
                return MagicMock()  # found
            return None

        page.query_selector.side_effect = _qs
        result = agent.resolve_selector(".submit-btn")
        assert result == "[class*='submit-btn']"


# ---------------------------------------------------------------------------
# T1-2  _call_vision_openai
# ---------------------------------------------------------------------------

def _make_openai_mock(content: str) -> tuple[MagicMock, MagicMock]:
    """Return (mock_openai_module, mock_client) wired to return *content*."""
    fake_response = MagicMock()
    fake_response.choices[0].message.content = content
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_response
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai, mock_client


class TestCallVisionOpenai:
    def _run(self, content: str, intent: str = "open example.com", b64: str = "base64=="):
        import sys
        mock_openai, mock_client = _make_openai_mock(content)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                return _call_vision_openai(intent, b64), mock_client

    def test_returns_steps_from_valid_json_array(self):
        steps, _ = self._run('[{"action": "navigate", "url": "https://example.com"}]')
        assert steps[0]["action"] == "navigate"

    def test_strips_markdown_fences(self):
        content = '```json\n[{"action": "navigate", "url": "https://example.com"}]\n```'
        steps, _ = self._run(content)
        assert len(steps) == 1

    def test_raises_on_no_json_array(self):
        import sys
        mock_openai, _ = _make_openai_mock("Sorry, I cannot help.")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                with pytest.raises(StepValidationError, match="no JSON array"):
                    _call_vision_openai("do something", "base64==")

    def test_uses_custom_model_env(self):
        """OPENAI_VISION_MODEL env var should override the default model."""
        import sys
        mock_openai, mock_client = _make_openai_mock(
            '[{"action": "navigate", "url": "https://example.com"}]'
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test", "OPENAI_VISION_MODEL": "gpt-4-turbo"}):
            with patch.dict(sys.modules, {"openai": mock_openai}):
                _call_vision_openai("open example.com", "b64")
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4-turbo"


# ---------------------------------------------------------------------------
# T1-2  TaskPlanner.vision_plan
# ---------------------------------------------------------------------------

class TestVisionPlan:
    def test_falls_back_to_plan_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        planner = TaskPlanner()
        agent = MagicMock()
        with patch.object(planner, "plan", return_value=[{"action": "navigate", "url": "https://google.com"}]) as mock_plan:
            result = planner.vision_plan("go to google", agent)
        mock_plan.assert_called_once_with("go to google")
        assert result[0]["action"] == "navigate"

    def test_calls_vision_openai_when_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        planner = TaskPlanner()
        agent = MagicMock()
        agent.screenshot.return_value = {"base64": "abc123"}
        expected_steps = [{"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        with patch("task_planner._call_vision_openai", return_value=expected_steps) as mock_vision:
            result = planner.vision_plan("go to example", agent)
        mock_vision.assert_called_once_with("go to example", "abc123")
        assert result == expected_steps

    def test_falls_back_when_screenshot_fails(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        planner = TaskPlanner()
        agent = MagicMock()
        agent.screenshot.side_effect = RuntimeError("browser not started")
        with patch.object(planner, "plan", return_value=[{"action": "navigate", "url": "https://google.com"}]) as mock_plan:
            result = planner.vision_plan("go to google", agent)
        mock_plan.assert_called_once()
        assert result[0]["action"] == "navigate"

    def test_falls_back_when_screenshot_is_empty(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        planner = TaskPlanner()
        agent = MagicMock()
        agent.screenshot.return_value = {"base64": ""}
        with patch.object(planner, "plan", return_value=[{"action": "navigate", "url": "https://google.com"}]) as mock_plan:
            planner.vision_plan("go to google", agent)
        mock_plan.assert_called_once()

    def test_falls_back_when_vision_llm_raises(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        planner = TaskPlanner()
        agent = MagicMock()
        agent.screenshot.return_value = {"base64": "abc123"}
        with patch("task_planner._call_vision_openai", side_effect=Exception("network error")):
            with patch.object(planner, "plan", return_value=[{"action": "navigate", "url": "https://example.com"}]) as mock_plan:
                result = planner.vision_plan("go to example", agent)
        mock_plan.assert_called_once()
        assert result[0]["action"] == "navigate"

    def test_strips_intent(self, monkeypatch):
        """Leading/trailing whitespace in intent is stripped before processing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        planner = TaskPlanner()
        agent = MagicMock()
        with patch.object(planner, "plan", return_value=[{"action": "navigate", "url": "https://google.com"}]) as mock_plan:
            planner.vision_plan("  go to google  ", agent)
        mock_plan.assert_called_once_with("go to google")


# ---------------------------------------------------------------------------
# T1-3  Per-request session isolation (X-Agent-Id header / ?agent_id= param)
# ---------------------------------------------------------------------------

def _make_mock_agent(*, page_active: bool = True) -> MagicMock:
    agent = MagicMock()
    agent._page = MagicMock() if page_active else None
    agent._pages = [MagicMock()] if page_active else []
    agent.headless = True
    agent.navigate.return_value = {"url": "https://example.com"}
    agent.get_page_info.return_value = {"url": "https://example.com", "title": "Ex"}
    return agent


class TestSessionIsolation:
    def setup_method(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        # Patch global singletons to avoid "not started" 400 errors
        self._agent_patch = patch.object(api_server, "_agent", _make_mock_agent())
        self._planner_patch = patch.object(api_server, "_planner", MagicMock())
        self._agent_patch.start()
        self._planner_patch.start()

    def teardown_method(self):
        self._agent_patch.stop()
        self._planner_patch.stop()
        # Clean up pool
        with api_server._pool_lock:
            api_server._agent_pool.clear()
            api_server._planner_pool.clear()

    def test_no_header_uses_default_agent(self):
        """Without X-Agent-Id the global _agent is used."""
        resp = self.client.post("/navigate", json={"url": "https://example.com"})
        assert resp.status_code == 200

    def test_invalid_agent_id_returns_404(self):
        """Requesting a non-existent agent ID returns 404."""
        resp = self.client.post(
            "/navigate",
            json={"url": "https://example.com"},
            headers={"X-Agent-Id": "nonexistent-agent"},
        )
        assert resp.status_code == 404

    def test_valid_pool_agent_routed_correctly(self):
        """When X-Agent-Id matches a pooled agent, that agent is used."""
        pooled = _make_mock_agent()
        planner = MagicMock()
        with api_server._pool_lock:
            api_server._agent_pool["test-agent"] = pooled
            api_server._planner_pool["test-agent"] = planner

        resp = self.client.post(
            "/navigate",
            json={"url": "https://example.com"},
            headers={"X-Agent-Id": "test-agent"},
        )
        assert resp.status_code == 200
        pooled.navigate.assert_called_once()

    def test_agent_id_query_param_works(self):
        """?agent_id= query parameter routes the same as X-Agent-Id header."""
        pooled = _make_mock_agent()
        planner = MagicMock()
        with api_server._pool_lock:
            api_server._agent_pool["qp-agent"] = pooled
            api_server._planner_pool["qp-agent"] = planner

        resp = self.client.post(
            "/navigate?agent_id=qp-agent",
            json={"url": "https://example.com"},
        )
        assert resp.status_code == 200
        pooled.navigate.assert_called_once()

    def test_multiple_routes_route_to_pool_agent(self):
        """Multiple different routes (click, fill, evaluate) all respect X-Agent-Id."""
        pooled = _make_mock_agent()
        pooled.click.return_value = {"clicked": "btn"}
        pooled.fill.return_value = {"filled": "val"}
        pooled.evaluate.return_value = {"result": 42}
        planner = MagicMock()
        with api_server._pool_lock:
            api_server._agent_pool["multi-agent"] = pooled
            api_server._planner_pool["multi-agent"] = planner

        headers = {"X-Agent-Id": "multi-agent"}
        self.client.post("/click", json={"selector": "btn"}, headers=headers)
        self.client.post("/fill", json={"selector": "inp", "value": "val"}, headers=headers)
        self.client.post("/evaluate", json={"script": "1+1"}, headers=headers)

        pooled.click.assert_called_once()
        pooled.fill.assert_called_once()
        pooled.evaluate.assert_called_once()


# ---------------------------------------------------------------------------
# T1-4  Vision API routes
# ---------------------------------------------------------------------------

class TestVisionApiRoutes:
    def setup_method(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_vision_plan_returns_400_when_no_session(self):
        with patch.object(api_server, "_agent", None), \
             patch.object(api_server, "_planner", None):
            resp = self.client.post("/task/vision_plan", json={"intent": "click login"})
        assert resp.status_code == 400

    def test_vision_run_returns_400_when_no_session(self):
        with patch.object(api_server, "_agent", None), \
             patch.object(api_server, "_planner", None):
            resp = self.client.post("/task/vision_run", json={"intent": "click login"})
        assert resp.status_code == 400

    def test_vision_plan_returns_steps(self):
        mock_agent = _make_mock_agent()
        mock_planner = MagicMock()
        expected = [{"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        mock_planner.vision_plan.return_value = expected

        with patch.object(api_server, "_agent", mock_agent), \
             patch.object(api_server, "_planner", mock_planner):
            resp = self.client.post("/task/vision_plan", json={"intent": "go to example"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["vision"] is True
        assert body["count"] == 1
        assert body["steps"][0]["action"] == "navigate"

    def test_vision_run_executes_steps_and_returns_results(self):
        mock_agent = _make_mock_agent()
        mock_planner = MagicMock()
        steps = [{"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        mock_planner.vision_plan.return_value = steps
        mock_planner.execute.return_value = [{"step": 0, "action": "navigate", "status": "ok", "result": {}}]

        with patch.object(api_server, "_agent", mock_agent), \
             patch.object(api_server, "_planner", mock_planner):
            resp = self.client.post("/task/vision_run", json={"intent": "go to example"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["vision"] is True
        assert body["success"] is True

    def test_vision_run_returns_422_on_step_failure(self):
        mock_agent = _make_mock_agent()
        mock_planner = MagicMock()
        steps = [{"action": "click", "selector": "btn", "timeout": None, "retry": 0, "retry_delay": 1.0}]
        mock_planner.vision_plan.return_value = steps
        mock_planner.execute.return_value = [{"step": 0, "action": "click", "status": "error", "error": "element not found"}]

        with patch.object(api_server, "_agent", mock_agent), \
             patch.object(api_server, "_planner", mock_planner):
            resp = self.client.post("/task/vision_run", json={"intent": "click something"})

        assert resp.status_code == 422

    def test_vision_plan_422_when_planner_raises(self):
        mock_agent = _make_mock_agent()
        mock_planner = MagicMock()
        mock_planner.vision_plan.side_effect = ValueError("no LLM configured")

        with patch.object(api_server, "_agent", mock_agent), \
             patch.object(api_server, "_planner", mock_planner):
            resp = self.client.post("/task/vision_plan", json={"intent": "do something"})

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# T1-2  _call_vision_anthropic
# ---------------------------------------------------------------------------

class TestCallVisionAnthropic:
    def _run(self, content: str, intent: str = "open example.com", b64: str = "base64=="):
        import sys

        fake_content = MagicMock()
        fake_content.text = content
        fake_message = MagicMock()
        fake_message.content = [fake_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_message
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-test"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                return _call_vision_anthropic(intent, b64), mock_client

    def test_returns_steps_from_valid_json_array(self):
        steps, _ = self._run('[{"action": "navigate", "url": "https://example.com"}]')
        assert steps[0]["action"] == "navigate"

    def test_strips_markdown_fences(self):
        content = '```json\n[{"action": "navigate", "url": "https://example.com"}]\n```'
        steps, _ = self._run(content)
        assert len(steps) == 1

    def test_raises_on_no_json_array(self):
        import sys

        fake_content = MagicMock()
        fake_content.text = "Sorry, I cannot help."
        fake_message = MagicMock()
        fake_message.content = [fake_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_message
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-test"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                with pytest.raises(StepValidationError, match="no JSON array"):
                    _call_vision_anthropic("do something", "base64==")

    def test_uses_custom_model_env(self):
        import sys

        fake_content = MagicMock()
        fake_content.text = '[{"action": "navigate", "url": "https://example.com"}]'
        fake_message = MagicMock()
        fake_message.content = [fake_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_message
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "ant-test", "ANTHROPIC_VISION_MODEL": "claude-3-opus-20240229"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                _call_vision_anthropic("open example.com", "b64")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-3-opus-20240229"


# ---------------------------------------------------------------------------
# T1-2  _call_vision_gemini
# ---------------------------------------------------------------------------

class TestCallVisionGemini:
    def _run(self, response_text: str, intent: str = "open example.com", b64: str = "AAAA"):
        fake_response = MagicMock()
        fake_response.text = response_text
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = fake_response

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "goog-test"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel", return_value=mock_model_instance):
                    result = _call_vision_gemini(intent, b64)
        return result, mock_model_instance

    def test_returns_steps_from_valid_json_array(self):
        steps, _ = self._run('[{"action": "navigate", "url": "https://example.com"}]')
        assert steps[0]["action"] == "navigate"

    def test_strips_markdown_fences(self):
        content = '```json\n[{"action": "navigate", "url": "https://example.com"}]\n```'
        steps, _ = self._run(content)
        assert len(steps) == 1

    def test_raises_on_no_json_array(self):
        fake_response = MagicMock()
        fake_response.text = "I cannot help."
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = fake_response

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "goog-test"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel", return_value=mock_model_instance):
                    with pytest.raises(StepValidationError, match="no JSON array"):
                        _call_vision_gemini("do something", "AAAA")

    def test_uses_custom_model_env(self):
        fake_response = MagicMock()
        fake_response.text = '[{"action": "navigate", "url": "https://example.com"}]'
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = fake_response

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "goog-test", "GEMINI_VISION_MODEL": "gemini-1.5-flash"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel", return_value=mock_model_instance) as mock_gm:
                    _call_vision_gemini("open example.com", "AAAA")

        call_kwargs = mock_gm.call_args
        assert call_kwargs.kwargs.get("model_name") == "gemini-1.5-flash"


# ---------------------------------------------------------------------------
# T1-2  _call_vision_ollama
# ---------------------------------------------------------------------------

class TestCallVisionOllama:
    def _run(self, response_text: str, intent: str = "open example.com", b64: str = "base64=="):
        import json as _json

        body = _json.dumps({"response": response_text}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = body

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"}):
                return _call_vision_ollama(intent, b64)

    def test_returns_steps_from_valid_json_array(self):
        steps = self._run('[{"action": "navigate", "url": "https://example.com"}]')
        assert steps[0]["action"] == "navigate"

    def test_strips_markdown_fences(self):
        content = '```json\n[{"action": "navigate", "url": "https://example.com"}]\n```'
        steps = self._run(content)
        assert len(steps) == 1

    def test_raises_on_no_json_array(self):
        body = b'{"response": "I cannot help."}'
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = body

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(StepValidationError, match="no JSON array"):
                _call_vision_ollama("do something", "base64==")

    def test_uses_custom_model_env(self):
        import json as _json

        body = _json.dumps({"response": '[{"action": "navigate", "url": "https://example.com"}]'}).encode()
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = body

        captured = {}

        def _fake_urlopen(req, timeout=None):
            captured["payload"] = _json.loads(req.data)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            with patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434", "OLLAMA_VISION_MODEL": "bakllava"}):
                _call_vision_ollama("open example.com", "b64")

        assert captured["payload"]["model"] == "bakllava"


# ---------------------------------------------------------------------------
# T1-2  vision_plan provider dispatch
# ---------------------------------------------------------------------------

class TestVisionPlanProviderDispatch:
    def test_uses_openai_when_key_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        planner = TaskPlanner()
        agent = MagicMock()
        agent.screenshot.return_value = {"base64": "abc123"}
        expected = [{"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        with patch("task_planner._call_vision_openai", return_value=expected) as mock_fn:
            result = planner.vision_plan("go to example", agent)
        mock_fn.assert_called_once_with("go to example", "abc123")
        assert result == expected

    def test_uses_anthropic_when_only_anthropic_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        planner = TaskPlanner()
        agent = MagicMock()
        agent.screenshot.return_value = {"base64": "abc123"}
        expected = [{"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        with patch("task_planner._call_vision_anthropic", return_value=expected) as mock_fn:
            result = planner.vision_plan("go to example", agent)
        mock_fn.assert_called_once_with("go to example", "abc123")
        assert result == expected

    def test_uses_gemini_when_only_google_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "goog-test")
        planner = TaskPlanner()
        agent = MagicMock()
        agent.screenshot.return_value = {"base64": "abc123"}
        expected = [{"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        with patch("task_planner._call_vision_gemini", return_value=expected) as mock_fn:
            result = planner.vision_plan("go to example", agent)
        mock_fn.assert_called_once_with("go to example", "abc123")
        assert result == expected

    def test_explicit_provider_overrides_auto_detect(self, monkeypatch):
        """Explicit provider= overrides even when OPENAI_API_KEY is present."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        planner = TaskPlanner()
        agent = MagicMock()
        agent.screenshot.return_value = {"base64": "abc123"}
        expected = [{"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        with patch("task_planner._call_vision_anthropic", return_value=expected) as mock_fn:
            result = planner.vision_plan("go to example", agent, provider="anthropic")
        mock_fn.assert_called_once_with("go to example", "abc123")
        assert result == expected

    def test_falls_back_to_plan_when_no_provider(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        planner = TaskPlanner()
        agent = MagicMock()
        fallback = [{"action": "navigate", "url": "https://google.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        with patch.object(planner, "plan", return_value=fallback) as mock_plan:
            with patch("task_planner._ollama_running", return_value=False):
                result = planner.vision_plan("go to google", agent)
        mock_plan.assert_called_once_with("go to google")
        assert result == fallback

    def test_unknown_provider_falls_back_to_plan(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        planner = TaskPlanner()
        agent = MagicMock()
        agent.screenshot.return_value = {"base64": "abc123"}
        fallback = [{"action": "navigate", "url": "https://google.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        with patch.object(planner, "plan", return_value=fallback) as mock_plan:
            result = planner.vision_plan("go to google", agent, provider="unknown_model")
        mock_plan.assert_called_once_with("go to google")
        assert result == fallback


# ---------------------------------------------------------------------------
# T1-4  Vision API routes — provider field
# ---------------------------------------------------------------------------

class TestVisionApiRoutesProvider:
    def setup_method(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_vision_plan_passes_provider_to_planner(self):
        mock_agent = _make_mock_agent()
        mock_planner = MagicMock()
        expected = [{"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        mock_planner.vision_plan.return_value = expected

        with patch.object(api_server, "_agent", mock_agent), \
             patch.object(api_server, "_planner", mock_planner):
            resp = self.client.post(
                "/task/vision_plan",
                json={"intent": "go to example", "provider": "anthropic"},
            )

        assert resp.status_code == 200
        mock_planner.vision_plan.assert_called_once_with(
            "go to example", mock_agent, provider="anthropic"
        )

    def test_vision_run_passes_provider_to_planner(self):
        mock_agent = _make_mock_agent()
        mock_planner = MagicMock()
        steps = [{"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}]
        mock_planner.vision_plan.return_value = steps
        mock_planner.execute.return_value = [{"step": 0, "action": "navigate", "status": "ok", "result": {}}]

        with patch.object(api_server, "_agent", mock_agent), \
             patch.object(api_server, "_planner", mock_planner):
            resp = self.client.post(
                "/task/vision_run",
                json={"intent": "go to example", "provider": "gemini"},
            )

        assert resp.status_code == 200
        mock_planner.vision_plan.assert_called_once_with(
            "go to example", mock_agent, provider="gemini"
        )

    def test_vision_plan_no_provider_passes_none(self):
        mock_agent = _make_mock_agent()
        mock_planner = MagicMock()
        mock_planner.vision_plan.return_value = [
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded", "retry": 0, "retry_delay": 1.0}
        ]

        with patch.object(api_server, "_agent", mock_agent), \
             patch.object(api_server, "_planner", mock_planner):
            self.client.post("/task/vision_plan", json={"intent": "go to example"})

        mock_planner.vision_plan.assert_called_once_with(
            "go to example", mock_agent, provider=None
        )
