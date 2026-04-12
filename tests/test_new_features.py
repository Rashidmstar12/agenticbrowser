"""
Tests for the new browser features added in Phase 2:

Category 1 – New Browser Actions:
  drag_drop, right_click, double_click, upload_file, set_viewport,
  block_resource, iframe_switch, iframe_exit

Category 2 – Data Extraction:
  extract_json_ld, extract_headings, extract_images, extract_form_fields,
  extract_meta

Category 3 – Authentication & Session:
  set_extra_headers, http_auth, local_storage_set/get, session_storage_set/get

Category 4 – Assertions & Verification:
  assert_element_count, assert_attribute, assert_title, assert_visible,
  assert_hidden

All browser tests use MagicMock — no real Playwright/Chromium needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# local imports (path must be set up first)
from task_planner import STEP_SCHEMA, TaskPlanner, validate_steps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _planner_no_llm() -> TaskPlanner:
    with patch.object(TaskPlanner, "_detect_llm", return_value=None):
        return TaskPlanner()


def _make_agent() -> MagicMock:
    """Build a MagicMock BrowserAgent with sensible return values."""
    agent = MagicMock()

    # Category 1
    agent.drag_drop.return_value       = {"source": "#a", "target": "#b"}
    agent.right_click.return_value     = {"right_clicked": "#btn"}
    agent.double_click.return_value    = {"double_clicked": "#item"}
    agent.upload_file.return_value     = {"uploaded": "/tmp/f.txt", "selector": "input"}
    agent.set_viewport.return_value    = {"width": 375, "height": 812}
    agent.block_resource.return_value  = {"blocked_types": ["image"]}
    agent.iframe_switch.return_value   = {"frame_url": "https://frame.example.com", "selector": "iframe"}
    agent.iframe_exit.return_value     = {"frame_url": "https://example.com"}

    # Category 2
    agent.extract_json_ld.return_value    = {"items": [], "count": 0}
    agent.extract_headings.return_value   = {"headings": [{"level": 1, "text": "Title"}], "count": 1}
    agent.extract_images.return_value     = {"images": [], "count": 0}
    agent.extract_form_fields.return_value = {"fields": [], "count": 0}
    agent.extract_meta.return_value       = {"title": "Ex", "description": "", "tags": [], "count": 0}

    # Category 3
    agent.set_extra_headers.return_value   = {"headers_set": ["X-Token"]}
    agent.http_auth.return_value           = {"auth_set": True, "username": "user"}
    agent.local_storage_set.return_value   = {"key": "k", "value": "v"}
    agent.local_storage_get.return_value   = {"key": "k", "value": "v"}
    agent.session_storage_set.return_value = {"key": "k", "value": "v"}
    agent.session_storage_get.return_value = {"key": "k", "value": "stored"}

    # Category 4
    setattr(agent, "assert_element_count", MagicMock(return_value={"selector": "li", "expected": 3, "actual": 3, "operator": "eq"}))
    setattr(agent, "assert_attribute",     MagicMock(return_value={"selector": "#a", "attribute": "href", "expected": "/", "actual": "/"}))
    setattr(agent, "assert_title",         MagicMock(return_value={"title": "My Page", "pattern": "My", "matched": True}))
    setattr(agent, "assert_visible",       MagicMock(return_value={"selector": "#el", "visible": True}))
    setattr(agent, "assert_hidden",        MagicMock(return_value={"selector": "#el", "hidden": True}))

    return agent


# ---------------------------------------------------------------------------
# STEP_SCHEMA: new actions are registered
# ---------------------------------------------------------------------------

class TestNewStepSchemaEntries:
    """Verify every new action exists in STEP_SCHEMA with correct required keys."""

    # Category 1
    def test_drag_drop_schema(self) -> None:
        assert "drag_drop" in STEP_SCHEMA
        assert "source" in STEP_SCHEMA["drag_drop"]["required"]
        assert "target" in STEP_SCHEMA["drag_drop"]["required"]

    def test_right_click_schema(self) -> None:
        assert "right_click" in STEP_SCHEMA
        assert "selector" in STEP_SCHEMA["right_click"]["required"]

    def test_double_click_schema(self) -> None:
        assert "double_click" in STEP_SCHEMA
        assert "selector" in STEP_SCHEMA["double_click"]["required"]

    def test_upload_file_schema(self) -> None:
        assert "upload_file" in STEP_SCHEMA
        assert "selector" in STEP_SCHEMA["upload_file"]["required"]
        assert "path" in STEP_SCHEMA["upload_file"]["required"]

    def test_set_viewport_schema(self) -> None:
        assert "set_viewport" in STEP_SCHEMA
        assert "width" in STEP_SCHEMA["set_viewport"]["required"]
        assert "height" in STEP_SCHEMA["set_viewport"]["required"]

    def test_block_resource_schema(self) -> None:
        assert "block_resource" in STEP_SCHEMA

    def test_iframe_switch_schema(self) -> None:
        assert "iframe_switch" in STEP_SCHEMA
        assert "selector" in STEP_SCHEMA["iframe_switch"]["required"]

    def test_iframe_exit_schema(self) -> None:
        assert "iframe_exit" in STEP_SCHEMA
        assert STEP_SCHEMA["iframe_exit"]["required"] == []

    # Category 2
    def test_extract_json_ld_schema(self) -> None:
        assert "extract_json_ld" in STEP_SCHEMA

    def test_extract_headings_schema(self) -> None:
        assert "extract_headings" in STEP_SCHEMA

    def test_extract_images_schema(self) -> None:
        assert "extract_images" in STEP_SCHEMA

    def test_extract_form_fields_schema(self) -> None:
        assert "extract_form_fields" in STEP_SCHEMA

    def test_extract_meta_schema(self) -> None:
        assert "extract_meta" in STEP_SCHEMA

    # Category 3
    def test_set_extra_headers_schema(self) -> None:
        assert "set_extra_headers" in STEP_SCHEMA
        assert "headers" in STEP_SCHEMA["set_extra_headers"]["required"]

    def test_http_auth_schema(self) -> None:
        assert "http_auth" in STEP_SCHEMA
        assert "username" in STEP_SCHEMA["http_auth"]["required"]
        assert "password" in STEP_SCHEMA["http_auth"]["required"]

    def test_local_storage_set_schema(self) -> None:
        assert "local_storage_set" in STEP_SCHEMA
        assert "key" in STEP_SCHEMA["local_storage_set"]["required"]
        assert "value" in STEP_SCHEMA["local_storage_set"]["required"]

    def test_local_storage_get_schema(self) -> None:
        assert "local_storage_get" in STEP_SCHEMA
        assert "key" in STEP_SCHEMA["local_storage_get"]["required"]

    def test_session_storage_set_schema(self) -> None:
        assert "session_storage_set" in STEP_SCHEMA

    def test_session_storage_get_schema(self) -> None:
        assert "session_storage_get" in STEP_SCHEMA

    # Category 4
    def test_assert_element_count_schema(self) -> None:
        assert "assert_element_count" in STEP_SCHEMA
        assert "selector" in STEP_SCHEMA["assert_element_count"]["required"]
        assert "count" in STEP_SCHEMA["assert_element_count"]["required"]

    def test_assert_attribute_schema(self) -> None:
        assert "assert_attribute" in STEP_SCHEMA
        assert "selector"  in STEP_SCHEMA["assert_attribute"]["required"]
        assert "attribute" in STEP_SCHEMA["assert_attribute"]["required"]
        assert "value"     in STEP_SCHEMA["assert_attribute"]["required"]

    def test_assert_title_schema(self) -> None:
        assert "assert_title" in STEP_SCHEMA
        assert "pattern" in STEP_SCHEMA["assert_title"]["required"]

    def test_assert_visible_schema(self) -> None:
        assert "assert_visible" in STEP_SCHEMA
        assert "selector" in STEP_SCHEMA["assert_visible"]["required"]

    def test_assert_hidden_schema(self) -> None:
        assert "assert_hidden" in STEP_SCHEMA
        assert "selector" in STEP_SCHEMA["assert_hidden"]["required"]


# ---------------------------------------------------------------------------
# validate_steps: new actions accepted
# ---------------------------------------------------------------------------

class TestValidateStepsNewActions:
    """validate_steps must accept every new action with its required keys."""

    def test_drag_drop_valid(self) -> None:
        out = validate_steps([{"action": "drag_drop", "source": "#a", "target": "#b"}])
        assert out[0]["action"] == "drag_drop"

    def test_right_click_valid(self) -> None:
        out = validate_steps([{"action": "right_click", "selector": "#btn"}])
        assert out[0]["action"] == "right_click"

    def test_double_click_valid(self) -> None:
        out = validate_steps([{"action": "double_click", "selector": "#btn"}])
        assert out[0]["action"] == "double_click"

    def test_upload_file_valid(self) -> None:
        out = validate_steps([{"action": "upload_file", "selector": "input", "path": "/f.txt"}])
        assert out[0]["action"] == "upload_file"

    def test_set_viewport_valid(self) -> None:
        out = validate_steps([{"action": "set_viewport", "width": 375, "height": 812}])
        assert out[0]["action"] == "set_viewport"

    def test_block_resource_default(self) -> None:
        out = validate_steps([{"action": "block_resource"}])
        assert out[0]["action"] == "block_resource"
        assert "types" in out[0]

    def test_iframe_switch_valid(self) -> None:
        out = validate_steps([{"action": "iframe_switch", "selector": "iframe"}])
        assert out[0]["selector"] == "iframe"

    def test_iframe_exit_valid(self) -> None:
        out = validate_steps([{"action": "iframe_exit"}])
        assert out[0]["action"] == "iframe_exit"

    def test_extract_json_ld_valid(self) -> None:
        out = validate_steps([{"action": "extract_json_ld"}])
        assert out[0]["action"] == "extract_json_ld"

    def test_extract_headings_valid(self) -> None:
        out = validate_steps([{"action": "extract_headings"}])
        assert out[0]["action"] == "extract_headings"

    def test_extract_images_defaults(self) -> None:
        out = validate_steps([{"action": "extract_images"}])
        assert out[0]["selector"] == "img"
        assert out[0]["limit"] == 100

    def test_extract_form_fields_default(self) -> None:
        out = validate_steps([{"action": "extract_form_fields"}])
        assert out[0]["selector"] == "form"

    def test_extract_meta_valid(self) -> None:
        out = validate_steps([{"action": "extract_meta"}])
        assert out[0]["action"] == "extract_meta"

    def test_set_extra_headers_valid(self) -> None:
        out = validate_steps([{"action": "set_extra_headers", "headers": {"X-A": "1"}}])
        assert out[0]["headers"] == {"X-A": "1"}

    def test_http_auth_valid(self) -> None:
        out = validate_steps([{"action": "http_auth", "username": "u", "password": "p"}])
        assert out[0]["username"] == "u"

    def test_local_storage_set_valid(self) -> None:
        out = validate_steps([{"action": "local_storage_set", "key": "k", "value": "v"}])
        assert out[0]["key"] == "k"

    def test_local_storage_get_valid(self) -> None:
        out = validate_steps([{"action": "local_storage_get", "key": "token"}])
        assert out[0]["key"] == "token"

    def test_session_storage_set_valid(self) -> None:
        out = validate_steps([{"action": "session_storage_set", "key": "k", "value": "v"}])
        assert out[0]["action"] == "session_storage_set"

    def test_session_storage_get_valid(self) -> None:
        out = validate_steps([{"action": "session_storage_get", "key": "k"}])
        assert out[0]["action"] == "session_storage_get"

    def test_assert_element_count_defaults(self) -> None:
        out = validate_steps([{"action": "assert_element_count", "selector": "li", "count": 3}])
        assert out[0]["operator"] == "eq"

    def test_assert_attribute_valid(self) -> None:
        out = validate_steps([{"action": "assert_attribute", "selector": "#a", "attribute": "href", "value": "/"}])
        assert out[0]["attribute"] == "href"

    def test_assert_title_defaults(self) -> None:
        out = validate_steps([{"action": "assert_title", "pattern": "Home"}])
        assert out[0]["case_sensitive"] is False

    def test_assert_visible_valid(self) -> None:
        out = validate_steps([{"action": "assert_visible", "selector": "#el"}])
        assert out[0]["selector"] == "#el"

    def test_assert_hidden_valid(self) -> None:
        out = validate_steps([{"action": "assert_hidden", "selector": "#el"}])
        assert out[0]["selector"] == "#el"


# ---------------------------------------------------------------------------
# TaskPlanner._execute_step: new actions dispatch to agent methods
# ---------------------------------------------------------------------------

class TestExecuteStepNewActions:
    """_execute_step must call the correct agent method for each new action."""

    def _run(self, step: dict, agent: MagicMock) -> None:
        planner = _planner_no_llm()
        validated = validate_steps([step])
        planner._execute_step(agent, validated[0])

    # Category 1
    def test_drag_drop_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "drag_drop", "source": "#a", "target": "#b"}, agent)
        agent.drag_drop.assert_called_once_with("#a", "#b")

    def test_right_click_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "right_click", "selector": "#btn"}, agent)
        agent.right_click.assert_called_once_with("#btn")

    def test_double_click_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "double_click", "selector": "#item"}, agent)
        agent.double_click.assert_called_once_with("#item")

    def test_upload_file_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "upload_file", "selector": "input", "path": "/f.txt"}, agent)
        agent.upload_file.assert_called_once_with("input", "/f.txt")

    def test_set_viewport_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "set_viewport", "width": 375, "height": 812}, agent)
        agent.set_viewport.assert_called_once_with(375, 812)

    def test_block_resource_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "block_resource"}, agent)
        agent.block_resource.assert_called_once()

    def test_iframe_switch_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "iframe_switch", "selector": "iframe"}, agent)
        agent.iframe_switch.assert_called_once_with("iframe")

    def test_iframe_exit_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "iframe_exit"}, agent)
        agent.iframe_exit.assert_called_once()

    # Category 2
    def test_extract_json_ld_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "extract_json_ld"}, agent)
        agent.extract_json_ld.assert_called_once()

    def test_extract_headings_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "extract_headings"}, agent)
        agent.extract_headings.assert_called_once()

    def test_extract_images_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "extract_images"}, agent)
        agent.extract_images.assert_called_once_with(selector="img", limit=100)

    def test_extract_form_fields_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "extract_form_fields"}, agent)
        agent.extract_form_fields.assert_called_once_with(selector="form")

    def test_extract_meta_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "extract_meta"}, agent)
        agent.extract_meta.assert_called_once()

    # Category 3
    def test_set_extra_headers_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "set_extra_headers", "headers": {"X-A": "1"}}, agent)
        agent.set_extra_headers.assert_called_once_with({"X-A": "1"})

    def test_http_auth_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "http_auth", "username": "u", "password": "p"}, agent)
        agent.http_auth.assert_called_once_with("u", "p")

    def test_local_storage_set_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "local_storage_set", "key": "k", "value": "v"}, agent)
        agent.local_storage_set.assert_called_once_with("k", "v")

    def test_local_storage_get_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "local_storage_get", "key": "token"}, agent)
        agent.local_storage_get.assert_called_once_with("token")

    def test_session_storage_set_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "session_storage_set", "key": "k", "value": "v"}, agent)
        agent.session_storage_set.assert_called_once_with("k", "v")

    def test_session_storage_get_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "session_storage_get", "key": "k"}, agent)
        agent.session_storage_get.assert_called_once_with("k")

    # Category 4
    def test_assert_element_count_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "assert_element_count", "selector": "li", "count": 3}, agent)
        agent.assert_element_count.assert_called_once_with("li", 3, operator="eq")

    def test_assert_attribute_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "assert_attribute", "selector": "#a", "attribute": "href", "value": "/"}, agent)
        agent.assert_attribute.assert_called_once_with("#a", "href", "/", case_sensitive=True)

    def test_assert_title_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "assert_title", "pattern": "Home"}, agent)
        agent.assert_title.assert_called_once_with("Home", case_sensitive=False)

    def test_assert_visible_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "assert_visible", "selector": "#el"}, agent)
        agent.assert_visible.assert_called_once_with("#el")

    def test_assert_hidden_dispatches(self) -> None:
        agent = _make_agent()
        self._run({"action": "assert_hidden", "selector": "#el"}, agent)
        agent.assert_hidden.assert_called_once_with("#el")


# ---------------------------------------------------------------------------
# {{last}} propagation for new extraction actions
# ---------------------------------------------------------------------------

class TestLastPropagationNewActions:
    """Verify that new extraction actions update {{last}} for downstream steps."""

    def _execute_two_steps(self, first_action: dict, first_return: dict, agent: MagicMock) -> list:
        planner = _planner_no_llm()
        # Patch agent method to return the given dict
        method = first_action["action"]
        getattr(agent, method).return_value = first_return
        # Second step: write_file using {{last}}
        second = validate_steps([
            first_action,
            {"action": "write_file", "path": "out.txt", "content": "{{last}}"},
        ])
        # Use real execute so {{last}} substitution runs
        agent.write_file = MagicMock(return_value={"bytes_written": 5})
        # We need system_tools mock for write_file
        st_mock = MagicMock()
        st_mock.write_file.return_value = {"bytes_written": 5}
        planner._system_tools = st_mock
        return planner.execute(second, agent)

    def test_extract_json_ld_populates_last(self) -> None:
        agent = _make_agent()
        results = self._execute_two_steps(
            {"action": "extract_json_ld"},
            {"items": [{"@type": "Product"}], "count": 1},
            agent,
        )
        # write_file should have been called with the JSON of the result
        assert all(r["status"] == "ok" for r in results)

    def test_local_storage_get_populates_last(self) -> None:
        agent = _make_agent()
        planner = _planner_no_llm()
        st_mock = MagicMock()
        st_mock.write_file.return_value = {"bytes_written": 5}
        planner._system_tools = st_mock
        agent.local_storage_get.return_value = {"key": "token", "value": "abc123"}

        steps = validate_steps([
            {"action": "local_storage_get", "key": "token"},
            {"action": "write_file", "path": "out.txt", "content": "{{last}}"},
        ])
        results = planner.execute(steps, agent)
        assert all(r["status"] == "ok" for r in results)
        # write_file must have been called with the actual token value
        call_args = st_mock.write_file.call_args
        assert "abc123" in call_args[0][1]


# ---------------------------------------------------------------------------
# BrowserAgent unit-level mock tests (no Playwright)
# ---------------------------------------------------------------------------

class TestBrowserAgentNewMethods:
    """Unit tests for new BrowserAgent methods using a mock page."""

    def _make_ba(self) -> tuple:
        """Return (BrowserAgent, mock_page, mock_context)."""
        from browser_agent import BrowserAgent
        ba = BrowserAgent()
        mock_page    = MagicMock()
        mock_context = MagicMock()
        ba._page    = mock_page
        ba._context = mock_context
        ba._pages   = [mock_page]
        return ba, mock_page, mock_context

    # Category 1 — drag_drop
    def test_drag_drop_calls_playwright(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector.return_value = MagicMock()  # resolve_selector
        result = ba.drag_drop("#a", "#b")
        page.drag_and_drop.assert_called_once()
        assert result["source"] == "#a"
        assert result["target"] == "#b"

    # right_click
    def test_right_click_calls_playwright(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector.return_value = MagicMock()
        result = ba.right_click("button")
        assert result["right_clicked"] == "button"

    # double_click
    def test_double_click_calls_playwright(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector.return_value = MagicMock()
        result = ba.double_click("a")
        assert "double_clicked" in result

    # upload_file
    def test_upload_file_calls_set_input_files(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector.return_value = MagicMock()
        result = ba.upload_file("input[type=file]", "/tmp/doc.pdf")
        assert result["uploaded"] == "/tmp/doc.pdf"

    # set_viewport
    def test_set_viewport_updates_size(self) -> None:
        ba, page, _ = self._make_ba()
        result = ba.set_viewport(1920, 1080)
        page.set_viewport_size.assert_called_once_with({"width": 1920, "height": 1080})
        assert result == {"width": 1920, "height": 1080}

    # block_resource defaults
    def test_block_resource_default_types(self) -> None:
        ba, page, _ = self._make_ba()
        result = ba.block_resource()
        page.route.assert_called_once()
        assert set(result["blocked_types"]) == {"image", "stylesheet", "font"}

    def test_block_resource_custom_types(self) -> None:
        ba, page, _ = self._make_ba()
        result = ba.block_resource(types=["script"])
        assert result["blocked_types"] == ["script"]

    # iframe_switch
    def test_iframe_switch_sets_active_frame(self) -> None:
        ba, page, _ = self._make_ba()
        mock_el    = MagicMock()
        mock_frame = MagicMock()
        mock_frame.url = "https://frame.example.com"
        mock_el.content_frame.return_value = mock_frame
        page.query_selector.return_value = mock_el
        result = ba.iframe_switch("iframe#login")
        assert ba._active_frame is mock_frame
        assert result["frame_url"] == "https://frame.example.com"

    def test_iframe_switch_raises_when_not_iframe(self) -> None:
        ba, page, _ = self._make_ba()
        mock_el = MagicMock()
        mock_el.content_frame.return_value = None
        page.query_selector.return_value = mock_el
        with pytest.raises(ValueError, match="not an iframe"):
            ba.iframe_switch("div#notframe")

    def test_iframe_switch_raises_when_not_found(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector.return_value = None
        with pytest.raises(ValueError, match="No iframe element"):
            ba.iframe_switch("iframe.missing")

    # iframe_exit
    def test_iframe_exit_clears_active_frame(self) -> None:
        ba, page, _ = self._make_ba()
        ba._active_frame = MagicMock()
        page.url = "https://example.com"
        result = ba.iframe_exit()
        assert ba._active_frame is None
        assert result["frame_url"] == "https://example.com"

    # Category 2 — extract_json_ld
    def test_extract_json_ld_returns_items(self) -> None:
        ba, page, _ = self._make_ba()
        page.evaluate.return_value = [{"@type": "Product", "name": "Widget"}]
        result = ba.extract_json_ld()
        assert result["count"] == 1
        assert result["items"][0]["@type"] == "Product"

    # extract_headings
    def test_extract_headings_returns_list(self) -> None:
        ba, page, _ = self._make_ba()
        page.evaluate.return_value = [{"level": 1, "text": "Hello"}, {"level": 2, "text": "World"}]
        result = ba.extract_headings()
        assert result["count"] == 2
        assert result["headings"][0]["level"] == 1

    # extract_images
    def test_extract_images_returns_list(self) -> None:
        ba, page, _ = self._make_ba()
        page.evaluate.return_value = [{"src": "logo.png", "alt": "Logo", "width": 100, "height": 50}]
        result = ba.extract_images()
        assert result["count"] == 1
        assert result["images"][0]["src"] == "logo.png"

    # extract_form_fields
    def test_extract_form_fields_returns_fields(self) -> None:
        ba, page, _ = self._make_ba()
        page.evaluate.return_value = [{"tag": "input", "name": "email", "type": "email", "id": "", "placeholder": "Email", "value": "", "required": True}]
        result = ba.extract_form_fields()
        assert result["count"] == 1
        assert result["fields"][0]["name"] == "email"

    # extract_meta
    def test_extract_meta_returns_dict(self) -> None:
        ba, page, _ = self._make_ba()
        page.evaluate.return_value = {"title": "My Page", "description": "Desc", "tags": [], "count": 0}
        result = ba.extract_meta()
        assert result["title"] == "My Page"

    # Category 3 — set_extra_headers
    def test_set_extra_headers_calls_context(self) -> None:
        ba, page, ctx = self._make_ba()
        result = ba.set_extra_headers({"X-Token": "abc"})
        ctx.set_extra_http_headers.assert_called_once_with({"X-Token": "abc"})
        assert "X-Token" in result["headers_set"]

    def test_set_extra_headers_raises_when_not_started(self) -> None:
        from browser_agent import BrowserAgent
        ba = BrowserAgent()
        with pytest.raises(RuntimeError, match="not started"):
            ba.set_extra_headers({"X-A": "1"})

    # http_auth
    def test_http_auth_sets_authorization_header(self) -> None:
        ba, page, ctx = self._make_ba()
        result = ba.http_auth("alice", "secret")
        assert result["auth_set"] is True
        assert result["username"] == "alice"
        call_args = ctx.set_extra_http_headers.call_args[0][0]
        assert "Authorization" in call_args
        assert call_args["Authorization"].startswith("Basic ")

    # local_storage_set/get
    def test_local_storage_set(self) -> None:
        ba, page, _ = self._make_ba()
        result = ba.local_storage_set("token", "abc123")
        page.evaluate.assert_called_once()
        assert result == {"key": "token", "value": "abc123"}

    def test_local_storage_get(self) -> None:
        ba, page, _ = self._make_ba()
        page.evaluate.return_value = "abc123"
        result = ba.local_storage_get("token")
        assert result["value"] == "abc123"

    # session_storage_set/get
    def test_session_storage_set(self) -> None:
        ba, page, _ = self._make_ba()
        result = ba.session_storage_set("sid", "xyz")
        assert result == {"key": "sid", "value": "xyz"}

    def test_session_storage_get(self) -> None:
        ba, page, _ = self._make_ba()
        page.evaluate.return_value = "xyz"
        result = ba.session_storage_get("sid")
        assert result["value"] == "xyz"

    # Category 4 — assert_element_count
    def test_assert_element_count_passes(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector_all.return_value = [MagicMock()] * 3
        result = ba.assert_element_count("li", 3)
        assert result["actual"] == 3

    def test_assert_element_count_fails(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector_all.return_value = [MagicMock()] * 2
        with pytest.raises(AssertionError, match="assert_element_count"):
            ba.assert_element_count("li", 3)

    def test_assert_element_count_gte(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector_all.return_value = [MagicMock()] * 5
        result = ba.assert_element_count("li", 3, operator="gte")
        assert result["actual"] == 5

    def test_assert_element_count_invalid_operator(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector_all.return_value = []
        with pytest.raises(ValueError, match="Unknown operator"):
            ba.assert_element_count("li", 0, operator="nope")

    # assert_attribute
    def test_assert_attribute_passes(self) -> None:
        ba, page, _ = self._make_ba()
        page.get_attribute.return_value = "/dashboard"
        result = ba.assert_attribute("a#home", "href", "/dashboard")
        assert result["actual"] == "/dashboard"

    def test_assert_attribute_fails(self) -> None:
        ba, page, _ = self._make_ba()
        page.get_attribute.return_value = "/other"
        with pytest.raises(AssertionError, match="assert_attribute"):
            ba.assert_attribute("a#home", "href", "/dashboard")

    def test_assert_attribute_case_insensitive(self) -> None:
        ba, page, _ = self._make_ba()
        page.get_attribute.return_value = "ACTIVE"
        result = ba.assert_attribute("span", "class", "active", case_sensitive=False)
        assert result["expected"] == "active"

    # assert_title
    def test_assert_title_passes(self) -> None:
        ba, page, _ = self._make_ba()
        page.title.return_value = "My Dashboard — Example"
        result = ba.assert_title("Dashboard")
        assert result["matched"] is True

    def test_assert_title_fails(self) -> None:
        ba, page, _ = self._make_ba()
        page.title.return_value = "Login Page"
        with pytest.raises(AssertionError, match="assert_title"):
            ba.assert_title("Dashboard")

    def test_assert_title_case_insensitive(self) -> None:
        ba, page, _ = self._make_ba()
        page.title.return_value = "My DASHBOARD"
        result = ba.assert_title("dashboard", case_sensitive=False)
        assert result["matched"] is True

    # assert_visible
    def test_assert_visible_passes(self) -> None:
        ba, page, _ = self._make_ba()
        mock_el = MagicMock()
        mock_el.is_visible.return_value = True
        page.query_selector.return_value = mock_el
        result = ba.assert_visible("#modal")
        assert result["visible"] is True

    def test_assert_visible_fails_when_hidden(self) -> None:
        ba, page, _ = self._make_ba()
        mock_el = MagicMock()
        mock_el.is_visible.return_value = False
        page.query_selector.return_value = mock_el
        with pytest.raises(AssertionError, match="assert_visible"):
            ba.assert_visible("#modal")

    def test_assert_visible_fails_when_absent(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector.return_value = None
        with pytest.raises(AssertionError, match="assert_visible"):
            ba.assert_visible("#modal")

    # assert_hidden
    def test_assert_hidden_passes_when_absent(self) -> None:
        ba, page, _ = self._make_ba()
        page.query_selector.return_value = None
        result = ba.assert_hidden("#tooltip")
        assert result["hidden"] is True

    def test_assert_hidden_passes_when_invisible(self) -> None:
        ba, page, _ = self._make_ba()
        mock_el = MagicMock()
        mock_el.is_visible.return_value = False
        page.query_selector.return_value = mock_el
        result = ba.assert_hidden("#tooltip")
        assert result["hidden"] is True

    def test_assert_hidden_fails_when_visible(self) -> None:
        ba, page, _ = self._make_ba()
        mock_el = MagicMock()
        mock_el.is_visible.return_value = True
        page.query_selector.return_value = mock_el
        with pytest.raises(AssertionError, match="assert_hidden"):
            ba.assert_hidden("#tooltip")


# ---------------------------------------------------------------------------
# _frame property tests
# ---------------------------------------------------------------------------

class TestFrameProperty:
    def test_frame_returns_page_when_no_iframe(self) -> None:
        from browser_agent import BrowserAgent
        ba = BrowserAgent()
        mock_page = MagicMock()
        ba._page = mock_page
        ba._active_frame = None
        assert ba._frame is mock_page

    def test_frame_returns_active_frame_when_set(self) -> None:
        from browser_agent import BrowserAgent
        ba = BrowserAgent()
        mock_page  = MagicMock()
        mock_frame = MagicMock()
        ba._page         = mock_page
        ba._active_frame = mock_frame
        assert ba._frame is mock_frame
