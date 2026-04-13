"""
Phase 1 reliability tests.

Covers:
- fill → type_text routing (default) and raw-fill opt-out
- login-wall detection on navigate
- CMP selector presence in _POPUP_SELECTORS
- second-pass popup dismissal in close_popups
- prompt rule correctness (anti-verify removed, mandatory-verify added)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from browser_agent import _POPUP_SELECTORS, BrowserAgent, _run_popup_scan
from task_planner import (
    _SYSTEM_PROMPT,
    _VISION_SYSTEM_PROMPT,
    STEP_SCHEMA,
    TaskPlanner,
    validate_steps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_mock() -> MagicMock:
    agent = MagicMock()
    agent.navigate.return_value = {"url": "https://example.com", "title": "Test"}
    agent.fill.return_value = {"filled": "value", "selector": "input"}
    agent.type_text.return_value = {"typed": "text", "selector": "input"}
    agent.page = MagicMock()
    agent.page.url = "https://example.com"
    return agent


def _make_planner() -> TaskPlanner:
    with patch.object(TaskPlanner, "_detect_llm", return_value=None):
        return TaskPlanner()


def _make_browser_agent(**kwargs) -> tuple[BrowserAgent, MagicMock]:
    """Return (agent, mock_page) with a fully wired MagicMock page."""
    agent = BrowserAgent(**kwargs)
    page = MagicMock()
    page.url = "https://example.com"
    page.query_selector.return_value = None
    page.evaluate.return_value = False
    page.keyboard = MagicMock()
    agent._page = page
    agent._context = MagicMock()
    agent._pages = [page]
    return agent, page


# ---------------------------------------------------------------------------
# STEP_SCHEMA changes
# ---------------------------------------------------------------------------

class TestStepSchemaFillOptional:
    def test_framework_safe_key_present_in_schema(self):
        assert "framework_safe" in STEP_SCHEMA["fill"]["optional"]

    def test_framework_safe_default_is_true(self):
        assert STEP_SCHEMA["fill"]["optional"]["framework_safe"] is True

    def test_fill_schema_accepts_framework_safe_false(self):
        """validate_steps must not reject framework_safe: False."""
        steps = validate_steps([{"action": "fill", "selector": "#q", "value": "v", "framework_safe": False}])
        assert steps[0]["framework_safe"] is False


# ---------------------------------------------------------------------------
# fill → type_text routing
# ---------------------------------------------------------------------------

class TestFillRoutingViaExecuteStep:
    def test_fill_routes_to_type_text_by_default(self, tmp_path):
        planner = _make_planner()
        agent = _make_agent_mock()
        steps = validate_steps([{"action": "fill", "selector": "#q", "value": "hello"}])
        planner.execute(steps, agent)
        agent.type_text.assert_called_once_with("#q", "hello", clear_first=True)
        agent.fill.assert_not_called()

    def test_fill_uses_raw_fill_when_framework_safe_false(self, tmp_path):
        planner = _make_planner()
        agent = _make_agent_mock()
        steps = validate_steps([{"action": "fill", "selector": "#q", "value": "hello", "framework_safe": False}])
        planner.execute(steps, agent)
        agent.fill.assert_called_once_with("#q", "hello")
        agent.type_text.assert_not_called()

    def test_fill_framework_safe_true_explicit_routes_to_type_text(self, tmp_path):
        planner = _make_planner()
        agent = _make_agent_mock()
        steps = validate_steps([{"action": "fill", "selector": "#q", "value": "hi", "framework_safe": True}])
        planner.execute(steps, agent)
        agent.type_text.assert_called_once_with("#q", "hi", clear_first=True)
        agent.fill.assert_not_called()


# ---------------------------------------------------------------------------
# Login-wall detection
# ---------------------------------------------------------------------------

class TestLoginWallDetection:
    def test_login_wall_raises_when_auth_redirect_detected(self):
        planner = _make_planner()
        agent = _make_agent_mock()
        # Navigating to a non-auth URL but landing on a login page
        agent.page.url = "https://example.com/login?next=/dashboard"
        steps = validate_steps([{"action": "navigate", "url": "https://example.com/dashboard"}])
        with pytest.raises(ValueError, match="Login wall detected"):
            planner._execute_step(agent, steps[0])

    def test_login_wall_no_false_positive_when_target_is_auth(self):
        planner = _make_planner()
        agent = _make_agent_mock()
        # The target itself is a login page — no false positive
        agent.page.url = "https://example.com/login"
        steps = validate_steps([{"action": "navigate", "url": "https://example.com/login"}])
        # Should not raise
        planner._execute_step(agent, steps[0])

    def test_login_wall_no_false_positive_on_normal_redirect(self):
        planner = _make_planner()
        agent = _make_agent_mock()
        # Normal redirect: target is /dashboard, actual is /home — no auth token
        agent.page.url = "https://example.com/home"
        steps = validate_steps([{"action": "navigate", "url": "https://example.com/dashboard"}])
        # Should not raise
        planner._execute_step(agent, steps[0])

    def test_login_wall_stops_subsequent_steps_via_execute(self):
        planner = _make_planner()
        agent = _make_agent_mock()
        agent.page.url = "https://example.com/login"

        steps = validate_steps([
            {"action": "navigate", "url": "https://example.com/dashboard"},
            {"action": "fill", "selector": "#user", "value": "test"},
        ])
        results = planner.execute(steps, agent, stop_on_error=True)
        # First step should have error status
        assert results[0]["status"] == "error"
        assert "Login wall" in results[0]["error"]
        # Second step should not have been executed
        agent.fill.assert_not_called()
        agent.type_text.assert_not_called()


# ---------------------------------------------------------------------------
# CMP selectors
# ---------------------------------------------------------------------------

class TestCmpPopupSelectors:
    def test_onetrust_selector_present(self):
        assert "#onetrust-accept-btn-handler" in _POPUP_SELECTORS

    def test_cookiebot_selector_present(self):
        assert "#CybotCookiebotDialogBodyButtonAccept" in _POPUP_SELECTORS

    def test_trustarc_selector_present(self):
        assert "#truste-consent-button" in _POPUP_SELECTORS

    def test_osano_selector_present(self):
        assert ".osano-cm-accept--all" in _POPUP_SELECTORS

    def test_quantcast_selector_present(self):
        assert "[data-testid='qc-cmp2-accept-all-button']" in _POPUP_SELECTORS

    def test_cc_allow_selector_present(self):
        assert ".cc-allow" in _POPUP_SELECTORS


# ---------------------------------------------------------------------------
# _run_popup_scan helper
# ---------------------------------------------------------------------------

class TestRunPopupScan:
    def test_returns_empty_list_when_no_elements(self):
        page = MagicMock()
        page.query_selector.return_value = None
        result = _run_popup_scan(page)
        assert result == []

    def test_clicks_visible_element_and_returns_selector(self):
        page = MagicMock()
        el = MagicMock()
        el.is_visible.return_value = True
        # First selector matches; rest return None
        page.query_selector.side_effect = [el] + [None] * 200
        result = _run_popup_scan(page)
        el.click.assert_called_once_with(timeout=2_000)
        assert len(result) >= 1

    def test_skips_invisible_element(self):
        page = MagicMock()
        el = MagicMock()
        el.is_visible.return_value = False
        page.query_selector.return_value = el
        result = _run_popup_scan(page)
        el.click.assert_not_called()
        assert result == []

    def test_swallows_exceptions(self):
        page = MagicMock()
        page.query_selector.side_effect = Exception("playwright error")
        result = _run_popup_scan(page)
        assert result == []


# ---------------------------------------------------------------------------
# close_popups second-pass refactor
# ---------------------------------------------------------------------------

class TestClosePopupsSecondPass:
    def test_wait_for_timeout_called_with_800(self):
        agent, page = _make_browser_agent()
        page.evaluate.return_value = False
        agent.close_popups()
        page.wait_for_timeout.assert_called_once_with(800)

    def test_second_pass_catches_late_banner(self):
        """Element absent in first pass but present in second → still dismissed."""
        agent, page = _make_browser_agent()
        page.evaluate.return_value = False

        el = MagicMock()
        el.is_visible.return_value = True

        call_count = [0]

        def query_side_effect(selector):
            call_count[0] += 1
            # Return None for all selectors during first pass (1..N), then el
            # on the very first selector of the second pass.
            # We don't know exact N (selector list length), so we approximate:
            # first pass calls len(_POPUP_SELECTORS) times, second pass starts after.
            n = len(_POPUP_SELECTORS)
            if call_count[0] <= n:
                return None
            if call_count[0] == n + 1:
                return el
            return None

        page.query_selector.side_effect = query_side_effect

        result = agent.close_popups()
        el.click.assert_called_once_with(timeout=2_000)
        assert result["count"] >= 1

    def test_deduplication_of_selector_dismissed_in_both_passes(self):
        """Selector matched in both passes must appear only once in dismissed."""
        agent, page = _make_browser_agent()
        page.evaluate.return_value = False

        el = MagicMock()
        el.is_visible.return_value = True

        # The first selector always matches in both passes
        def query_side_effect(selector):
            if selector == _POPUP_SELECTORS[0]:
                return el
            return None

        page.query_selector.side_effect = query_side_effect

        result = agent.close_popups()
        dismissed = result["dismissed"]
        assert dismissed.count(_POPUP_SELECTORS[0]) == 1

    def test_timeout_exception_does_not_abort_second_pass(self):
        """wait_for_timeout failure must not prevent second pass from running."""
        agent, page = _make_browser_agent()
        page.wait_for_timeout.side_effect = Exception("timeout error")
        page.evaluate.return_value = False

        el = MagicMock()
        el.is_visible.return_value = True

        n = len(_POPUP_SELECTORS)
        call_count = [0]

        def query_side_effect(selector):
            call_count[0] += 1
            if call_count[0] == n + 1:
                return el
            return None

        page.query_selector.side_effect = query_side_effect

        # Should not raise
        result = agent.close_popups()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Prompt rule assertions
# ---------------------------------------------------------------------------

class TestPromptRules:
    def test_anti_verification_rule_removed_from_system_prompt(self):
        assert "Never add steps to" not in _SYSTEM_PROMPT

    def test_anti_verification_rule_removed_from_vision_prompt(self):
        assert "Never add steps to" not in _VISION_SYSTEM_PROMPT

    def test_mandatory_verification_in_system_prompt(self):
        assert "MANDATORY VERIFICATION" in _SYSTEM_PROMPT

    def test_mandatory_verification_in_vision_prompt(self):
        assert "MANDATORY VERIFICATION" in _VISION_SYSTEM_PROMPT

    def test_fill_type_guidance_in_system_prompt(self):
        assert "FORM INPUT" in _SYSTEM_PROMPT

    def test_fill_type_guidance_in_vision_prompt(self):
        assert "FORM INPUT" in _VISION_SYSTEM_PROMPT

    def test_few_shot_google_example_uses_type_not_fill(self):
        # The Google few-shot example should use "type" action, not "fill"
        google_block_start = _SYSTEM_PROMPT.find('Task: "go to google')
        assert google_block_start != -1
        google_block = _SYSTEM_PROMPT[google_block_start:google_block_start + 600]
        assert '"action": "fill"' not in google_block

    def test_few_shot_youtube_example_uses_type_not_fill(self):
        youtube_block_start = _SYSTEM_PROMPT.find('Task: "search for machine learning on YouTube"')
        assert youtube_block_start != -1
        youtube_block = _SYSTEM_PROMPT[youtube_block_start:youtube_block_start + 600]
        assert '"action": "fill"' not in youtube_block
