"""Phase 2 reliability tests.

Covers:
- ``verified`` field semantics in ``TaskPlanner.run()``
- Template builders gain ``wait_selector`` + ``assert_text``/``assert_url`` steps
- Extraction readiness: ``wait_for_ready`` guard in ``_execute_step``
- Backward compatibility: ``success`` field unchanged
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_planner import (
    _KNOWN_SELECTORS,
    TaskPlanner,
    _bing_search_steps,
    _ddg_search_steps,
    _google_search_steps,
    _navigate_steps,
    _wikipedia_search_steps,
    _youtube_search_steps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_planner() -> TaskPlanner:
    return TaskPlanner()


def _make_agent(**overrides) -> MagicMock:
    """Return a minimal mock BrowserAgent.

    ``unsafe=True`` is required so that attributes named ``assert_text`` and
    ``assert_url`` can be set and called without MagicMock raising
    AttributeError for names starting with "assert".
    """
    agent = MagicMock(unsafe=True)
    agent.navigate.return_value = {"url": "https://example.com", "status": 200}
    agent.close_popups.return_value = {"dismissed": []}
    agent.wait_for_selector.return_value = {"selector": "body", "found": True}
    agent.wait_for_load_state.return_value = {"state": "networkidle", "reached": True}
    agent.assert_text.return_value = {"found": True, "text": "hello"}
    agent.assert_url.return_value = {"matched": True, "pattern": "example.com"}
    agent.fill.return_value = {"filled": True}
    agent.type_text.return_value = {"typed": True}
    agent.press.return_value = {"key": "Enter"}
    agent.extract_links.return_value = {"links": [], "count": 0}
    agent.extract_table.return_value = {"rows": [], "count": 0, "headers": []}
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


# ---------------------------------------------------------------------------
# 1. verified field semantics
# ---------------------------------------------------------------------------


class TestVerifiedField:
    def test_verified_true_when_assert_text_passes(self):
        planner = _make_planner()
        agent = _make_agent()
        # assert_text.return_value already set to {"found": True} by _make_agent

        steps = [
            {"action": "assert_text", "text": "hello"},
        ]
        results = planner.execute(steps, agent)

        # Simulate run() logic directly
        assertion_results = [
            r for r in results
            if r.get("action") in ("assert_text", "assert_url") and r.get("status") == "ok"
        ]
        last_r = assertion_results[-1].get("result") or {}
        verified = bool(last_r.get("found") or last_r.get("matched"))
        assert verified is True

    def test_verified_false_when_assert_text_result_not_found(self):
        planner = _make_planner()
        agent = _make_agent()
        # assert_text returns found=False (no exception, just not found)
        agent.assert_text.return_value = {"found": False, "text": "missing"}

        steps = [{"action": "assert_text", "text": "missing"}]
        results = planner.execute(steps, agent)

        assertion_results = [
            r for r in results
            if r.get("action") in ("assert_text", "assert_url") and r.get("status") == "ok"
        ]
        last_r = assertion_results[-1].get("result") or {} if assertion_results else {}
        verified = bool(last_r.get("found") or last_r.get("matched")) if assertion_results else None
        assert verified is False

    def test_verified_none_when_no_assertion_steps(self):
        """Backward compat canary: callers without assertions see verified=None."""
        planner = _make_planner()
        agent = _make_agent()

        with patch.object(planner, "plan", return_value=[
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
        ]):
            summary = planner.run("go to example.com", agent)

        assert summary["verified"] is None

    def test_success_still_true_when_verified_none(self):
        """No assertion steps, no errors → success=True, verified=None."""
        planner = _make_planner()
        agent = _make_agent()

        with patch.object(planner, "plan", return_value=[
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
        ]):
            summary = planner.run("go to example.com", agent)

        assert summary["success"] is True
        assert summary["verified"] is None

    def test_verified_true_when_assert_url_passes(self):
        planner = _make_planner()
        agent = _make_agent()
        # assert_url.return_value already set to {"matched": True} by _make_agent

        steps = [{"action": "assert_url", "pattern": "example.com"}]
        results = planner.execute(steps, agent)

        assertion_results = [
            r for r in results
            if r.get("action") in ("assert_text", "assert_url") and r.get("status") == "ok"
        ]
        last_r = assertion_results[-1].get("result") or {}
        verified = bool(last_r.get("found") or last_r.get("matched"))
        assert verified is True

    def test_verified_field_present_in_run_output(self):
        """verified key is always present in run() output."""
        planner = _make_planner()
        agent = _make_agent()

        with patch.object(planner, "plan", return_value=[
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
        ]):
            summary = planner.run("go to example.com", agent)

        assert "verified" in summary

    def test_success_field_still_present_in_run_output(self):
        """success key is still always present — backward compat."""
        planner = _make_planner()
        agent = _make_agent()

        with patch.object(planner, "plan", return_value=[
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
        ]):
            summary = planner.run("go to example.com", agent)

        assert "success" in summary

    def test_verified_uses_last_assertion_result(self):
        """When two assert steps run, verified reflects the last one."""
        planner = _make_planner()
        agent = MagicMock(unsafe=True)
        # First assert: found=False, second: found=True
        agent.assert_text.side_effect = [
            {"found": False, "text": "first"},
            {"found": True,  "text": "second"},
        ]
        agent.navigate.return_value = {}

        steps = [
            {"action": "assert_text", "text": "first"},
            {"action": "assert_text", "text": "second"},
        ]
        results = planner.execute(steps, agent, stop_on_error=False)

        assertion_results = [
            r for r in results
            if r.get("action") in ("assert_text", "assert_url") and r.get("status") == "ok"
        ]
        last_r = assertion_results[-1].get("result") or {} if assertion_results else {}
        verified = bool(last_r.get("found") or last_r.get("matched")) if assertion_results else None
        assert verified is True

    def test_run_verified_field_integration(self):
        """Full run() returns verified=True when last assertion passes."""
        planner = _make_planner()
        agent = _make_agent()
        # assert_text.return_value already {"found": True} from _make_agent

        with patch.object(planner, "plan", return_value=[
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
            {"action": "assert_text", "text": "ok"},
        ]):
            summary = planner.run("check ok on example", agent)

        assert summary["verified"] is True
        assert summary["success"] is True


# ---------------------------------------------------------------------------
# 2. Template verification coverage
# ---------------------------------------------------------------------------


class TestTemplateVerification:
    def test_google_search_steps_ends_with_assert_text(self):
        steps = _google_search_steps("python")
        assert steps[-1]["action"] == "assert_text"

    def test_google_search_steps_assert_uses_results_selector(self):
        steps = _google_search_steps("python")
        expected_sel = _KNOWN_SELECTORS["google.com"]["results"]
        assert steps[-1].get("selector") == expected_sel

    def test_google_search_steps_assert_text_is_query(self):
        steps = _google_search_steps("machine learning")
        assert steps[-1]["text"] == "machine learning"

    def test_google_search_steps_has_wait_selector_before_assert(self):
        steps = _google_search_steps("python")
        actions = [s["action"] for s in steps]
        ai = actions.index("assert_text") if "assert_text" in actions else -1
        # The last wait_selector (for results) should come before the assert_text
        last_wait = max(i for i, a in enumerate(actions) if a == "wait_selector")
        assert last_wait < ai

    def test_bing_search_steps_ends_with_assert_text(self):
        steps = _bing_search_steps("python")
        assert steps[-1]["action"] == "assert_text"

    def test_bing_search_steps_assert_text_is_query(self):
        steps = _bing_search_steps("deep learning")
        assert steps[-1]["text"] == "deep learning"

    def test_youtube_search_steps_ends_with_assert_text(self):
        steps = _youtube_search_steps("tutorials")
        assert steps[-1]["action"] == "assert_text"

    def test_youtube_search_steps_assert_uses_results_selector(self):
        steps = _youtube_search_steps("tutorials")
        expected_sel = _KNOWN_SELECTORS["youtube.com"]["results"]
        assert steps[-1].get("selector") == expected_sel

    def test_ddg_search_steps_ends_with_assert_text(self):
        steps = _ddg_search_steps("privacy")
        assert steps[-1]["action"] == "assert_text"

    def test_ddg_search_steps_assert_text_is_query(self):
        steps = _ddg_search_steps("open source")
        assert steps[-1]["text"] == "open source"

    def test_wikipedia_search_steps_ends_with_assert_text(self):
        steps = _wikipedia_search_steps("recursion")
        assert steps[-1]["action"] == "assert_text"

    def test_wikipedia_search_steps_assert_uses_results_selector(self):
        steps = _wikipedia_search_steps("recursion")
        expected_sel = _KNOWN_SELECTORS["wikipedia.org"]["results"]
        assert steps[-1].get("selector") == expected_sel

    def test_navigate_steps_ends_with_assert_url(self):
        steps = _navigate_steps("https://example.com")
        assert steps[-1]["action"] == "assert_url"

    def test_navigate_steps_assert_url_pattern_is_domain(self):
        steps = _navigate_steps("https://news.ycombinator.com/newest")
        assert steps[-1]["pattern"] == "news.ycombinator.com"

    def test_navigate_steps_prepends_https_when_missing(self):
        steps = _navigate_steps("example.com")
        assert steps[0]["url"] == "https://example.com"
        assert steps[-1]["pattern"] == "example.com"

    def test_navigate_steps_assert_url_pattern_strips_path(self):
        steps = _navigate_steps("https://docs.python.org/3/library/os.html")
        assert steps[-1]["pattern"] == "docs.python.org"


# ---------------------------------------------------------------------------
# 3. Extraction readiness
# ---------------------------------------------------------------------------


class TestExtractionReadiness:
    def _run_extract_step(self, step: dict, agent: MagicMock) -> None:
        planner = _make_planner()
        planner.execute([step], agent)

    def test_extract_links_with_custom_selector_calls_wait_for_selector(self):
        agent = _make_agent()
        self._run_extract_step(
            {"action": "extract_links", "selector": "#search-results"},
            agent,
        )
        agent.wait_for_selector.assert_any_call("#search-results", timeout=5_000)

    def test_extract_links_with_default_selector_skips_wait(self):
        agent = _make_agent()
        self._run_extract_step(
            {"action": "extract_links"},
            agent,
        )
        # wait_for_selector should NOT have been called for the default selector "a"
        for c in agent.wait_for_selector.call_args_list:
            assert c.args[0] != "a"

    def test_extract_links_explicit_default_selector_skips_wait(self):
        agent = _make_agent()
        self._run_extract_step(
            {"action": "extract_links", "selector": "a"},
            agent,
        )
        for c in agent.wait_for_selector.call_args_list:
            assert c.args[0] != "a"

    def test_extract_table_with_custom_selector_calls_wait_for_selector(self):
        agent = _make_agent()
        self._run_extract_step(
            {"action": "extract_table", "selector": ".data-table"},
            agent,
        )
        agent.wait_for_selector.assert_any_call(".data-table", timeout=5_000)

    def test_extract_table_with_default_selector_skips_wait(self):
        agent = _make_agent()
        self._run_extract_step(
            {"action": "extract_table"},
            agent,
        )
        for c in agent.wait_for_selector.call_args_list:
            assert c.args[0] != "table"

    def test_extract_table_wait_for_ready_false_skips_wait(self):
        agent = _make_agent()
        self._run_extract_step(
            {"action": "extract_table", "selector": ".data-table", "wait_for_ready": False},
            agent,
        )
        for c in agent.wait_for_selector.call_args_list:
            assert c.args[0] != ".data-table"

    def test_extract_links_wait_for_ready_false_skips_wait(self):
        agent = _make_agent()
        self._run_extract_step(
            {"action": "extract_links", "selector": "#links", "wait_for_ready": False},
            agent,
        )
        for c in agent.wait_for_selector.call_args_list:
            assert c.args[0] != "#links"

    def test_extract_links_proceeds_even_if_wait_selector_raises(self):
        agent = _make_agent()
        agent.wait_for_selector.side_effect = Exception("timeout")
        agent.extract_links.return_value = {"links": [{"text": "hi", "href": "/"}], "count": 1}

        planner = _make_planner()
        results = planner.execute(
            [{"action": "extract_links", "selector": "#results"}],
            agent,
        )
        assert results[0]["status"] == "ok"
        assert results[0]["result"]["count"] == 1

    def test_extract_table_proceeds_even_if_wait_selector_raises(self):
        agent = _make_agent()
        agent.wait_for_selector.side_effect = Exception("timeout")
        agent.extract_table.return_value = {"rows": [{"col": "val"}], "count": 1, "headers": ["col"]}
        planner = _make_planner()
        results = planner.execute(
            [{"action": "extract_table", "selector": ".tbl"}],
            agent,
        )
        assert results[0]["status"] == "ok"

    def test_extract_links_respects_custom_timeout(self):
        agent = _make_agent()
        self._run_extract_step(
            {"action": "extract_links", "selector": "#results", "timeout": 10_000},
            agent,
        )
        agent.wait_for_selector.assert_any_call("#results", timeout=10_000)
