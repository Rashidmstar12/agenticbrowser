"""
Unit tests for task_planner: validate_steps, _interpolate_last, template
matching, and TaskPlanner.plan (template path only — no LLM required).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_planner import (
    STEP_SCHEMA,
    StepValidationError,
    TaskPlanner,
    _interpolate_last,
    validate_steps,
)

# ---------------------------------------------------------------------------
# validate_steps — happy-path
# ---------------------------------------------------------------------------

class TestValidateStepsHappy:
    def test_minimal_navigate(self) -> None:
        steps = [{"action": "navigate", "url": "https://example.com"}]
        out = validate_steps(steps)
        assert len(out) == 1
        assert out[0]["action"] == "navigate"
        assert out[0]["url"] == "https://example.com"

    def test_defaults_applied(self) -> None:
        steps = [{"action": "navigate", "url": "https://example.com"}]
        out = validate_steps(steps)
        # Optional key "wait_until" should be set to its default
        assert "wait_until" in out[0]
        assert out[0]["wait_until"] == "domcontentloaded"

    def test_optional_override_preserved(self) -> None:
        steps = [{"action": "navigate", "url": "https://example.com", "wait_until": "load"}]
        out = validate_steps(steps)
        assert out[0]["wait_until"] == "load"

    def test_universal_keys_present(self) -> None:
        steps = [{"action": "press", "key": "Enter"}]
        out = validate_steps(steps)
        assert "retry" in out[0]
        assert "retry_delay" in out[0]

    def test_retry_override(self) -> None:
        steps = [{"action": "press", "key": "Enter", "retry": 3}]
        out = validate_steps(steps)
        assert out[0]["retry"] == 3

    def test_multiple_steps_validated(self) -> None:
        steps = [
            {"action": "navigate", "url": "https://example.com"},
            {"action": "click", "selector": "button"},
        ]
        out = validate_steps(steps)
        assert len(out) == 2

    def test_close_popups_no_required(self) -> None:
        out = validate_steps([{"action": "close_popups"}])
        assert out[0]["action"] == "close_popups"

    def test_all_known_actions_accepted(self) -> None:
        """Every action in STEP_SCHEMA should be buildable with required keys."""
        for action, schema in STEP_SCHEMA.items():
            step: dict = {"action": action}
            for req in schema["required"]:
                step[req] = "dummy"
            out = validate_steps([step])
            assert out[0]["action"] == action


# ---------------------------------------------------------------------------
# validate_steps — error cases
# ---------------------------------------------------------------------------

class TestValidateStepsErrors:
    def test_empty_list_raises(self) -> None:
        with pytest.raises(StepValidationError, match="at least one step"):
            validate_steps([])

    def test_not_a_list_raises(self) -> None:
        with pytest.raises(StepValidationError, match="JSON array"):
            validate_steps({"action": "navigate"})  # type: ignore[arg-type]

    def test_step_not_a_dict_raises(self) -> None:
        with pytest.raises(StepValidationError, match="not an object"):
            validate_steps(["navigate"])  # type: ignore[list-item]

    def test_missing_action_key_raises(self) -> None:
        with pytest.raises(StepValidationError, match="missing the 'action'"):
            validate_steps([{"url": "https://example.com"}])

    def test_unknown_action_raises(self) -> None:
        with pytest.raises(StepValidationError, match="unknown action"):
            validate_steps([{"action": "fly_to_mars"}])

    def test_missing_required_key_raises(self) -> None:
        with pytest.raises(StepValidationError, match="missing required key"):
            validate_steps([{"action": "navigate"}])  # missing "url"

    def test_missing_required_key_click(self) -> None:
        with pytest.raises(StepValidationError, match="missing required key"):
            validate_steps([{"action": "click"}])  # missing "selector"

    def test_too_many_steps_raises(self) -> None:
        steps = [{"action": "press", "key": "Enter"}] * 21
        with pytest.raises(StepValidationError, match="maximum is 20"):
            validate_steps(steps)

    def test_exactly_20_steps_ok(self) -> None:
        steps = [{"action": "press", "key": "Enter"}] * 20
        out = validate_steps(steps)
        assert len(out) == 20


# ---------------------------------------------------------------------------
# _interpolate_last
# ---------------------------------------------------------------------------

class TestInterpolateLast:
    def test_replaces_in_string_value(self) -> None:
        step = {"action": "write_file", "path": "out.txt", "content": "{{last}}"}
        result = _interpolate_last(step, "hello world")
        assert result["content"] == "hello world"

    def test_leaves_non_string_unchanged(self) -> None:
        step = {"action": "scroll", "y": 500}
        result = _interpolate_last(step, "ignored")
        assert result["y"] == 500

    def test_multiple_occurrences_replaced(self) -> None:
        step = {"action": "fill", "selector": "{{last}}", "value": "{{last}}"}
        result = _interpolate_last(step, "X")
        assert result["selector"] == "X"
        assert result["value"] == "X"

    def test_no_placeholder_unchanged(self) -> None:
        step = {"action": "navigate", "url": "https://example.com"}
        result = _interpolate_last(step, "ignored")
        assert result["url"] == "https://example.com"

    def test_does_not_mutate_original(self) -> None:
        step = {"action": "write_file", "path": "f.txt", "content": "{{last}}"}
        _interpolate_last(step, "new")
        assert step["content"] == "{{last}}"


# ---------------------------------------------------------------------------
# TaskPlanner.plan — template matching (no LLM)
# ---------------------------------------------------------------------------

class TestTaskPlannerTemplates:
    """Tests that template intents produce the right step sequences without
    any LLM calls.  The planner is forced into template-only mode by patching
    _detect_llm to return None."""

    def _planner(self) -> TaskPlanner:
        with patch.object(TaskPlanner, "_detect_llm", return_value=None):
            return TaskPlanner()

    def _actions(self, intent: str) -> list[str]:
        planner = self._planner()
        steps = planner.plan(intent)
        return [s["action"] for s in steps]

    # Google
    def test_google_search(self) -> None:
        actions = self._actions("go to google and search python")
        assert "navigate" in actions
        assert "fill" in actions
        # First navigate URL must point to google
        planner = self._planner()
        steps = planner.plan("go to google and search python")
        assert steps[0]["url"] == "https://www.google.com"

    def test_google_search_variant(self) -> None:
        actions = self._actions("search for cats on google")
        assert "navigate" in actions
        assert "fill" in actions

    # Bing
    def test_bing_search(self) -> None:
        actions = self._actions("search python on bing")
        assert "navigate" in actions
        planner = self._planner()
        steps = planner.plan("search python on bing")
        assert steps[0]["url"] == "https://www.bing.com"

    # DuckDuckGo
    def test_ddg_search(self) -> None:
        planner = self._planner()
        steps = planner.plan("search privacy on duckduckgo")
        assert steps[0]["url"] == "https://duckduckgo.com"

    # YouTube
    def test_youtube_search(self) -> None:
        planner = self._planner()
        steps = planner.plan("search cats on youtube")
        assert steps[0]["url"] == "https://www.youtube.com"

    # Wikipedia
    def test_wikipedia_search(self) -> None:
        planner = self._planner()
        steps = planner.plan("search Python on wikipedia")
        assert steps[0]["url"] == "https://en.wikipedia.org"

    # Navigate
    def test_navigate_https_url(self) -> None:
        actions = self._actions("go to https://news.ycombinator.com")
        assert actions[0] == "navigate"

    def test_navigate_bare_domain(self) -> None:
        planner = self._planner()
        steps = planner.plan("open example.com")
        assert steps[0]["url"] == "https://example.com"

    # Scrape-and-save template
    def test_scrape_and_save(self) -> None:
        planner = self._planner()
        steps = planner.plan(
            "collect text from https://example.com and save to output.txt"
        )
        actions = [s["action"] for s in steps]
        assert "navigate" in actions
        assert "get_text" in actions
        assert "write_file" in actions
        # The write_file step should use {{last}}
        write = next(s for s in steps if s["action"] == "write_file")
        assert write["content"] == "{{last}}"
        assert write["path"] == "output.txt"

    # Unknown intent without LLM → ValueError
    def test_unknown_intent_raises(self) -> None:
        planner = self._planner()
        with pytest.raises(ValueError, match="No template matched"):
            planner.plan("do something completely unknown xyz123")

    # Steps are always valid
    def test_template_steps_are_valid(self) -> None:
        planner = self._planner()
        steps = planner.plan("go to google and search pytest")
        # validate_steps should not raise
        validated = validate_steps(steps)
        assert len(validated) == len(steps)


# ---------------------------------------------------------------------------
# TaskPlanner — plan with mocked LLM backend
# ---------------------------------------------------------------------------

class TestTaskPlannerLLMMock:
    def test_openai_backend_called(self) -> None:
        good_steps = [{"action": "navigate", "url": "https://example.com"}]
        with patch("task_planner._call_openai", return_value=good_steps) as mock_llm:
            with patch.object(TaskPlanner, "_detect_llm", return_value="openai"):
                planner = TaskPlanner()
                result = planner.plan("do something unique qwerty9999")
        mock_llm.assert_called_once()
        assert result == good_steps

    def test_ollama_backend_called(self) -> None:
        good_steps = [{"action": "navigate", "url": "https://example.com"}]
        with patch("task_planner._call_ollama", return_value=good_steps) as mock_llm:
            with patch.object(TaskPlanner, "_detect_llm", return_value="ollama"):
                planner = TaskPlanner()
                result = planner.plan("do something unique qwerty9999")
        mock_llm.assert_called_once()
        assert result == good_steps

    def test_template_wins_over_llm(self) -> None:
        """Template match should short-circuit before any LLM call."""
        with patch("task_planner._call_openai") as mock_llm:
            with patch.object(TaskPlanner, "_detect_llm", return_value="openai"):
                planner = TaskPlanner()
                planner.plan("go to google and search python")
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# set_network_intercept key collision regression test
# ---------------------------------------------------------------------------

class TestSetNetworkInterceptSchema:
    def test_intercept_action_key_not_overwritten_by_action(self) -> None:
        """
        The optional key for set_network_intercept must be 'intercept_action',
        NOT 'action' — to avoid the step's mandatory 'action' field being
        overwritten during validation.
        """
        schema = STEP_SCHEMA["set_network_intercept"]
        # The optional dict must NOT contain an 'action' key
        assert "action" not in schema["optional"], (
            "set_network_intercept optional key named 'action' collides with "
            "the mandatory step 'action' key. Rename it to 'intercept_action'."
        )
        # The intercept_action key must be present
        assert "intercept_action" in schema["optional"]

    def test_validate_set_network_intercept_preserves_action(self) -> None:
        """After validation the step action must still be 'set_network_intercept'."""
        steps = [{"action": "set_network_intercept", "url_pattern": "**/*.png"}]
        out = validate_steps(steps)
        assert out[0]["action"] == "set_network_intercept"

    def test_validate_set_network_intercept_default_intercept_action(self) -> None:
        """Default intercept_action should be 'abort'."""
        steps = [{"action": "set_network_intercept", "url_pattern": "**/*.png"}]
        out = validate_steps(steps)
        assert out[0]["intercept_action"] == "abort"

    def test_validate_set_network_intercept_custom_intercept_action(self) -> None:
        """Custom intercept_action 'continue' should be preserved."""
        steps = [{"action": "set_network_intercept", "url_pattern": "**/*.png", "intercept_action": "continue"}]
        out = validate_steps(steps)
        assert out[0]["intercept_action"] == "continue"


# ---------------------------------------------------------------------------
# _SYSTEM_PROMPT step count consistency
# ---------------------------------------------------------------------------

class TestSystemPromptStepCount:
    def test_system_prompt_max_steps_matches_validate_steps(self) -> None:
        """_SYSTEM_PROMPT must say 20 steps (matching validate_steps max of 20)."""
        from task_planner import _SYSTEM_PROMPT
        assert "20 steps" in _SYSTEM_PROMPT, (
            "_SYSTEM_PROMPT should say 'Maximum 20 steps' to match validate_steps limit"
        )
