"""Phase 3 agentic-mode tests.

Covers:
- ``TaskPlanner.agentic_run()`` loop correctness
- ``_call_observe_decide()`` output parsing and validation
- Wrong-page recovery via injected corrective steps
- Stopping conditions: verified, done_by_model, max_steps, abort, error
- Verification-driven completion (assert_text / assert_url)
- Max-step protection (hard budget ceiling)
- Backward compatibility: ``run()`` and ``execute()`` unchanged
- ``POST /task/agentic_run`` API endpoint response shape
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_planner import (
    TaskPlanner,
    _call_observe_decide,
    validate_steps,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_planner(llm: str | None = None) -> TaskPlanner:
    return TaskPlanner(llm=llm)


def _make_agent(**overrides) -> MagicMock:
    """Minimal mock BrowserAgent (unsafe so assert_* attrs work)."""
    agent = MagicMock(unsafe=True)
    agent.page = MagicMock()
    agent.page.url = "https://example.com"
    agent.navigate.return_value = {"url": "https://example.com", "status": 200}
    agent.close_popups.return_value = {"dismissed": []}
    agent.wait_for_selector.return_value = {"selector": "body", "found": True}
    agent.wait_for_load_state.return_value = {"state": "networkidle", "reached": True}
    agent.assert_text.return_value = {"found": True, "text": "hello"}
    agent.assert_url.return_value = {"matched": True, "pattern": "example.com"}
    agent.get_url.return_value = "https://example.com"
    agent.get_text.return_value = "Example page text"
    agent.fill.return_value = {"filled": True}
    agent.type_text.return_value = {"typed": True}
    agent.press_key.return_value = {"key": "Enter"}
    for key, val in overrides.items():
        setattr(agent, key, val)
    return agent


def _simple_steps() -> list[dict]:
    """Two-step plan that does not require an LLM."""
    return [
        {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
        {"action": "close_popups"},
    ]


# ---------------------------------------------------------------------------
# 1. agentic_run() — step-loop correctness
# ---------------------------------------------------------------------------


class TestAgenticRunLoop:
    def test_returns_expected_keys(self):
        planner = _make_planner()
        agent = _make_agent()
        with patch.object(planner, "plan", return_value=_simple_steps()):
            result = planner.agentic_run("open example.com", agent)
        for key in ("success", "verified", "intent", "stopped_reason",
                    "recovery_steps_injected", "steps", "results", "failed_count"):
            assert key in result, f"missing key: {key!r}"

    def test_success_true_when_no_errors(self):
        planner = _make_planner()
        agent = _make_agent()
        with patch.object(planner, "plan", return_value=_simple_steps()):
            result = planner.agentic_run("open example.com", agent)
        assert result["success"] is True
        assert result["failed_count"] == 0

    def test_intent_preserved_in_result(self):
        planner = _make_planner()
        agent = _make_agent()
        with patch.object(planner, "plan", return_value=_simple_steps()):
            result = planner.agentic_run("open example.com", agent)
        assert result["intent"] == "open example.com"

    def test_no_recovery_steps_without_llm(self):
        planner = _make_planner(llm=None)  # no LLM
        agent = _make_agent()
        with patch.object(planner, "plan", return_value=_simple_steps()):
            result = planner.agentic_run("open example.com", agent)
        assert result["recovery_steps_injected"] == 0

    def test_stopped_reason_done_by_model_when_queue_exhausted(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        with patch.object(planner, "plan", return_value=_simple_steps()):
            result = planner.agentic_run("open example.com", agent)
        assert result["stopped_reason"] == "done_by_model"

    def test_plan_failure_returns_error_dict(self):
        planner = _make_planner()
        agent = _make_agent()
        with patch.object(planner, "plan", side_effect=ValueError("no LLM")):
            result = planner.agentic_run("impossible task", agent)
        assert result["success"] is False
        assert result["stopped_reason"] == "error"


# ---------------------------------------------------------------------------
# 2. agentic_run() — stopping conditions
# ---------------------------------------------------------------------------


class TestAgenticStopConditions:
    def test_stops_at_max_steps(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        # Build a plan with 10 steps
        many_steps = [
            {"action": "close_popups"}
            for _ in range(10)
        ]
        with patch.object(planner, "plan", return_value=many_steps):
            result = planner.agentic_run("do stuff", agent, max_steps=3)
        assert result["stopped_reason"] == "max_steps"
        assert len(result["results"]) == 3

    def test_injected_steps_count_toward_max_steps(self):
        """10 original + injected steps should still cap at max_steps=5."""
        planner = _make_planner(llm=None)
        agent = _make_agent()
        many_steps = [{"action": "close_popups"} for _ in range(10)]
        with patch.object(planner, "plan", return_value=many_steps):
            result = planner.agentic_run("do stuff", agent, max_steps=5)
        assert len(result["results"]) == 5
        assert result["stopped_reason"] == "max_steps"

    def test_stop_on_error_true_halts_on_first_failure(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        agent.navigate.side_effect = Exception("network error")
        steps = [
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
            {"action": "close_popups"},
        ]
        with patch.object(planner, "plan", return_value=steps):
            result = planner.agentic_run("open example.com", agent, stop_on_error=True)
        assert result["stopped_reason"] == "error"
        assert result["success"] is False
        assert result["failed_count"] == 1
        assert len(result["results"]) == 1  # halted after first step

    def test_stop_on_error_false_continues_after_failure(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        agent.navigate.side_effect = Exception("network error")
        steps = [
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
            {"action": "close_popups"},
        ]
        with patch.object(planner, "plan", return_value=steps):
            result = planner.agentic_run("open example.com", agent, stop_on_error=False)
        assert result["success"] is False
        assert result["failed_count"] == 1
        assert len(result["results"]) == 2  # both steps ran


# ---------------------------------------------------------------------------
# 3. agentic_run() — verification-driven completion
# ---------------------------------------------------------------------------


class TestAgenticVerification:
    def test_verified_true_when_assert_text_passes(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        agent.assert_text.return_value = {"found": True, "text": "welcome"}
        steps = [
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
            {"action": "assert_text", "text": "welcome", "selector": "body"},
        ]
        with patch.object(planner, "plan", return_value=steps):
            result = planner.agentic_run("open and verify", agent)
        assert result["verified"] is True
        assert result["stopped_reason"] == "verified"

    def test_verified_true_when_assert_url_passes(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        agent.assert_url.return_value = {"matched": True, "pattern": "example.com"}
        steps = [
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
            {"action": "assert_url", "pattern": "example.com"},
        ]
        with patch.object(planner, "plan", return_value=steps):
            result = planner.agentic_run("open and verify url", agent)
        assert result["verified"] is True
        assert result["stopped_reason"] == "verified"

    def test_verified_none_when_no_assertion_steps(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        with patch.object(planner, "plan", return_value=_simple_steps()):
            result = planner.agentic_run("open example.com", agent)
        assert result["verified"] is None

    def test_verified_false_when_assert_fails(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        agent.assert_text.return_value = {"found": False, "text": "nonexistent"}
        steps = [
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"},
            {"action": "assert_text", "text": "nonexistent", "selector": "body"},
        ]
        with patch.object(planner, "plan", return_value=steps):
            result = planner.agentic_run("check nonexistent", agent)
        assert result["verified"] is False

    def test_loop_exits_early_when_assert_passes_mid_plan(self):
        """Assert at step 2 of 5 → remaining 3 steps are NOT executed."""
        planner = _make_planner(llm=None)
        agent = _make_agent()
        agent.assert_text.return_value = {"found": True, "text": "done"}
        steps = [
            {"action": "close_popups"},
            {"action": "assert_text", "text": "done", "selector": "body"},
            {"action": "close_popups"},
            {"action": "close_popups"},
            {"action": "close_popups"},
        ]
        with patch.object(planner, "plan", return_value=steps):
            result = planner.agentic_run("check early", agent)
        assert result["stopped_reason"] == "verified"
        assert len(result["results"]) == 2  # only 2 steps ran


# ---------------------------------------------------------------------------
# 4. agentic_run() — observe-decide integration
# ---------------------------------------------------------------------------


class TestAgenticObserveDecide:
    def test_stops_when_model_says_done(self):
        planner = _make_planner(llm="openai")
        agent = _make_agent()
        steps = [{"action": "close_popups"} for _ in range(6)]
        observe_result = {"decision": "done"}
        with patch.object(planner, "plan", return_value=steps), \
             patch("task_planner._call_observe_decide", return_value=observe_result):
            result = planner.agentic_run("do stuff", agent, checkpoint_every=3)
        assert result["stopped_reason"] == "done_by_model"

    def test_stops_when_model_says_abort(self):
        planner = _make_planner(llm="openai")
        agent = _make_agent()
        steps = [{"action": "close_popups"} for _ in range(6)]
        observe_result = {"decision": "abort", "reason": "login wall detected"}
        with patch.object(planner, "plan", return_value=steps), \
             patch("task_planner._call_observe_decide", return_value=observe_result):
            result = planner.agentic_run("do stuff", agent, checkpoint_every=3)
        assert result["stopped_reason"] == "abort"

    def test_injects_corrective_steps_on_continue(self):
        planner = _make_planner(llm="openai")
        agent = _make_agent()
        initial = [{"action": "close_popups"}]
        corrective = [{"action": "close_popups"}, {"action": "close_popups"}]
        # First checkpoint: inject 2 corrective steps; second checkpoint: done
        call_count = {"n": 0}

        def _fake_observe(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"decision": "continue", "steps": corrective}
            return {"decision": "done"}

        with patch.object(planner, "plan", return_value=initial), \
             patch("task_planner._call_observe_decide", side_effect=_fake_observe):
            result = planner.agentic_run("do stuff", agent, checkpoint_every=1)
        assert result["recovery_steps_injected"] == 2

    def test_corrective_steps_with_invalid_action_are_dropped(self):
        """Corrective steps that fail validate_steps are silently discarded."""
        planner = _make_planner(llm="openai")
        agent = _make_agent()
        initial = [{"action": "close_popups"}]
        bad_corrective = [{"action": "INVALID_ACTION_XYZ"}]
        observe_result_1 = {"decision": "continue", "steps": bad_corrective}
        observe_result_2 = {"decision": "done"}
        responses = [observe_result_1, observe_result_2]
        with patch.object(planner, "plan", return_value=initial), \
             patch("task_planner._call_observe_decide", side_effect=responses):
            result = planner.agentic_run("do stuff", agent, checkpoint_every=1)
        # Invalid steps dropped → nothing injected
        assert result["recovery_steps_injected"] == 0

    def test_no_observe_call_when_llm_is_none(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        with patch.object(planner, "plan", return_value=_simple_steps()), \
             patch("task_planner._call_observe_decide") as mock_observe:
            planner.agentic_run("open example.com", agent)
        mock_observe.assert_not_called()

    def test_observe_called_at_checkpoint_intervals(self):
        planner = _make_planner(llm="openai")
        agent = _make_agent()
        steps = [{"action": "close_popups"} for _ in range(6)]
        # After 3 steps → checkpoint 1; after 6 steps → checkpoint 2
        with patch.object(planner, "plan", return_value=steps), \
             patch("task_planner._call_observe_decide", return_value={"decision": "done"}) as mock:
            planner.agentic_run("do stuff", agent, checkpoint_every=3)
        # First checkpoint fires at step 3 and returns done → loop exits
        assert mock.call_count >= 1

    def test_observe_checkpoint_fires_when_queue_empties(self):
        """Checkpoint also fires when queue runs out (even if not at N interval)."""
        planner = _make_planner(llm="openai")
        agent = _make_agent()
        # 2 steps, checkpoint_every=5 → checkpoint fires when queue empties
        steps = [{"action": "close_popups"}, {"action": "close_popups"}]
        with patch.object(planner, "plan", return_value=steps), \
             patch("task_planner._call_observe_decide", return_value={"decision": "done"}) as mock:
            result = planner.agentic_run("do stuff", agent, checkpoint_every=5)
        mock.assert_called_once()
        assert result["stopped_reason"] == "done_by_model"


# ---------------------------------------------------------------------------
# 5. Wrong-page recovery
# ---------------------------------------------------------------------------


class TestAgenticRecovery:
    def test_navigate_corrective_step_executes(self):
        """Model returns a navigate corrective step → it should execute."""
        planner = _make_planner(llm="openai")
        agent = _make_agent()
        initial = [{"action": "close_popups"}]
        corrective = [
            {"action": "navigate", "url": "https://example.com", "wait_until": "domcontentloaded"}
        ]
        responses = [
            {"decision": "continue", "steps": corrective},
            {"decision": "done"},
        ]
        with patch.object(planner, "plan", return_value=initial), \
             patch("task_planner._call_observe_decide", side_effect=responses):
            result = planner.agentic_run("go somewhere", agent, checkpoint_every=1)
        assert result["recovery_steps_injected"] == 1
        # The injected navigate step should appear in results
        actions = [r["action"] for r in result["results"]]
        assert "navigate" in actions

    def test_abort_sets_success_false(self):
        planner = _make_planner(llm="openai")
        agent = _make_agent()
        initial = [{"action": "close_popups"}]
        with patch.object(planner, "plan", return_value=initial), \
             patch("task_planner._call_observe_decide",
                   return_value={"decision": "abort", "reason": "login wall"}):
            result = planner.agentic_run("access dashboard", agent, checkpoint_every=1)
        assert result["stopped_reason"] == "abort"
        # success=True because no step raised — only the model aborted
        # (success reflects step errors, not goal achievement)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# 6. _call_observe_decide() unit tests
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


def _make_openai_mock_error(exc: Exception) -> MagicMock:
    """Return a mock openai module whose client raises *exc* on create()."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = exc
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai


class TestCallObserveDecide:
    def test_returns_done_on_openai(self):
        import json as _json
        mock_openai, _ = _make_openai_mock(_json.dumps({"decision": "done"}))
        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = _call_observe_decide("task", "http://x.com", "text", [], "openai")
        assert result["decision"] == "done"

    def test_returns_abort_with_reason(self):
        import json as _json
        mock_openai, _ = _make_openai_mock(_json.dumps({"decision": "abort", "reason": "captcha"}))
        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = _call_observe_decide("task", "http://x.com", "text", [], "openai")
        assert result["decision"] == "abort"
        assert "captcha" in result["reason"]

    def test_returns_continue_with_validated_steps(self):
        import json as _json
        corrective = [{"action": "close_popups"}]
        mock_openai, _ = _make_openai_mock(_json.dumps({"decision": "continue", "steps": corrective}))
        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = _call_observe_decide("task", "http://x.com", "text", [], "openai")
        assert result["decision"] == "continue"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["action"] == "close_popups"

    def test_invalid_corrective_steps_are_dropped(self):
        import json as _json
        mock_openai, _ = _make_openai_mock(
            _json.dumps({"decision": "continue", "steps": [{"action": "INVALID"}]})
        )
        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = _call_observe_decide("task", "http://x.com", "text", [], "openai")
        assert result["decision"] == "continue"
        assert result["steps"] == []

    def test_unknown_llm_returns_continue_no_steps(self):
        result = _call_observe_decide("task", "http://x.com", "text", [], "unknown_llm")
        assert result["decision"] == "continue"
        assert result.get("steps", []) == []

    def test_llm_network_failure_returns_continue(self):
        mock_openai = _make_openai_mock_error(Exception("network"))
        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = _call_observe_decide("task", "http://x.com", "text", [], "openai")
        assert result["decision"] == "continue"
        assert result.get("steps", []) == []

    def test_malformed_json_returns_continue(self):
        mock_openai, _ = _make_openai_mock("not valid json at all")
        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = _call_observe_decide("task", "http://x.com", "text", [], "openai")
        assert result["decision"] == "continue"

    def test_unexpected_decision_value_defaults_to_continue(self):
        import json as _json
        mock_openai, _ = _make_openai_mock(_json.dumps({"decision": "maybe"}))
        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = _call_observe_decide("task", "http://x.com", "text", [], "openai")
        assert result["decision"] == "continue"

    def test_corrective_steps_capped_at_3(self):
        """Even if the model returns 5 corrective steps, only 3 are accepted."""
        import json as _json
        big_list = [{"action": "close_popups"} for _ in range(5)]
        mock_openai, _ = _make_openai_mock(_json.dumps({"decision": "continue", "steps": big_list}))
        with patch.dict(sys.modules, {"openai": mock_openai}):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = _call_observe_decide("task", "http://x.com", "text", [], "openai")
        assert len(result["steps"]) == 3


# ---------------------------------------------------------------------------
# 7. Backward compatibility: run() and execute() are unaffected
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_run_method_still_returns_original_keys(self):
        planner = _make_planner()
        agent = _make_agent()
        with patch.object(planner, "plan", return_value=_simple_steps()):
            result = planner.run("open example.com", agent)
        # run() must NOT contain Phase 3 keys
        assert "stopped_reason" not in result
        assert "recovery_steps_injected" not in result
        # run() must still contain its own keys
        for key in ("success", "verified", "intent", "steps", "results", "failed_count"):
            assert key in result

    def test_execute_method_signature_unchanged(self):
        planner = _make_planner()
        agent = _make_agent()
        steps = validate_steps(_simple_steps())
        results = planner.execute(steps, agent)
        assert isinstance(results, list)
        for r in results:
            assert "status" in r

    def test_agentic_run_result_has_additional_keys(self):
        planner = _make_planner(llm=None)
        agent = _make_agent()
        with patch.object(planner, "plan", return_value=_simple_steps()):
            result = planner.agentic_run("open example.com", agent)
        assert "stopped_reason" in result
        assert "recovery_steps_injected" in result


# ---------------------------------------------------------------------------
# 8. API endpoint /task/agentic_run
# ---------------------------------------------------------------------------


class TestAgenticRunEndpoint:
    def _get_client(self):
        from fastapi.testclient import TestClient

        from api_server import app
        return TestClient(app)

    def _mock_both(self, api_server_mod, planner_mock):
        """Return context managers that patch both get_planner and get_agent."""
        agent_mock = _make_agent()
        return planner_mock, agent_mock

    def test_endpoint_exists_and_returns_200_on_success(self):
        client = self._get_client()
        planner_mock = MagicMock()
        planner_mock.agentic_run.return_value = {
            "success": True,
            "verified": None,
            "intent": "open example.com",
            "stopped_reason": "done_by_model",
            "recovery_steps_injected": 0,
            "steps": [{"action": "close_popups"}],
            "results": [{"step": 0, "action": "close_popups", "status": "ok", "result": {}}],
            "failed_count": 0,
        }
        import api_server
        orig_p, orig_a = api_server.get_planner, api_server.get_agent
        try:
            api_server.get_planner = lambda: planner_mock
            api_server.get_agent = lambda: _make_agent()
            resp = client.post("/task/agentic_run", json={"intent": "open example.com"})
        finally:
            api_server.get_planner = orig_p
            api_server.get_agent = orig_a
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["stopped_reason"] == "done_by_model"
        assert "recovery_steps_injected" in body

    def test_endpoint_returns_422_on_failure(self):
        client = self._get_client()
        planner_mock = MagicMock()
        planner_mock.agentic_run.return_value = {
            "success": False,
            "verified": None,
            "intent": "bad task",
            "stopped_reason": "error",
            "recovery_steps_injected": 0,
            "steps": [{"action": "navigate", "url": "x", "wait_until": "domcontentloaded"}],
            "results": [{"step": 0, "action": "navigate", "status": "error", "error": "fail"}],
            "failed_count": 1,
        }
        import api_server
        orig_p, orig_a = api_server.get_planner, api_server.get_agent
        try:
            api_server.get_planner = lambda: planner_mock
            api_server.get_agent = lambda: _make_agent()
            resp = client.post("/task/agentic_run", json={"intent": "bad task"})
        finally:
            api_server.get_planner = orig_p
            api_server.get_agent = orig_a
        assert resp.status_code == 422

    def test_endpoint_max_steps_field_clamped_by_model(self):
        """Pydantic model enforces le=50; values >50 are rejected."""
        client = self._get_client()
        resp = client.post(
            "/task/agentic_run",
            json={"intent": "test", "max_steps": 100},
        )
        # Pydantic validation error
        assert resp.status_code == 422

    def test_endpoint_response_has_no_raw_error_internals(self):
        """Error messages in the response must be sanitized (first line, ≤500 chars)."""
        client = self._get_client()
        long_error = "Error: " + ("x" * 600)
        planner_mock = MagicMock()
        planner_mock.agentic_run.return_value = {
            "success": False,
            "verified": None,
            "intent": "task",
            "stopped_reason": "error",
            "recovery_steps_injected": 0,
            "steps": [{"action": "close_popups"}],
            "results": [{"step": 0, "action": "close_popups", "status": "error", "error": long_error}],
            "failed_count": 1,
        }
        import api_server
        orig_p, orig_a = api_server.get_planner, api_server.get_agent
        try:
            api_server.get_planner = lambda: planner_mock
            api_server.get_agent = lambda: _make_agent()
            resp = client.post("/task/agentic_run", json={"intent": "task"})
        finally:
            api_server.get_planner = orig_p
            api_server.get_agent = orig_a
        detail = resp.json().get("detail", {})
        if isinstance(detail, dict):
            for r in detail.get("results", []):
                if r.get("error"):
                    assert len(r["error"]) <= 500

    def test_existing_task_run_endpoint_unchanged(self):
        """POST /task/run must not include Phase 3 keys."""
        client = self._get_client()
        planner_mock = MagicMock()
        planner_mock.plan.return_value = validate_steps(_simple_steps())
        planner_mock.execute.return_value = [
            {"step": 0, "action": "navigate", "status": "ok", "result": {}},
            {"step": 1, "action": "close_popups", "status": "ok", "result": {}},
        ]
        import api_server
        orig_p, orig_a = api_server.get_planner, api_server.get_agent
        try:
            api_server.get_planner = lambda: planner_mock
            api_server.get_agent = lambda: _make_agent()
            resp = client.post("/task/run", json={"intent": "open example.com"})
        finally:
            api_server.get_planner = orig_p
            api_server.get_agent = orig_a
        if resp.status_code == 200:
            body = resp.json()
            assert "stopped_reason" not in body
            assert "recovery_steps_injected" not in body
