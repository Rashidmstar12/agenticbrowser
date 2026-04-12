from task_planner import (
    STEP_SCHEMA,
    Step,
    Task,
    TaskResult,
    merge_tasks,
    plan_task,
    summarize_results,
    validate_step,
    validate_task,
)

# ── STEP_SCHEMA ───────────────────────────────────────────────────────────────

def test_step_schema_length():
    assert len(STEP_SCHEMA) == 32


def test_step_schema_contains_navigate():
    assert "navigate" in STEP_SCHEMA


def test_step_schema_contains_all_expected():
    expected = [
        "navigate", "click", "fill", "type_text", "clear", "press_key",
        "scroll", "scroll_to_element", "screenshot", "get_text", "get_attribute",
        "get_url", "get_title", "hover", "double_click", "right_click",
        "wait_for_selector", "wait_text", "select_option", "check", "uncheck",
        "upload_file", "extract_links", "extract_table", "assert_text", "assert_url",
        "save_cookies", "load_cookies", "new_tab", "switch_tab", "close_tab", "list_tabs",
    ]
    for action in expected:
        assert action in STEP_SCHEMA, f"Missing: {action}"


# ── validate_step ─────────────────────────────────────────────────────────────

def test_validate_step_valid():
    assert validate_step({"action": "navigate"}) is True


def test_validate_step_all_valid():
    for action in STEP_SCHEMA:
        assert validate_step({"action": action}) is True


def test_validate_step_invalid_action():
    assert validate_step({"action": "fly"}) is False


def test_validate_step_missing_action():
    assert validate_step({}) is False


def test_validate_step_action_not_string():
    assert validate_step({"action": 42}) is False


# ── validate_task ─────────────────────────────────────────────────────────────

def test_validate_task_valid():
    task = {"name": "t", "steps": [{"action": "navigate"}]}
    assert validate_task(task) == []


def test_validate_task_missing_name():
    task = {"steps": [{"action": "navigate"}]}
    errors = validate_task(task)
    assert any("name" in e for e in errors)


def test_validate_task_empty_name():
    task = {"name": "", "steps": [{"action": "navigate"}]}
    errors = validate_task(task)
    assert any("name" in e for e in errors)


def test_validate_task_empty_steps():
    task = {"name": "t", "steps": []}
    errors = validate_task(task)
    assert any("step" in e.lower() for e in errors)


def test_validate_task_no_steps_key():
    task = {"name": "t"}
    errors = validate_task(task)
    assert len(errors) > 0


def test_validate_task_invalid_step():
    task = {"name": "t", "steps": [{"action": "bad_action"}]}
    errors = validate_task(task)
    assert any("bad_action" in e for e in errors)


def test_validate_task_multiple_errors():
    task = {"steps": [{"action": "bad"}]}
    errors = validate_task(task)
    assert len(errors) >= 2


def test_validate_task_step_index_in_error():
    task = {"name": "t", "steps": [{"action": "navigate"}, {"action": "oops"}]}
    errors = validate_task(task)
    assert any("1" in e for e in errors)


# ── plan_task ─────────────────────────────────────────────────────────────────

def test_plan_task_with_url():
    task = plan_task("my task", "Go to https://example.com")
    assert task.name == "my task"
    assert len(task.steps) == 1
    assert task.steps[0].action == "navigate"
    assert task.steps[0].params["url"] == "https://example.com"


def test_plan_task_without_url():
    task = plan_task("fallback", "Search something")
    assert task.steps[0].params["url"] == "https://example.com"


def test_plan_task_extracts_first_url():
    task = plan_task("t", "Visit https://foo.com and also https://bar.com")
    assert task.steps[0].params["url"] == "https://foo.com"


def test_plan_task_returns_task_instance():
    result = plan_task("t", "do something")
    assert isinstance(result, Task)


# ── summarize_results ─────────────────────────────────────────────────────────

def test_summarize_all_success():
    results = [{"success": True}, {"success": True}]
    s = summarize_results(results)
    assert s == {"total": 2, "success": 2, "failed": 0, "errors": []}


def test_summarize_all_failed():
    results = [{"success": False, "error": "e1"}, {"success": False, "error": "e2"}]
    s = summarize_results(results)
    assert s["total"] == 2
    assert s["success"] == 0
    assert s["failed"] == 2
    assert s["errors"] == ["e1", "e2"]


def test_summarize_mixed():
    results = [{"success": True}, {"success": False, "error": "oops"}]
    s = summarize_results(results)
    assert s["total"] == 2
    assert s["success"] == 1
    assert s["failed"] == 1


def test_summarize_empty():
    s = summarize_results([])
    assert s == {"total": 0, "success": 0, "failed": 0, "errors": []}


def test_summarize_no_error_key():
    results = [{"success": False}]
    s = summarize_results(results)
    assert s["errors"] == []


# ── merge_tasks ───────────────────────────────────────────────────────────────

def test_merge_two_tasks():
    t1 = Task(name="a", steps=[Step(action="navigate")])
    t2 = Task(name="b", steps=[Step(action="click")])
    merged = merge_tasks(t1, t2)
    assert merged.name == "a + b"
    assert len(merged.steps) == 2


def test_merge_three_tasks():
    t1 = Task(name="a", steps=[Step(action="navigate")])
    t2 = Task(name="b", steps=[Step(action="click")])
    t3 = Task(name="c", steps=[Step(action="fill")])
    merged = merge_tasks(t1, t2, t3)
    assert len(merged.steps) == 3
    assert "a + b + c" == merged.name


def test_merge_no_tasks():
    merged = merge_tasks()
    assert merged.name == "merged"
    assert merged.steps == []


def test_merge_single_task():
    t = Task(name="solo", steps=[Step(action="navigate")])
    merged = merge_tasks(t)
    assert merged.name == "solo"
    assert len(merged.steps) == 1


# ── Model validation ──────────────────────────────────────────────────────────

def test_step_model_defaults():
    s = Step(action="click")
    assert s.params == {}


def test_step_model_with_params():
    s = Step(action="fill", params={"selector": "#id", "value": "hello"})
    assert s.params["selector"] == "#id"


def test_task_model():
    t = Task(name="test", steps=[Step(action="navigate")])
    assert t.name == "test"
    assert len(t.steps) == 1


def test_task_result_model():
    tr = TaskResult(name="t", success=True, results=[{"success": True}])
    assert tr.error is None


def test_task_result_with_error():
    tr = TaskResult(name="t", success=False, results=[], error="failed")
    assert tr.error == "failed"
