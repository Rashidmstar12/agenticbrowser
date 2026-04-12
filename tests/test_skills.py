"""
tests/test_skills.py — Unit tests for skills.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from skills import (
    SkillDef,
    SkillLoadError,
    SkillRegistry,
    _trigger_to_pattern,
    load_from_file,
    load_from_url,
    resolve_steps,
)

# ---------------------------------------------------------------------------
# Trigger pattern compilation
# ---------------------------------------------------------------------------

def test_trigger_no_params():
    pattern, params = _trigger_to_pattern("google search")
    assert params == []
    assert pattern.fullmatch("google search")


def test_trigger_single_param():
    pattern, params = _trigger_to_pattern("search github for {{query}}")
    assert params == ["query"]
    m = pattern.fullmatch("search github for python asyncio")
    assert m is not None
    assert m.group(1) == "python asyncio"


def test_trigger_two_params():
    pattern, params = _trigger_to_pattern("login to {{site}} as {{user}}")
    assert params == ["site", "user"]
    m = pattern.fullmatch("login to github as alice")
    assert m is not None
    assert m.group(1) == "github"
    assert m.group(2) == "alice"


def test_trigger_case_insensitive():
    pattern, _ = _trigger_to_pattern("Search GitHub for {{q}}")
    assert pattern.fullmatch("search github for pytest")


# ---------------------------------------------------------------------------
# SkillDef
# ---------------------------------------------------------------------------

MINIMAL_SKILL_DICT = {
    "name":    "test_skill",
    "triggers": ["do the thing"],
    "steps":   [{"action": "navigate", "url": "https://example.com"}],
}


def test_skill_def_from_dict():
    s = SkillDef(
        name="test_skill",
        triggers=["do the thing"],
        steps=[{"action": "navigate", "url": "https://example.com"}],
    )
    assert s.name == "test_skill"
    assert len(s._compiled) == 1


def test_skill_def_to_dict():
    s = SkillDef(name="foo", steps=[{"action": "close_popups"}])
    d = s.to_dict()
    assert d["name"] == "foo"
    assert isinstance(d["steps"], list)


def test_resolve_steps_substitution():
    skill = SkillDef(
        name="github_search",
        triggers=["search github for {{query}}"],
        steps=[{"action": "navigate", "url": "https://github.com/search?q={{query}}"}],
    )
    steps = resolve_steps(skill, {"query": "python async"})
    assert steps[0]["url"] == "https://github.com/search?q=python async"


def test_resolve_steps_no_params():
    skill = SkillDef(
        name="simple",
        steps=[{"action": "close_popups"}],
    )
    steps = resolve_steps(skill, {})
    assert steps == [{"action": "close_popups"}]


# ---------------------------------------------------------------------------
# load_from_file
# ---------------------------------------------------------------------------

def test_load_from_file_single(tmp_path):
    skill_file = tmp_path / "skill.json"
    skill_file.write_text(json.dumps(MINIMAL_SKILL_DICT))
    with patch("skills._get_allowed_actions", return_value={"navigate"}):
        skills = load_from_file(skill_file)
    assert len(skills) == 1
    assert skills[0].name == "test_skill"


def test_load_from_file_list(tmp_path):
    data = [MINIMAL_SKILL_DICT, {**MINIMAL_SKILL_DICT, "name": "skill2"}]
    f = tmp_path / "skills.json"
    f.write_text(json.dumps(data))
    with patch("skills._get_allowed_actions", return_value={"navigate"}):
        skills = load_from_file(f)
    assert len(skills) == 2


def test_load_from_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_from_file("/nonexistent/path/skill.json")


def test_load_from_file_invalid_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("not json at all {{{")
    with pytest.raises(SkillLoadError):
        load_from_file(f)


def test_load_from_file_missing_name(tmp_path):
    f = tmp_path / "skill.json"
    f.write_text(json.dumps({"steps": [{"action": "navigate", "url": "x"}]}))
    with patch("skills._get_allowed_actions", return_value={"navigate"}):
        with pytest.raises(SkillLoadError, match="missing required keys"):
            load_from_file(f)


def test_load_from_file_missing_steps(tmp_path):
    f = tmp_path / "skill.json"
    f.write_text(json.dumps({"name": "foo"}))
    with patch("skills._get_allowed_actions", return_value={"navigate"}):
        with pytest.raises(SkillLoadError):
            load_from_file(f)


def test_load_from_file_unknown_action(tmp_path):
    f = tmp_path / "skill.json"
    f.write_text(json.dumps({"name": "foo", "steps": [{"action": "hack_the_planet"}]}))
    with patch("skills._get_allowed_actions", return_value={"navigate", "click"}):
        with pytest.raises(SkillLoadError, match="unknown action"):
            load_from_file(f)


# ---------------------------------------------------------------------------
# load_from_url
# ---------------------------------------------------------------------------

def test_load_from_url_success():
    skill_json = json.dumps(MINIMAL_SKILL_DICT)
    resp_mock = MagicMock()
    resp_mock.__enter__ = lambda s: s
    resp_mock.__exit__ = MagicMock(return_value=False)
    resp_mock.read.return_value = skill_json.encode()

    with patch("urllib.request.urlopen", return_value=resp_mock), \
         patch("skills._get_allowed_actions", return_value={"navigate"}):
        skills = load_from_url("https://example.com/skill.json")

    assert len(skills) == 1
    assert skills[0].name == "test_skill"
    assert skills[0].source == "https://example.com/skill.json"


def test_load_from_url_network_error():
    with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
        with pytest.raises(SkillLoadError, match="Failed to fetch"):
            load_from_url("https://example.com/skill.json")


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

def test_registry_register_and_list():
    reg = SkillRegistry()
    skill = SkillDef(name="my_skill", steps=[{"action": "close_popups"}])
    reg.register(skill)
    assert len(reg) == 1
    assert reg.get("my_skill") is skill
    assert reg.list_skills() == [skill]


def test_registry_unregister():
    reg = SkillRegistry()
    skill = SkillDef(name="x", steps=[{"action": "close_popups"}])
    reg.register(skill)
    assert reg.unregister("x") is True
    assert reg.get("x") is None
    assert reg.unregister("x") is False


def test_registry_match_no_params():
    reg = SkillRegistry()
    reg.register(SkillDef(
        name="close_all",
        triggers=["close everything"],
        steps=[{"action": "close_popups"}],
    ))
    result = reg.match("close everything")
    assert result is not None
    skill, params = result
    assert skill.name == "close_all"
    assert params == {}


def test_registry_match_with_params():
    reg = SkillRegistry()
    reg.register(SkillDef(
        name="github_search",
        triggers=["search github for {{query}}"],
        steps=[{"action": "navigate", "url": "https://github.com/search?q={{query}}"}],
    ))
    result = reg.match("search github for python asyncio")
    assert result is not None
    skill, params = result
    assert skill.name == "github_search"
    assert params == {"query": "python asyncio"}


def test_registry_match_no_match():
    reg = SkillRegistry()
    reg.register(SkillDef(name="foo", triggers=["foo"], steps=[{"action": "close_popups"}]))
    assert reg.match("totally unrelated intent") is None


def test_registry_overwrite():
    reg = SkillRegistry()
    v1 = SkillDef(name="skill", version="1.0", steps=[{"action": "close_popups"}])
    v2 = SkillDef(name="skill", version="2.0", steps=[{"action": "close_popups"}])
    reg.register(v1)
    reg.register(v2)
    assert reg.get("skill").version == "2.0"
    assert len(reg) == 1


def test_registry_load_source_file(tmp_path):
    f = tmp_path / "skill.json"
    f.write_text(json.dumps(MINIMAL_SKILL_DICT))
    reg = SkillRegistry()
    with patch("skills._get_allowed_actions", return_value={"navigate"}):
        loaded = reg.load_from_source(str(f))
    assert len(loaded) == 1
    assert reg.get("test_skill") is not None


def test_registry_load_source_directory(tmp_path):
    for i in range(3):
        (tmp_path / f"skill{i}.json").write_text(
            json.dumps({**MINIMAL_SKILL_DICT, "name": f"skill{i}"})
        )
    reg = SkillRegistry()
    with patch("skills._get_allowed_actions", return_value={"navigate"}):
        loaded = reg.load_from_source(str(tmp_path))
    assert len(loaded) == 3
    assert len(reg) == 3


def test_registry_load_source_url():
    skill_json = json.dumps(MINIMAL_SKILL_DICT)
    resp_mock = MagicMock()
    resp_mock.__enter__ = lambda s: s
    resp_mock.__exit__ = MagicMock(return_value=False)
    resp_mock.read.return_value = skill_json.encode()

    reg = SkillRegistry()
    with patch("urllib.request.urlopen", return_value=resp_mock), \
         patch("skills._get_allowed_actions", return_value={"navigate"}):
        loaded = reg.load_from_source("https://example.com/skill.json")
    assert len(loaded) == 1


def test_registry_clear():
    reg = SkillRegistry()
    reg.register(SkillDef(name="x", steps=[{"action": "close_popups"}]))
    reg.clear()
    assert len(reg) == 0
