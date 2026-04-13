"""
Extra tests for skills.py — covering _semver_satisfies edge cases,
load_from_file/load_from_directory edge cases, SkillRegistry.load_from_source,
_validate_source_for_api, _validate_github_ref, _parse_text YAML fallback, etc.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from skills import (
    SkillDef,
    SkillLoadError,
    SkillRegistry,
    _dict_to_skill,
    _make_github_raw_url,
    _parse_semver,
    _semver_satisfies,
    _validate_github_ref,
    _validate_source_for_api,
    _validate_url_scheme,
    load_from_directory,
    load_from_file,
    load_from_github,
    load_from_url,
)

# ---------------------------------------------------------------------------
# _parse_semver edge cases
# ---------------------------------------------------------------------------

class TestParseSemver:
    def test_full_version(self):
        assert _parse_semver("1.2.3") == (1, 2, 3)

    def test_short_major_only(self):
        assert _parse_semver("2") == (2, 0, 0)

    def test_short_major_minor(self):
        assert _parse_semver("1.4") == (1, 4, 0)

    def test_leading_trailing_whitespace(self):
        assert _parse_semver("  2.0.0  ") == (2, 0, 0)

    def test_too_many_parts_raises(self):
        with pytest.raises(SkillLoadError, match="Invalid semver"):
            _parse_semver("1.2.3.4")

    def test_non_numeric_raises(self):
        with pytest.raises(SkillLoadError, match="Invalid semver"):
            _parse_semver("1.x.0")

    def test_empty_raises(self):
        with pytest.raises(SkillLoadError):
            _parse_semver("")


# ---------------------------------------------------------------------------
# _semver_satisfies
# ---------------------------------------------------------------------------

class TestSemverSatisfies:
    def test_ge_satisfied(self):
        assert _semver_satisfies("2.0.0", ">=1.0.0") is True

    def test_ge_equal(self):
        assert _semver_satisfies("1.0.0", ">=1.0.0") is True

    def test_ge_not_satisfied(self):
        assert _semver_satisfies("0.9.0", ">=1.0.0") is False

    def test_gt_satisfied(self):
        assert _semver_satisfies("1.1.0", ">1.0.0") is True

    def test_gt_not_satisfied_equal(self):
        assert _semver_satisfies("1.0.0", ">1.0.0") is False

    def test_le_satisfied(self):
        assert _semver_satisfies("1.0.0", "<=2.0.0") is True

    def test_le_equal(self):
        assert _semver_satisfies("2.0.0", "<=2.0.0") is True

    def test_le_not_satisfied(self):
        assert _semver_satisfies("3.0.0", "<=2.0.0") is False

    def test_lt_satisfied(self):
        assert _semver_satisfies("1.9.9", "<2.0.0") is True

    def test_lt_not_satisfied_equal(self):
        assert _semver_satisfies("2.0.0", "<2.0.0") is False

    def test_exact_match(self):
        assert _semver_satisfies("1.2.3", "==1.2.3") is True

    def test_exact_no_match(self):
        assert _semver_satisfies("1.2.4", "==1.2.3") is False

    def test_bare_version_treated_as_ge(self):
        assert _semver_satisfies("1.5.0", "1.0.0") is True
        assert _semver_satisfies("0.9.0", "1.0.0") is False

    def test_single_equals_treated_as_ge(self):
        assert _semver_satisfies("1.0.0", "=1.0.0") is True


# ---------------------------------------------------------------------------
# _validate_url_scheme
# ---------------------------------------------------------------------------

class TestValidateUrlScheme:
    def test_https_ok(self):
        _validate_url_scheme("https://example.com/skill.json")

    def test_http_ok(self):
        _validate_url_scheme("http://example.com/skill.json")

    def test_ftp_raises(self):
        with pytest.raises(SkillLoadError, match="Only http"):
            _validate_url_scheme("ftp://example.com/skill.json")

    def test_file_raises(self):
        with pytest.raises(SkillLoadError):
            _validate_url_scheme("file:///etc/passwd")


# ---------------------------------------------------------------------------
# _validate_github_ref
# ---------------------------------------------------------------------------

class TestValidateGithubRef:
    def test_valid_repo(self):
        _validate_github_ref("owner/repo", "skills/index.json", "main")

    def test_invalid_repo_extra_slash(self):
        with pytest.raises(SkillLoadError, match="Invalid repo"):
            _validate_github_ref("owner/repo/extra", "skills/index.json", "main")

    def test_invalid_branch_special_chars(self):
        with pytest.raises(SkillLoadError, match="Invalid branch"):
            _validate_github_ref("owner/repo", "skills/index.json", "br<anch!")

    def test_path_traversal_rejected(self):
        with pytest.raises(SkillLoadError, match="traversal"):
            _validate_github_ref("owner/repo", "skills/../etc/passwd", "main")

    def test_repo_with_allowed_chars(self):
        _validate_github_ref("my-org/my_repo.v2", "path/to/skill.json", "feat-branch")


# ---------------------------------------------------------------------------
# _make_github_raw_url
# ---------------------------------------------------------------------------

class TestMakeGithubRawUrl:
    def test_builds_correct_url(self):
        url = _make_github_raw_url("owner/repo", "main", "skills/login.json")
        assert url.startswith("https://raw.githubusercontent.com/owner/repo/main/")
        assert "login.json" in url


# ---------------------------------------------------------------------------
# _validate_source_for_api
# ---------------------------------------------------------------------------

class TestValidateSourceForApi:
    def test_gh_shorthand_ok(self):
        _validate_source_for_api("gh:owner/repo")

    def test_owner_slash_repo_ok(self):
        _validate_source_for_api("owner/repo")

    def test_https_url_ok(self):
        _validate_source_for_api("https://example.com/skill.json")

    def test_http_url_rejected(self):
        with pytest.raises(SkillLoadError, match="http://"):
            _validate_source_for_api("http://example.com/skill.json")

    def test_local_path_rejected(self):
        with pytest.raises(SkillLoadError, match="Local filesystem"):
            _validate_source_for_api("/path/to/skill.json")

    def test_relative_path_rejected(self):
        with pytest.raises(SkillLoadError, match="Local filesystem"):
            _validate_source_for_api("./skills/skill.json")


# ---------------------------------------------------------------------------
# load_from_file — edge cases
# ---------------------------------------------------------------------------

class TestLoadFromFile:
    def test_loads_single_skill_dict(self, tmp_path):
        skill_data = {
            "name": "file_skill",
            "steps": [{"action": "close_popups"}],
        }
        f = tmp_path / "skill.json"
        f.write_text(json.dumps(skill_data))
        skills = load_from_file(f)
        assert len(skills) == 1
        assert skills[0].name == "file_skill"

    def test_loads_list_of_skills(self, tmp_path):
        skills_data = [
            {"name": "skill_a", "steps": [{"action": "close_popups"}]},
            {"name": "skill_b", "steps": [{"action": "press", "key": "Escape"}]},
        ]
        f = tmp_path / "skills.json"
        f.write_text(json.dumps(skills_data))
        skills = load_from_file(f)
        assert len(skills) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_from_file(tmp_path / "nonexistent.json")

    def test_invalid_json_without_yaml_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not: valid json [}")
        # JSON parse fails; without PyYAML it raises SkillLoadError
        with pytest.raises(SkillLoadError):
            load_from_file(f)

    def test_wrong_root_type_raises(self, tmp_path):
        f = tmp_path / "wrong.json"
        f.write_text(json.dumps("just a string"))
        with pytest.raises(SkillLoadError, match="expected a skill object"):
            load_from_file(f)


# ---------------------------------------------------------------------------
# load_from_directory
# ---------------------------------------------------------------------------

class TestLoadFromDirectory:
    def test_loads_all_json_files(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps(
            {"name": "skill_a", "steps": [{"action": "close_popups"}]}
        ))
        (tmp_path / "b.json").write_text(json.dumps(
            {"name": "skill_b", "steps": [{"action": "press", "key": "Escape"}]}
        ))
        skills = load_from_directory(tmp_path)
        names = {s.name for s in skills}
        assert "skill_a" in names
        assert "skill_b" in names

    def test_ignores_non_skill_files(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps(
            {"name": "skill_a", "steps": [{"action": "close_popups"}]}
        ))
        (tmp_path / "readme.md").write_text("# readme")
        (tmp_path / "config.txt").write_text("some config")
        skills = load_from_directory(tmp_path)
        assert len(skills) == 1

    def test_skips_invalid_files_with_warning(self, tmp_path, capsys):
        (tmp_path / "good.json").write_text(json.dumps(
            {"name": "good", "steps": [{"action": "close_popups"}]}
        ))
        (tmp_path / "bad.json").write_text("invalid json {{{")
        skills = load_from_directory(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "good"
        err = capsys.readouterr().err
        assert "bad.json" in err or "Warning" in err

    def test_nonexistent_directory_raises(self):
        with pytest.raises(SkillLoadError, match="does not exist"):
            load_from_directory("/tmp/definitely_nonexistent_dir_xyz_999")


# ---------------------------------------------------------------------------
# load_from_url
# ---------------------------------------------------------------------------

class TestLoadFromUrl:
    def _mock_urlopen(self, content: bytes):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = content
        return resp

    def test_loads_from_url(self):
        skill_data = {"name": "url_skill", "steps": [{"action": "close_popups"}]}
        resp = self._mock_urlopen(json.dumps(skill_data).encode())
        with patch("urllib.request.urlopen", return_value=resp):
            skills = load_from_url("https://example.com/skill.json")
        assert len(skills) == 1
        assert skills[0].name == "url_skill"

    def test_raises_on_network_error(self):
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with pytest.raises(SkillLoadError, match="Failed to fetch"):
                load_from_url("https://example.com/skill.json")

    def test_rejects_http_url(self):
        with pytest.raises(SkillLoadError):
            load_from_url("http://example.com/skill.json")

    def test_wrong_root_type_raises(self):
        resp = self._mock_urlopen(json.dumps(42).encode())
        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(SkillLoadError, match="expected a skill object"):
                load_from_url("https://example.com/skill.json")


# ---------------------------------------------------------------------------
# SkillRegistry.load_from_source dispatch
# ---------------------------------------------------------------------------

class TestSkillRegistryLoadFromSource:
    def test_loads_local_file(self, tmp_path):
        reg = SkillRegistry()
        f = tmp_path / "skill.json"
        f.write_text(json.dumps({"name": "sk", "steps": [{"action": "close_popups"}]}))
        skills = reg.load_from_source(str(f))
        assert len(skills) == 1
        assert reg.get("sk") is not None

    def test_loads_local_directory(self, tmp_path):
        reg = SkillRegistry()
        (tmp_path / "a.json").write_text(json.dumps(
            {"name": "dir_skill", "steps": [{"action": "close_popups"}]}
        ))
        skills = reg.load_from_source(str(tmp_path))
        assert len(skills) == 1

    def test_loads_from_gh_shorthand(self):
        reg = SkillRegistry()
        skill = SkillDef(name="remote", steps=[{"action": "close_popups"}])
        with patch.object(reg, "load_from_github", return_value=[skill]) as mock_gh:
            skills = reg.load_from_source("gh:owner/repo")
        mock_gh.assert_called_once_with("gh:owner/repo")
        assert len(skills) == 1

    def test_loads_from_https_url(self):
        reg = SkillRegistry()
        skill = SkillDef(name="remote", steps=[{"action": "close_popups"}])
        with patch.object(reg, "load_from_url", return_value=[skill]) as mock_url:
            reg.load_from_source("https://example.com/skill.json")
        mock_url.assert_called_once_with("https://example.com/skill.json")

    def test_github_pattern_without_prefix(self):
        reg = SkillRegistry()
        skill = SkillDef(name="remote", steps=[{"action": "close_popups"}])
        with patch.object(reg, "load_from_github", return_value=[skill]) as mock_gh:
            reg.load_from_source("owner/my-repo")
        mock_gh.assert_called_once()

    def test_load_from_remote_source_rejects_local(self):
        reg = SkillRegistry()
        with pytest.raises(SkillLoadError, match="Local filesystem"):
            reg.load_from_remote_source("/local/path/skill.json")

    def test_load_from_remote_source_accepts_gh(self):
        reg = SkillRegistry()
        skill = SkillDef(name="remote", steps=[{"action": "close_popups"}])
        with patch.object(reg, "load_from_github", return_value=[skill]):
            skills = reg.load_from_remote_source("gh:owner/repo")
        assert len(skills) == 1

    def test_load_from_remote_source_accepts_https(self):
        reg = SkillRegistry()
        skill = SkillDef(name="remote", steps=[{"action": "close_popups"}])
        with patch.object(reg, "load_from_url", return_value=[skill]):
            skills = reg.load_from_remote_source("https://example.com/skill.json")
        assert len(skills) == 1


# ---------------------------------------------------------------------------
# SkillRegistry misc
# ---------------------------------------------------------------------------

class TestSkillRegistryMisc:
    def test_bool_false_when_empty(self):
        reg = SkillRegistry()
        assert not reg

    def test_bool_true_when_has_skills(self):
        reg = SkillRegistry()
        reg.register(SkillDef(name="s", steps=[{"action": "close_popups"}]))
        assert reg

    def test_len(self):
        reg = SkillRegistry()
        assert len(reg) == 0
        reg.register(SkillDef(name="s1", steps=[{"action": "close_popups"}]))
        assert len(reg) == 1
        reg.register(SkillDef(name="s2", steps=[{"action": "press", "key": "Escape"}]))
        assert len(reg) == 2

    def test_overwrite_existing(self):
        reg = SkillRegistry()
        reg.register(SkillDef(name="sk", steps=[{"action": "close_popups"}]))
        reg.register(SkillDef(name="sk", steps=[{"action": "press", "key": "Escape"}]))
        assert len(reg) == 1
        assert reg.get("sk").steps[0]["action"] == "press"

    def test_unregister_existing(self):
        reg = SkillRegistry()
        reg.register(SkillDef(name="sk", steps=[{"action": "close_popups"}]))
        removed = reg.unregister("sk")
        assert removed is True
        assert reg.get("sk") is None

    def test_unregister_nonexistent_returns_false(self):
        reg = SkillRegistry()
        assert reg.unregister("nonexistent") is False

    def test_list_skills(self):
        reg = SkillRegistry()
        reg.register(SkillDef(name="a", steps=[{"action": "close_popups"}]))
        reg.register(SkillDef(name="b", steps=[{"action": "close_popups"}]))
        names = {s.name for s in reg.list_skills()}
        assert names == {"a", "b"}

    def test_register_many(self):
        reg = SkillRegistry()
        skills = [
            SkillDef(name="x", steps=[{"action": "close_popups"}]),
            SkillDef(name="y", steps=[{"action": "press", "key": "Escape"}]),
        ]
        reg.register_many(skills)
        assert len(reg) == 2

    def test_match_returns_none_when_no_match(self):
        reg = SkillRegistry()
        reg.register(SkillDef(
            name="sk", triggers=["do the specific thing"],
            steps=[{"action": "close_popups"}]
        ))
        assert reg.match("unrelated intent xyz") is None

    def test_match_returns_skill_and_params(self):
        reg = SkillRegistry()
        reg.register(SkillDef(
            name="search_skill",
            triggers=["search github for {{query}}"],
            steps=[{"action": "navigate", "url": "https://github.com/search?q={{query}}"}],
        ))
        result = reg.match("search github for python asyncio")
        assert result is not None
        skill, params = result
        assert skill.name == "search_skill"
        assert params.get("query") == "python asyncio"


# ---------------------------------------------------------------------------
# _dict_to_skill — validation edge cases
# ---------------------------------------------------------------------------

class TestDictToSkill:
    def test_missing_name_raises(self):
        with pytest.raises(SkillLoadError, match="missing required keys"):
            _dict_to_skill({"steps": [{"action": "close_popups"}]})

    def test_missing_steps_raises(self):
        with pytest.raises(SkillLoadError, match="missing required keys"):
            _dict_to_skill({"name": "sk"})

    def test_empty_name_raises(self):
        with pytest.raises(SkillLoadError, match="non-empty"):
            _dict_to_skill({"name": "  ", "steps": [{"action": "close_popups"}]})

    def test_empty_steps_list_raises(self):
        with pytest.raises(SkillLoadError, match="non-empty"):
            _dict_to_skill({"name": "sk", "steps": []})

    def test_step_missing_action_raises(self):
        with pytest.raises(SkillLoadError, match="missing 'action'"):
            _dict_to_skill({"name": "sk", "steps": [{"url": "https://example.com"}]})

    def test_unknown_action_raises(self):
        with pytest.raises(SkillLoadError, match="unknown action"):
            _dict_to_skill({"name": "sk", "steps": [{"action": "fly_to_mars"}]})

    def test_valid_skill_creates_skilldef(self):
        skill = _dict_to_skill({
            "name": "my_skill",
            "description": "Does something",
            "version": "2.0.0",
            "steps": [{"action": "close_popups"}],
            "triggers": ["do something"],
        })
        assert skill.name == "my_skill"
        assert skill.version == "2.0.0"


# ---------------------------------------------------------------------------
# SkillDef.to_dict and resolve_steps
# ---------------------------------------------------------------------------

class TestSkillDefToDict:
    def test_to_dict_structure(self):
        skill = SkillDef(
            name="test",
            description="A test skill",
            version="1.0.0",
            triggers=["do something"],
            steps=[{"action": "close_popups"}],
        )
        d = skill.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "1.0.0"
        assert d["steps"][0]["action"] == "close_popups"

    def test_resolve_steps_no_params(self):
        skill = SkillDef(
            name="no_params",
            steps=[{"action": "navigate", "url": "https://fixed.example.com"}],
        )
        resolved = skill.resolve_steps({})
        assert resolved[0]["url"] == "https://fixed.example.com"

    def test_resolve_steps_with_params(self):
        skill = SkillDef(
            name="with_params",
            steps=[{"action": "navigate", "url": "https://example.com/search?q={{query}}"}],
            parameters={"query": ""},
        )
        resolved = skill.resolve_steps({"query": "python asyncio"})
        assert "python asyncio" in resolved[0]["url"]

    def test_min_version_check_satisfied(self):
        skill = SkillDef(
            name="versioned",
            version="2.0.0",
            min_version={"agenticbrowser": "1.0.0"},
            steps=[{"action": "close_popups"}],
        )
        # should not raise when loaded
        assert skill.version == "2.0.0"


# ---------------------------------------------------------------------------
# Additional skills coverage: load_from_github directory mode, edge cases
# ---------------------------------------------------------------------------



class TestLoadFromGithubDirectory:
    """Test skills.load_from_github in directory mode (no file extension)."""

    _VALID_SKILL = {
        "name": "test-skill",
        "trigger": "run test",
        "steps": [{"action": "navigate", "url": "https://example.com"}],
    }

    def _make_github_response(self, content: bytes):
        """Build a mock urllib response object."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = content
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_directory_mode_fetches_index_and_skills(self):

        index_json = json.dumps({"skills": ["skill1.yaml"]}).encode()
        skill_yaml = (
            "name: test-skill\n"
            "trigger: run test\n"
            "steps:\n"
            "  - action: navigate\n"
            "    url: https://example.com\n"
        ).encode()

        responses = [index_json, skill_yaml]
        call_count = [0]

        def _fake_urlopen(req, timeout, context):
            data = responses[call_count[0]]
            call_count[0] += 1
            return self._make_github_response(data)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = load_from_github("owner/repo", "skills")
        assert len(result) == 1
        assert result[0].name == "test-skill"

    def test_directory_mode_empty_index_raises(self):

        index_json = json.dumps({"skills": []}).encode()

        def _fake_urlopen(req, timeout, context):
            return self._make_github_response(index_json)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            with pytest.raises(SkillLoadError, match="no 'skills' entries"):
                load_from_github("owner/repo", "dir")

    def test_directory_mode_missing_index_raises(self):

        def _fake_urlopen(req, timeout, context):
            raise urllib.error.URLError("404")

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            with pytest.raises(SkillLoadError, match="Failed to fetch"):
                load_from_github("owner/repo", "missing-dir")

    def test_directory_mode_skips_bad_skill_file(self, capsys):
        """A bad skill file is warned and skipped, not a fatal error."""

        index_json = json.dumps({"skills": ["bad.yaml", "good.yaml"]}).encode()
        bad_yaml = b"not: a: valid: skill"
        good_yaml = (
            "name: good-skill\n"
            "trigger: good\n"
            "steps:\n"
            "  - action: navigate\n"
            "    url: https://good.example\n"
        ).encode()

        responses = [index_json, bad_yaml, good_yaml]
        call_count = [0]

        def _fake_urlopen(req, timeout, context):
            data = responses[call_count[0]]
            call_count[0] += 1
            return self._make_github_response(data)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = load_from_github("owner/repo", "mixed-dir")
        # good.yaml should still be loaded; bad.yaml is skipped with a warning
        assert any(s.name == "good-skill" for s in result)

    def test_file_mode_with_yaml_extension(self):
        """load_from_github with a .yaml path is treated as a single file."""

        skill_yaml = (
            "name: file-skill\n"
            "trigger: file test\n"
            "steps:\n"
            "  - action: navigate\n"
            "    url: https://example.com\n"
        ).encode()

        def _fake_urlopen(req, timeout, context):
            return self._make_github_response(skill_yaml)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = load_from_github("owner/repo", "skills/my-skill.yaml")
        assert len(result) == 1
        assert result[0].name == "file-skill"

    def test_file_mode_with_json_extension(self):

        skill_json = json.dumps(self._VALID_SKILL).encode()

        def _fake_urlopen(req, timeout, context):
            return self._make_github_response(skill_json)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = load_from_github("owner/repo", "skills/skill.json")
        assert len(result) == 1

    def test_gh_prefix_stripped(self):
        """'gh:owner/repo' prefix should be stripped before use."""

        skill_yaml = (
            "name: prefixed-skill\n"
            "trigger: prefixed\n"
            "steps:\n"
            "  - action: navigate\n"
            "    url: https://example.com\n"
        ).encode()

        def _fake_urlopen(req, timeout, context):
            return self._make_github_response(skill_yaml)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            result = load_from_github("gh:owner/repo", "skills/skill.yaml")
        assert len(result) == 1

    def test_invalid_repo_format_raises(self):

        with pytest.raises((SkillLoadError, ValueError)):
            load_from_github("not-valid-repo-format-nodash", "path/file.yaml")
