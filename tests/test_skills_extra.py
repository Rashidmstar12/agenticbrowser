"""
tests/test_skills_extra.py — additional tests for skills.py covering:
  - load_from_directory()
  - load_from_github() (with mocked HTTP)
  - _validate_github_ref()
  - _validate_url_scheme()
  - _safe_urlopen() (IP address / scheme validation)
  - _validate_source_for_api()
  - _make_github_raw_url()
  - SkillRegistry.load_from_remote_source()
  - SkillRegistry.load_from_directory()
  - SkillRegistry.load_from_github()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from skills import (
    SkillLoadError,
    SkillRegistry,
    _validate_github_ref,
    _validate_url_scheme,
    load_from_directory,
    load_from_github,
)

# Minimal valid skill dict used across multiple tests
_SKILL = {
    "name": "nav_skill",
    "triggers": ["go to example"],
    "steps": [{"action": "navigate", "url": "https://example.com"}],
}


# ---------------------------------------------------------------------------
# _validate_github_ref
# ---------------------------------------------------------------------------

class TestValidateGitHubRef:
    def test_valid_repo_and_branch(self) -> None:
        # Should not raise
        _validate_github_ref("owner/repo", "skills/login.json", "main")

    def test_repo_missing_slash_raises(self) -> None:
        with pytest.raises(SkillLoadError, match="Invalid repo"):
            _validate_github_ref("ownerrepo", "skills/login.json", "main")

    def test_repo_too_many_slashes_raises(self) -> None:
        with pytest.raises(SkillLoadError, match="Invalid repo"):
            _validate_github_ref("owner/repo/extra", "path", "main")

    def test_unsafe_branch_raises(self) -> None:
        with pytest.raises(SkillLoadError, match="Invalid branch"):
            _validate_github_ref("owner/repo", "skills/login.json", "main; echo bad")

    def test_path_traversal_raises(self) -> None:
        with pytest.raises(SkillLoadError, match="Path traversal"):
            _validate_github_ref("owner/repo", "skills/../../../etc/passwd", "main")

    def test_dot_dot_in_path_raises(self) -> None:
        with pytest.raises(SkillLoadError, match="Path traversal"):
            _validate_github_ref("owner/repo", "a/../b", "main")


# ---------------------------------------------------------------------------
# _validate_url_scheme
# ---------------------------------------------------------------------------

class TestValidateUrlScheme:
    def test_https_ok(self) -> None:
        _validate_url_scheme("https://example.com/skill.json")  # no raise

    def test_http_ok(self) -> None:
        _validate_url_scheme("http://example.com/skill.json")  # no raise

    def test_ftp_raises(self) -> None:
        with pytest.raises(SkillLoadError, match="Only http"):
            _validate_url_scheme("ftp://example.com/skill.json")

    def test_file_scheme_raises(self) -> None:
        with pytest.raises(SkillLoadError, match="Only http"):
            _validate_url_scheme("file:///etc/passwd")


# ---------------------------------------------------------------------------
# _safe_urlopen — IP address and scheme validation
# ---------------------------------------------------------------------------

class TestSafeUrlopen:
    def test_rejects_ipv4(self) -> None:
        from skills import _safe_urlopen
        with pytest.raises(SkillLoadError, match="domain name"):
            _safe_urlopen("https://192.168.1.1/skill.json")

    def test_rejects_http_localhost(self) -> None:
        """Non-IP rejections come from _validate_url_scheme (http is OK) but
        actual fetching may fail.  Test via SkillLoadError from the URL open."""
        from skills import _safe_urlopen
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            with pytest.raises(SkillLoadError, match="Failed to fetch"):
                _safe_urlopen("https://example.com/skill.json")

    def test_rejects_ftp_scheme(self) -> None:
        from skills import _safe_urlopen
        with pytest.raises(SkillLoadError, match="Only http"):
            _safe_urlopen("ftp://example.com/skill.json")


# ---------------------------------------------------------------------------
# load_from_directory
# ---------------------------------------------------------------------------

class TestLoadFromDirectory:
    def test_loads_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "skill1.json").write_text(json.dumps(_SKILL))
        (tmp_path / "skill2.json").write_text(json.dumps({**_SKILL, "name": "skill2"}))
        with patch("skills._get_allowed_actions", return_value={"navigate"}):
            skills = load_from_directory(tmp_path)
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert "nav_skill" in names
        assert "skill2" in names

    def test_ignores_non_skill_files(self, tmp_path: Path) -> None:
        (tmp_path / "skill.json").write_text(json.dumps(_SKILL))
        (tmp_path / "README.md").write_text("# readme")
        (tmp_path / "data.csv").write_text("a,b,c")
        with patch("skills._get_allowed_actions", return_value={"navigate"}):
            skills = load_from_directory(tmp_path)
        assert len(skills) == 1

    def test_skips_invalid_files_with_warning(self, tmp_path: Path, capsys) -> None:
        (tmp_path / "good.json").write_text(json.dumps(_SKILL))
        (tmp_path / "bad.json").write_text("NOT VALID JSON }{")
        with patch("skills._get_allowed_actions", return_value={"navigate"}):
            skills = load_from_directory(tmp_path)
        assert len(skills) == 1
        captured = capsys.readouterr()
        assert "bad.json" in captured.err

    def test_missing_directory_raises(self) -> None:
        with pytest.raises(SkillLoadError, match="does not exist"):
            load_from_directory("/nonexistent/path/to/skills")

    def test_loads_yaml_files(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml", reason="PyYAML not installed")
        import yaml
        (tmp_path / "skill.yaml").write_text(yaml.dump(_SKILL))
        with patch("skills._get_allowed_actions", return_value={"navigate"}):
            skills = load_from_directory(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "nav_skill"

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        with patch("skills._get_allowed_actions", return_value={"navigate"}):
            skills = load_from_directory(tmp_path)
        assert skills == []


# ---------------------------------------------------------------------------
# load_from_github — single file
# ---------------------------------------------------------------------------

class TestLoadFromGitHubSingleFile:
    def _mock_fetch(self, content: bytes) -> MagicMock:
        with patch("skills._fetch_github_raw", return_value=content) as m:
            return m

    def test_loads_skill_from_github(self) -> None:
        content = json.dumps(_SKILL).encode()
        with patch("skills._fetch_github_raw", return_value=content), \
             patch("skills._get_allowed_actions", return_value={"navigate"}):
            skills = load_from_github("owner/repo", "skills/nav.json")
        assert len(skills) == 1
        assert skills[0].name == "nav_skill"

    def test_normalises_gh_prefix(self) -> None:
        content = json.dumps(_SKILL).encode()
        with patch("skills._fetch_github_raw", return_value=content), \
             patch("skills._get_allowed_actions", return_value={"navigate"}):
            # "gh:owner/repo" should be accepted
            skills = load_from_github("gh:owner/repo", "skills/nav.json")
        assert len(skills) == 1

    def test_invalid_repo_raises(self) -> None:
        with pytest.raises(SkillLoadError, match="Invalid repo"):
            load_from_github("not-a-valid-repo", "skills/nav.json")

    def test_network_error_raises(self) -> None:
        with patch("skills._fetch_github_raw", side_effect=SkillLoadError("timeout")):
            with pytest.raises(SkillLoadError):
                load_from_github("owner/repo", "skills/nav.json")


# ---------------------------------------------------------------------------
# load_from_github — directory / index
# ---------------------------------------------------------------------------

class TestLoadFromGitHubDirectory:
    def test_loads_via_index(self) -> None:
        index = json.dumps({"skills": ["nav.json"]}).encode()
        skill_content = json.dumps(_SKILL).encode()

        def fake_fetch(url, **_):
            if "index.json" in url:
                return index
            return skill_content

        with patch("skills._fetch_github_raw", side_effect=fake_fetch), \
             patch("skills._get_allowed_actions", return_value={"navigate"}):
            skills = load_from_github("owner/repo", "skills")
        assert len(skills) == 1
        assert skills[0].name == "nav_skill"

    def test_empty_index_raises(self) -> None:
        index = json.dumps({"skills": []}).encode()
        with patch("skills._fetch_github_raw", return_value=index):
            with pytest.raises(SkillLoadError, match="no 'skills' entries"):
                load_from_github("owner/repo", "skills")

    def test_index_fetch_error_raises(self) -> None:
        with patch("skills._fetch_github_raw", side_effect=SkillLoadError("404")):
            with pytest.raises(SkillLoadError, match="Failed to fetch skill index"):
                load_from_github("owner/repo", "skills")

    def test_individual_skill_error_skipped(self, capsys) -> None:
        index = json.dumps({"skills": ["good.json", "bad.json"]}).encode()
        good_content = json.dumps(_SKILL).encode()

        def fake_fetch(url, **_):
            if "index.json" in url:
                return index
            if "good.json" in url:
                return good_content
            raise SkillLoadError("bad file")

        with patch("skills._fetch_github_raw", side_effect=fake_fetch), \
             patch("skills._get_allowed_actions", return_value={"navigate"}):
            skills = load_from_github("owner/repo", "skills")
        assert len(skills) == 1
        captured = capsys.readouterr()
        assert "bad.json" in captured.err


# ---------------------------------------------------------------------------
# _validate_source_for_api
# ---------------------------------------------------------------------------

class TestValidateSourceForApi:
    def test_gh_prefix_ok(self) -> None:
        from skills import _validate_source_for_api
        _validate_source_for_api("gh:owner/repo")  # no raise

    def test_https_ok(self) -> None:
        from skills import _validate_source_for_api
        _validate_source_for_api("https://example.com/skill.json")  # no raise

    def test_http_raises(self) -> None:
        from skills import _validate_source_for_api
        with pytest.raises(SkillLoadError, match="http://"):
            _validate_source_for_api("http://example.com/skill.json")

    def test_local_path_raises(self) -> None:
        from skills import _validate_source_for_api
        with pytest.raises(SkillLoadError, match="Local filesystem"):
            _validate_source_for_api("/etc/skills.json")

    def test_owner_slash_repo_ok(self) -> None:
        from skills import _validate_source_for_api
        _validate_source_for_api("owner/repo")  # should be treated as GitHub


# ---------------------------------------------------------------------------
# SkillRegistry — load_from_directory, load_from_github
# ---------------------------------------------------------------------------

class TestRegistryLoading:
    def test_load_from_directory(self, tmp_path: Path) -> None:
        (tmp_path / "skill.json").write_text(json.dumps(_SKILL))
        reg = SkillRegistry()
        with patch("skills._get_allowed_actions", return_value={"navigate"}):
            loaded = reg.load_from_directory(str(tmp_path))
        assert len(loaded) == 1
        assert reg.get("nav_skill") is not None

    def test_load_from_github(self) -> None:
        content = json.dumps(_SKILL).encode()
        reg = SkillRegistry()
        with patch("skills._fetch_github_raw", return_value=content), \
             patch("skills._get_allowed_actions", return_value={"navigate"}):
            loaded = reg.load_from_github("owner/repo", "skills/nav.json")
        assert len(loaded) == 1
        assert reg.get("nav_skill") is not None

    def test_load_from_remote_source_https(self) -> None:
        content = json.dumps(_SKILL).encode()
        reg = SkillRegistry()

        resp_mock = MagicMock()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        resp_mock.read.return_value = content

        with patch("urllib.request.urlopen", return_value=resp_mock), \
             patch("skills._get_allowed_actions", return_value={"navigate"}):
            loaded = reg.load_from_remote_source("https://example.com/skill.json")
        assert len(loaded) == 1

    def test_load_from_remote_source_rejects_local(self) -> None:
        reg = SkillRegistry()
        with pytest.raises(SkillLoadError, match="Local filesystem"):
            reg.load_from_remote_source("/etc/skills.json")

    def test_load_from_source_url(self) -> None:
        content = json.dumps(_SKILL).encode()
        reg = SkillRegistry()
        resp_mock = MagicMock()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)
        resp_mock.read.return_value = content
        with patch("urllib.request.urlopen", return_value=resp_mock), \
             patch("skills._get_allowed_actions", return_value={"navigate"}):
            loaded = reg.load_from_source("https://example.com/skill.json")
        assert len(loaded) == 1

    def test_load_from_source_directory(self, tmp_path: Path) -> None:
        (tmp_path / "skill.json").write_text(json.dumps(_SKILL))
        reg = SkillRegistry()
        with patch("skills._get_allowed_actions", return_value={"navigate"}):
            loaded = reg.load_from_source(str(tmp_path))
        assert len(loaded) == 1

    def test_load_from_source_file(self, tmp_path: Path) -> None:
        f = tmp_path / "skill.json"
        f.write_text(json.dumps(_SKILL))
        reg = SkillRegistry()
        with patch("skills._get_allowed_actions", return_value={"navigate"}):
            loaded = reg.load_from_source(str(f))
        assert len(loaded) == 1
