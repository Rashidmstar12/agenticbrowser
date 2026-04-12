import hashlib
import json
import os
from pathlib import Path

import pytest

import system_tools

# ── run_shell_command ─────────────────────────────────────────────────────────

def test_run_shell_command_success():
    result = system_tools.run_shell_command("echo hello")
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]


def test_run_shell_command_stderr():
    result = system_tools.run_shell_command("echo err >&2; exit 1")
    assert result["returncode"] != 0
    assert "err" in result["stderr"]


def test_run_shell_command_failure():
    result = system_tools.run_shell_command("exit 42")
    assert result["returncode"] == 42


def test_run_shell_command_timeout():
    result = system_tools.run_shell_command("sleep 10", timeout=1)
    assert result["returncode"] == -1
    assert "timed out" in result["stderr"]


def test_run_shell_command_empty():
    result = system_tools.run_shell_command("true")
    assert result["returncode"] == 0


# ── read_file / write_file ────────────────────────────────────────────────────

def test_write_and_read_file(tmp_path):
    p = tmp_path / "test.txt"
    system_tools.write_file(str(p), "hello world")
    assert system_tools.read_file(str(p)) == "hello world"


def test_write_file_creates_parents(tmp_path):
    p = tmp_path / "a" / "b" / "c.txt"
    system_tools.write_file(str(p), "nested")
    assert p.read_text() == "nested"


def test_read_file_missing(tmp_path):
    with pytest.raises(Exception):
        system_tools.read_file(str(tmp_path / "missing.txt"))


def test_write_file_overwrites(tmp_path):
    p = tmp_path / "f.txt"
    system_tools.write_file(str(p), "first")
    system_tools.write_file(str(p), "second")
    assert system_tools.read_file(str(p)) == "second"


# ── delete_file ───────────────────────────────────────────────────────────────

def test_delete_file_exists(tmp_path):
    p = tmp_path / "del.txt"
    p.write_text("x")
    assert system_tools.delete_file(str(p)) is True
    assert not p.exists()


def test_delete_file_missing(tmp_path):
    assert system_tools.delete_file(str(tmp_path / "nope.txt")) is False


# ── list_files ────────────────────────────────────────────────────────────────

def test_list_files_all(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.txt").write_text("")
    files = system_tools.list_files(str(tmp_path))
    assert len(files) == 2


def test_list_files_pattern(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.py").write_text("")
    files = system_tools.list_files(str(tmp_path), "*.txt")
    assert len(files) == 1
    assert files[0].endswith("a.txt")


def test_list_files_empty_dir(tmp_path):
    assert system_tools.list_files(str(tmp_path)) == []


# ── file_exists ───────────────────────────────────────────────────────────────

def test_file_exists_true(tmp_path):
    p = tmp_path / "exists.txt"
    p.write_text("")
    assert system_tools.file_exists(str(p)) is True


def test_file_exists_false(tmp_path):
    assert system_tools.file_exists(str(tmp_path / "no.txt")) is False


# ── get_file_size ─────────────────────────────────────────────────────────────

def test_get_file_size(tmp_path):
    p = tmp_path / "size.txt"
    p.write_bytes(b"12345")
    assert system_tools.get_file_size(str(p)) == 5


def test_get_file_size_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        system_tools.get_file_size(str(tmp_path / "missing.txt"))


# ── hash_file ─────────────────────────────────────────────────────────────────

def test_hash_file_sha256(tmp_path):
    p = tmp_path / "h.txt"
    p.write_bytes(b"data")
    expected = hashlib.sha256(b"data").hexdigest()
    assert system_tools.hash_file(str(p)) == expected


def test_hash_file_md5(tmp_path):
    p = tmp_path / "h.txt"
    p.write_bytes(b"data")
    expected = hashlib.md5(b"data").hexdigest()
    assert system_tools.hash_file(str(p), "md5") == expected


def test_hash_file_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_bytes(b"")
    result = system_tools.hash_file(str(p))
    assert isinstance(result, str) and len(result) == 64


# ── get_env / set_env ─────────────────────────────────────────────────────────

def test_get_env_existing(monkeypatch):
    monkeypatch.setenv("TEST_VAR_XYZ", "myvalue")
    assert system_tools.get_env("TEST_VAR_XYZ") == "myvalue"


def test_get_env_missing_default():
    assert system_tools.get_env("DEFINITELY_NOT_SET_12345", "fallback") == "fallback"


def test_get_env_missing_none():
    assert system_tools.get_env("DEFINITELY_NOT_SET_12345") is None


def test_set_env(monkeypatch):
    system_tools.set_env("MY_TEST_KEY", "hello")
    assert os.environ.get("MY_TEST_KEY") == "hello"


# ── get_system_info ───────────────────────────────────────────────────────────

def test_get_system_info_keys():
    info = system_tools.get_system_info()
    assert "platform" in info
    assert "python_version" in info
    assert "hostname" in info
    assert "machine" in info
    assert "processor" in info


def test_get_system_info_types():
    info = system_tools.get_system_info()
    assert isinstance(info["platform"], str)
    assert isinstance(info["hostname"], str)


# ── parse_json / to_json ──────────────────────────────────────────────────────

def test_parse_json_dict():
    result = system_tools.parse_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_list():
    result = system_tools.parse_json('[1, 2, 3]')
    assert result == [1, 2, 3]


def test_parse_json_invalid():
    with pytest.raises(ValueError, match="Invalid JSON"):
        system_tools.parse_json("not json")


def test_to_json_basic():
    result = system_tools.to_json({"a": 1})
    parsed = json.loads(result)
    assert parsed == {"a": 1}


def test_to_json_indent():
    result = system_tools.to_json({"x": 1}, indent=4)
    assert "    " in result


def test_to_json_list():
    result = system_tools.to_json([1, 2, 3])
    assert json.loads(result) == [1, 2, 3]


# ── sanitize_filename ─────────────────────────────────────────────────────────

def test_sanitize_filename_clean():
    assert system_tools.sanitize_filename("hello_world.txt") == "hello_world.txt"


def test_sanitize_filename_special_chars():
    result = system_tools.sanitize_filename('file<name>.txt')
    assert '<' not in result
    assert '>' not in result


def test_sanitize_filename_slash():
    result = system_tools.sanitize_filename("path/file")
    assert '/' not in result


def test_sanitize_filename_colon():
    result = system_tools.sanitize_filename("C:\\file")
    assert ':' not in result or result.count(':') == 0


def test_sanitize_filename_all_specials():
    name = '<>:"/\\|?*'
    result = system_tools.sanitize_filename(name)
    for ch in '<>:"/\\|?*':
        assert ch not in result


# ── ensure_dir ────────────────────────────────────────────────────────────────

def test_ensure_dir_creates(tmp_path):
    new_dir = tmp_path / "new" / "nested"
    result = system_tools.ensure_dir(str(new_dir))
    assert Path(result).is_dir()
    assert result == str(new_dir)


def test_ensure_dir_existing(tmp_path):
    result = system_tools.ensure_dir(str(tmp_path))
    assert result == str(tmp_path)


# ── get_timestamp ─────────────────────────────────────────────────────────────

def test_get_timestamp_format():
    ts = system_tools.get_timestamp()
    # Should be parseable as ISO 8601 with timezone
    from datetime import datetime
    dt = datetime.fromisoformat(ts)
    assert dt.tzinfo is not None


def test_get_timestamp_utc():
    ts = system_tools.get_timestamp()
    assert "+00:00" in ts or "Z" in ts or "UTC" in ts or "00:00" in ts


# ── retry ─────────────────────────────────────────────────────────────────────

def test_retry_success_first_try():
    calls = []
    def fn():
        calls.append(1)
        return "ok"
    result = system_tools.retry(fn, max_attempts=3, delay=0)
    assert result == "ok"
    assert len(calls) == 1


def test_retry_success_after_failures():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "success"
    result = system_tools.retry(fn, max_attempts=3, delay=0)
    assert result == "success"
    assert len(calls) == 3


def test_retry_max_failures():
    def fn():
        raise RuntimeError("always fails")
    with pytest.raises(RuntimeError, match="always fails"):
        system_tools.retry(fn, max_attempts=3, delay=0)


def test_retry_single_attempt():
    def fn():
        return 42
    assert system_tools.retry(fn, max_attempts=1, delay=0) == 42
