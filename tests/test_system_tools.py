"""
Unit tests for SystemTools: file I/O, path-traversal safety, run_python, run_shell.

All tests operate inside a temporary directory so nothing is written to the
repository checkout or to the default ``./workspace`` folder.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

# Ensure repo root is on the path so the bare module names resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

from system_tools import PathTraversalError, SystemTools, safe_path


# ---------------------------------------------------------------------------
# safe_path
# ---------------------------------------------------------------------------

class TestSafePath:
    def test_relative_path_inside_workspace(self, tmp_path: Path) -> None:
        result = safe_path(tmp_path, "data/output.txt")
        assert result == (tmp_path / "data" / "output.txt").resolve()

    def test_dot_path_resolves_to_workspace(self, tmp_path: Path) -> None:
        result = safe_path(tmp_path, ".")
        assert result == tmp_path.resolve()

    def test_traversal_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_path(tmp_path, "../escape.txt")

    def test_absolute_path_outside_workspace_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_path(tmp_path, "/etc/passwd")

    def test_nested_traversal_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_path(tmp_path, "sub/../../etc/passwd")

    def test_path_inside_subdirectory_ok(self, tmp_path: Path) -> None:
        result = safe_path(tmp_path, "a/b/c/d.txt")
        assert str(result).startswith(str(tmp_path.resolve()))


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

class TestWriteReadFile:
    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.write_file("hello.txt", "hello world")
        assert result["bytes_written"] == len("hello world".encode())
        assert result["mode"] == "w"

        read = st.read_file("hello.txt")
        assert read["content"] == "hello world"
        assert read["size"] == len("hello world")

    def test_write_creates_subdirectory(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.write_file("sub/dir/file.txt", "data")
        assert (tmp_path / "sub" / "dir" / "file.txt").exists()

    def test_overwrite_replaces_content(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.write_file("f.txt", "first")
        st.write_file("f.txt", "second")
        assert st.read_file("f.txt")["content"] == "second"

    def test_write_traversal_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        with pytest.raises(PathTraversalError):
            st.write_file("../escape.txt", "bad")

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        with pytest.raises(FileNotFoundError):
            st.read_file("nonexistent.txt")

    def test_read_traversal_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        with pytest.raises(PathTraversalError):
            st.read_file("../secret.txt")


class TestAppendFile:
    def test_append_adds_content(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.write_file("log.txt", "line1\n")
        st.append_file("log.txt", "line2\n")
        content = st.read_file("log.txt")["content"]
        assert content == "line1\nline2\n"

    def test_append_creates_file_if_missing(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.append_file("new.txt", "data")
        assert st.read_file("new.txt")["content"] == "data"

    def test_append_returns_mode_a(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.append_file("x.txt", "y")
        assert result["mode"] == "a"


class TestListDir:
    def test_lists_files_and_dirs(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.write_file("a.txt", "a")
        st.write_file("b.txt", "b")
        (tmp_path / "subdir").mkdir()
        result = st.list_dir(".")
        names = {e["name"] for e in result["entries"]}
        assert "a.txt" in names
        assert "b.txt" in names
        assert "subdir" in names

    def test_list_dir_entry_types(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.write_file("f.txt", "")
        (tmp_path / "d").mkdir()
        result = st.list_dir(".")
        types = {e["name"]: e["type"] for e in result["entries"]}
        assert types["f.txt"] == "file"
        assert types["d"] == "dir"

    def test_list_dir_missing_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        with pytest.raises(FileNotFoundError):
            st.list_dir("no_such_dir")

    def test_list_dir_on_file_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.write_file("f.txt", "")
        with pytest.raises(NotADirectoryError):
            st.list_dir("f.txt")


class TestMakeDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.make_dir("new/nested/dir")
        assert (tmp_path / "new" / "nested" / "dir").is_dir()
        assert result["created"] is True

    def test_make_dir_idempotent(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.make_dir("d")
        result = st.make_dir("d")  # should not raise
        assert result["created"] is True

    def test_make_dir_traversal_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        with pytest.raises(PathTraversalError):
            st.make_dir("../outside")


class TestDeleteFile:
    def test_delete_file(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.write_file("to_delete.txt", "bye")
        result = st.delete_file("to_delete.txt")
        assert result["deleted"] is True
        assert not (tmp_path / "to_delete.txt").exists()

    def test_delete_directory(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        st.make_dir("mydir")
        st.write_file("mydir/file.txt", "hi")
        st.delete_file("mydir")
        assert not (tmp_path / "mydir").exists()

    def test_delete_missing_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        with pytest.raises(FileNotFoundError):
            st.delete_file("ghost.txt")

    def test_delete_traversal_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        with pytest.raises(PathTraversalError):
            st.delete_file("../outside.txt")


# ---------------------------------------------------------------------------
# run_python
# ---------------------------------------------------------------------------

class TestRunPython:
    def test_simple_print(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_python("print('hello')")
        assert result["success"] is True
        assert result["stdout"].strip() == "hello"
        assert result["exit_code"] == 0

    def test_syntax_error_is_captured(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_python("def (")
        assert result["success"] is False
        assert result["exit_code"] != 0

    def test_runtime_exception_captured(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_python("raise ValueError('boom')")
        assert result["success"] is False
        assert "ValueError" in result["stderr"]

    def test_extra_vars_injected(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_python("print(my_var)", extra_vars={"my_var": "injected"})
        assert result["success"] is True
        assert "injected" in result["stdout"]

    def test_timeout_returns_failure(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path, exec_timeout=1)
        result = st.run_python("import time; time.sleep(10)", timeout=1)
        assert result["success"] is False
        assert "timed out" in result["stderr"].lower()

    def test_stderr_captured(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_python("import sys; sys.stderr.write('err msg')")
        assert "err msg" in result["stderr"]

    def test_exit_code_nonzero_on_failure(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_python("import sys; sys.exit(42)")
        assert result["exit_code"] == 42
        assert result["success"] is False


# ---------------------------------------------------------------------------
# run_shell
# ---------------------------------------------------------------------------

class TestRunShell:
    def test_echo_command(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_shell("echo hello")
        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_invalid_command_captured(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_shell("false")  # unix 'false' exits with 1
        assert result["success"] is False

    def test_empty_command_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        with pytest.raises(ValueError):
            st.run_shell("")

    def test_timeout_returns_failure(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_shell("sleep 10", timeout=1)
        assert result["success"] is False
        assert "timed out" in result["stderr"].lower()

    def test_exit_code_present(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        result = st.run_shell("echo ok")
        assert "exit_code" in result

    def test_invalid_syntax_raises(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        with pytest.raises(ValueError):
            st.run_shell("echo 'unterminated")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

class TestInfo:
    def test_info_returns_workspace(self, tmp_path: Path) -> None:
        st = SystemTools(workspace=tmp_path)
        info = st.info()
        assert "workspace" in info
        assert str(tmp_path) in info["workspace"]
        assert "python" in info
        assert "file_count" in info
        assert "dir_count" in info
