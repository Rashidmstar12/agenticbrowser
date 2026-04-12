import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def run_shell_command(cmd: str, timeout: int = 30) -> dict:
    """Run a shell command and return stdout, stderr, returncode."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "Command timed out", "returncode": -1}


def read_file(path: str) -> str:
    """Read a file and return its contents as a string."""
    return Path(path).read_text()


def write_file(path: str, content: str) -> None:
    """Write content to a file, creating parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def delete_file(path: str) -> bool:
    """Delete a file. Returns True if deleted, False if not found."""
    p = Path(path)
    if p.exists():
        p.unlink()
        return True
    return False


def list_files(directory: str, pattern: str = "*") -> list[str]:
    """List files in directory matching glob pattern."""
    return [str(p) for p in Path(directory).glob(pattern) if p.is_file()]


def file_exists(path: str) -> bool:
    """Check if a file exists."""
    return Path(path).exists()


def get_file_size(path: str) -> int:
    """Return file size in bytes. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.stat().st_size


def hash_file(path: str, algorithm: str = "sha256") -> str:
    """Return hex digest of file content."""
    h = hashlib.new(algorithm)
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable."""
    return os.environ.get(key, default)


def set_env(key: str, value: str) -> None:
    """Set environment variable in the current process."""
    os.environ[key] = value


def get_system_info() -> dict:
    """Return platform, python version, hostname info."""
    import sys
    return {
        "platform": platform.system(),
        "python_version": sys.version,
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def parse_json(text: str) -> dict | list:
    """Parse JSON string, raise ValueError on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


def to_json(obj: object, indent: int = 2) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, indent=indent)


def sanitize_filename(name: str) -> str:
    """Replace invalid filename characters with underscore."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if missing. Return path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Return current ISO 8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def retry(fn, max_attempts: int = 3, delay: float = 0.1):
    """Call fn up to max_attempts times, returning on first success."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(delay)
    raise last_exc
