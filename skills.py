"""
skills.py — External skill loader for agenticbrowser.

A *skill* is a named, reusable browser-automation recipe that can be loaded
from a local file, a directory, a remote URL, or a GitHub repository.
Once loaded, skills integrate seamlessly with the TaskPlanner so that natural-
language intents trigger skill steps instead of (or in addition to) the
built-in templates.

Skill file format — JSON (or YAML if PyYAML is installed):
---------------------------------------------------------------------------
{
    "name":        "github_search",
    "description": "Search for repositories on GitHub",
    "version":     "1.0.0",
    "author":      "example",
    "triggers": [
        "search github for {{query}}",
        "github search {{query}}",
        "find {{query}} on github"
    ],
    "parameters": {
        "query": {"description": "Search term", "required": true}
    },
    "steps": [
        {"action": "navigate",
         "url": "https://github.com/search?q={{query}}&type=repositories"},
        {"action": "close_popups"},
        {"action": "wait_state", "state": "networkidle"}
    ]
}
---------------------------------------------------------------------------

Loading sources
---------------
    load_from_file("path/to/skill.json")
    load_from_file("path/to/skill.yaml")          # requires PyYAML
    load_from_directory("skills/")                # loads *.json / *.yaml
    load_from_url("https://example.com/skill.json")
    load_from_github("owner/repo", "skills/login.json")
    load_from_github("owner/repo")                # fetches skills/index.json

Trigger parameters
------------------
Triggers may contain ``{{param_name}}`` placeholders that are converted to
regex capture groups.  When an intent matches, the captured values are
substituted into the skill's step definitions before execution.

Example:
    trigger  →  "search github for {{query}}"
    intent   →  "search github for python asyncio"
    result   →  query = "python asyncio"  →  substituted into steps
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse as _urlparse

__all__ = [
    "SkillDef",
    "SkillRegistry",
    "SkillLoadError",
    "load_from_file",
    "load_from_directory",
    "load_from_url",
    "load_from_github",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SkillLoadError(ValueError):
    """Raised when a skill file cannot be parsed or fails validation."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SkillDef:
    """An immutable skill definition loaded from an external source."""

    name:        str
    description: str                       = ""
    version:     str                       = "1.0.0"
    author:      str                       = ""
    triggers:    list[str]                 = field(default_factory=list)
    parameters:  dict[str, dict[str, Any]] = field(default_factory=dict)
    steps:       list[dict[str, Any]]      = field(default_factory=list)
    source:      str                       = ""   # path/URL this was loaded from

    # Compiled (trigger_pattern, [param_names]) pairs — populated by __post_init__
    _compiled: list[tuple[re.Pattern[str], list[str]]] = field(
        default_factory=list, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        self._compiled = [_trigger_to_pattern(t) for t in self.triggers]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name":        self.name,
            "description": self.description,
            "version":     self.version,
            "author":      self.author,
            "triggers":    self.triggers,
            "parameters":  self.parameters,
            "steps":       self.steps,
            "source":      self.source,
        }


# ---------------------------------------------------------------------------
# Trigger → regex conversion
# ---------------------------------------------------------------------------

def _trigger_to_pattern(trigger: str) -> tuple[re.Pattern[str], list[str]]:
    """
    Convert a trigger string with ``{{param}}`` placeholders into a compiled
    regex and an ordered list of parameter names.

    Example::

        _trigger_to_pattern("search github for {{query}}")
        # → (re.compile(r'\\s*search\\ github\\ for\\ (.+?)\\s*', re.I), ['query'])
    """
    param_names: list[str] = []
    parts: list[str] = []
    last = 0
    for m in re.finditer(r"\{\{(\w+)\}\}", trigger):
        parts.append(re.escape(trigger[last:m.start()]))
        parts.append(r"(.+?)")
        param_names.append(m.group(1))
        last = m.end()
    parts.append(re.escape(trigger[last:]))
    pattern_str = r"\s*" + "".join(parts) + r"\s*"
    return re.compile(pattern_str, re.IGNORECASE), param_names


def _substitute_params(steps: list[dict[str, Any]], params: dict[str, str]) -> list[dict[str, Any]]:
    """
    Return a deep copy of *steps* with all ``{{param_name}}`` tokens
    replaced by the corresponding values from *params*.
    """
    text = json.dumps(steps)
    for key, value in params.items():
        text = text.replace("{{" + key + "}}", value.replace('"', '\\"'))
    return json.loads(text)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_SKILL_KEYS = {"name", "steps"}
_STEP_ACTIONS: set[str] | None = None  # populated lazily from STEP_SCHEMA

# Strict allowlist for GitHub component characters (no traversal, no injection)
_GITHUB_SAFE_PART = re.compile(r"^[\w.\-]+$")


def _validate_github_ref(repo: str, path: str, branch: str) -> None:
    """
    Validate GitHub ``repo``, ``path``, and ``branch`` to prevent path-traversal
    and SSRF attacks.

    Each component is checked against a strict allowlist of safe characters.
    """
    repo_parts = repo.split("/")
    if len(repo_parts) != 2 or not all(_GITHUB_SAFE_PART.match(p) for p in repo_parts):
        raise SkillLoadError(
            f"Invalid repo {repo!r} — expected 'owner/repo' with only "
            "alphanumeric, '-', '_', and '.' characters"
        )
    if not _GITHUB_SAFE_PART.match(branch):
        raise SkillLoadError(
            f"Invalid branch {branch!r} — only alphanumeric, '-', '_', and '.' allowed"
        )
    # Reject path traversal sequences in the file path
    path_clean = path.strip("/")
    if ".." in path_clean.split("/"):
        raise SkillLoadError(f"Path traversal detected in skill path: {path!r}")


def _validate_url_scheme(url: str) -> None:
    """Reject non-https URLs to prevent SSRF via unexpected schemes."""
    parsed = _urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise SkillLoadError(
            f"Only http:// and https:// URLs are supported for skill loading: {url!r}"
        )


def _make_github_raw_url(repo: str, branch: str, file_path: str) -> str:
    """
    Build a ``raw.githubusercontent.com`` URL and verify the host was not
    altered by path-traversal in the components (defence-in-depth).
    """
    _GITHUB_RAW_HOST = "raw.githubusercontent.com"
    url = f"https://{_GITHUB_RAW_HOST}/{repo}/{branch}/{file_path}"
    parsed = _urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != _GITHUB_RAW_HOST:
        raise SkillLoadError(
            "Internal error: unexpected host in constructed GitHub URL — "
            "possible injection in repo/branch/path components"
        )
    return url


def _validate_source_for_api(source: str) -> None:
    """
    Guard for the HTTP API endpoint: only GitHub references and HTTPS URLs are
    permitted.  Local filesystem paths are rejected to prevent path injection
    through the API surface.
    """
    s = source.strip()
    if s.startswith("gh:") or re.match(r"^[\w-]+/[\w.\-]+(/|$)", s):
        return  # GitHub shorthand — OK
    parsed = _urlparse(s)
    if parsed.scheme == "https":
        return  # HTTPS URL — OK
    if parsed.scheme == "http":
        raise SkillLoadError("http:// is not allowed via the API; use https://")
    raise SkillLoadError(
        "Local filesystem paths are not permitted via the API. "
        "Use https:// or gh:owner/repo to load remote skills, "
        "or use  --skills <path>  from the CLI for local files."
    )


def _get_allowed_actions() -> set[str]:
    global _STEP_ACTIONS
    if _STEP_ACTIONS is None:
        try:
            from task_planner import STEP_SCHEMA
            _STEP_ACTIONS = set(STEP_SCHEMA.keys())
        except Exception:
            _STEP_ACTIONS = set()
    return _STEP_ACTIONS


def _validate_skill_dict(data: dict[str, Any], source: str = "") -> None:
    """Raise :class:`SkillLoadError` if *data* is not a valid skill definition."""
    missing = _REQUIRED_SKILL_KEYS - data.keys()
    if missing:
        raise SkillLoadError(
            f"Skill{' from ' + source if source else ''} is missing required keys: {missing}"
        )
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise SkillLoadError("Skill 'name' must be a non-empty string")
    if not isinstance(data.get("steps"), list) or not data["steps"]:
        raise SkillLoadError(f"Skill '{data['name']}' must have a non-empty 'steps' list")

    allowed = _get_allowed_actions()
    if allowed:
        for i, step in enumerate(data["steps"]):
            if not isinstance(step, dict):
                raise SkillLoadError(f"Skill '{data['name']}': step {i} must be a dict")
            action = step.get("action")
            if not action:
                raise SkillLoadError(f"Skill '{data['name']}': step {i} is missing 'action'")
            if action not in allowed:
                raise SkillLoadError(
                    f"Skill '{data['name']}': step {i} has unknown action {action!r}. "
                    f"Allowed: {sorted(allowed)}"
                )


def _dict_to_skill(data: dict[str, Any], source: str = "") -> SkillDef:
    _validate_skill_dict(data, source)
    return SkillDef(
        name        = str(data["name"]).strip(),
        description = str(data.get("description", "")),
        version     = str(data.get("version", "1.0.0")),
        author      = str(data.get("author", "")),
        triggers    = [str(t) for t in data.get("triggers", [])],
        parameters  = dict(data.get("parameters", {})),
        steps       = list(data["steps"]),
        source      = source,
    )


# ---------------------------------------------------------------------------
# File / text parsing
# ---------------------------------------------------------------------------

def _parse_text(text: str, source: str = "") -> dict[str, Any] | list[dict[str, Any]]:
    """Parse JSON (always) or YAML (if PyYAML is available)."""
    # Try JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try YAML
    try:
        import yaml  # type: ignore[import]
        return yaml.safe_load(text)  # type: ignore[return-value]
    except ImportError:
        raise SkillLoadError(
            f"Cannot parse {source!r} as JSON. "
            "Install PyYAML to enable YAML support:  pip install pyyaml"
        )
    except Exception as exc:
        raise SkillLoadError(f"Failed to parse {source!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Public loading helpers
# ---------------------------------------------------------------------------

def load_from_file(path: str | Path) -> list[SkillDef]:
    """
    Load one or more skills from a local JSON or YAML file.

    The file may contain a single skill object ``{...}`` or a list of skill
    objects ``[{...}, {...}]``.

    Parameters
    ----------
    path:
        Path to a ``.json`` or ``.yaml`` / ``.yml`` skill file.

    Returns
    -------
    list[SkillDef]
        One or more loaded skill definitions.

    Raises
    ------
    SkillLoadError
        If the file cannot be read or fails validation.
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path).resolve()
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise SkillLoadError(f"Cannot read {path}: {exc}") from exc

    data = _parse_text(text, str(path))
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise SkillLoadError(f"{path}: expected a skill object or list of skill objects")

    return [_dict_to_skill(item, str(path)) for item in data]


def load_from_directory(directory: str | Path) -> list[SkillDef]:
    """
    Load all skill files found in *directory* (non-recursive).

    Recognises files ending in ``.json``, ``.yaml``, and ``.yml``.

    Parameters
    ----------
    directory:
        Path to a local directory containing skill files.

    Returns
    -------
    list[SkillDef]
        All successfully loaded skills (files with errors are skipped with a
        warning printed to stderr).
    """
    import sys

    d = Path(directory).resolve()
    if not d.is_dir():
        raise SkillLoadError(f"Skills directory does not exist: {directory}")

    skills: list[SkillDef] = []
    for p in sorted(d.glob("*")):
        # Use relative_to() to confirm p is genuinely inside d
        # (guards against symlink escape or .glob() quirks)
        try:
            p.relative_to(d)
        except ValueError:
            continue
        if p.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        try:
            skills.extend(load_from_file(p))
        except (SkillLoadError, FileNotFoundError, OSError) as exc:
            print(f"[skills] Warning: skipping {p.name} — {exc}", file=sys.stderr)
    return skills


def load_from_url(url: str, *, timeout: int = 15) -> list[SkillDef]:
    """
    Fetch and load a skill definition from an HTTP/HTTPS URL.

    Parameters
    ----------
    url:
        Direct URL to a JSON (or YAML) skill file, e.g.
        ``"https://raw.githubusercontent.com/owner/repo/main/skills/login.json"``
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    list[SkillDef]
    """
    _validate_url_scheme(url)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "agenticbrowser-skills/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
    except Exception as exc:
        raise SkillLoadError(f"Failed to fetch skill from {url!r}: {exc}") from exc

    data = _parse_text(text, url)
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise SkillLoadError(f"{url}: expected a skill object or list of skill objects")

    return [_dict_to_skill(item, url) for item in data]


def load_from_github(
    repo: str,
    path: str = "skills/index.json",
    *,
    branch: str = "main",
    timeout: int = 15,
) -> list[SkillDef]:
    """
    Load skills from a GitHub repository using the raw content API.

    Parameters
    ----------
    repo:
        Repository in ``"owner/repo"`` format, e.g. ``"openclaw/skills"``.
        Also accepts the ``"gh:owner/repo"`` shorthand.
    path:
        Path inside the repository to a skill file or a directory.
        When *path* ends with ``.json`` or ``.yaml`` / ``.yml`` a single
        file is fetched.  Otherwise the path is treated as a directory
        and the loader fetches ``{path}/index.json`` (an index file that
        lists the skill file names to load).
    branch:
        Git branch/tag to fetch from (default: ``"main"``).
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    list[SkillDef]

    Examples
    --------
    Load a single skill::

        load_from_github("openclaw/skills", "browser/google_login.json")

    Load all skills listed in an index::

        load_from_github("openclaw/skills")  # fetches skills/index.json

    Index file format (``index.json``)::

        {"skills": ["google_login.json", "github_search.json"]}
    """
    # Normalise "gh:owner/repo" → "owner/repo"
    repo = repo.removeprefix("gh:")

    path = path.rstrip("/")
    _validate_github_ref(repo, path, branch)

    def _raw_url(file_path: str) -> str:
        # Host is hardcoded to raw.githubusercontent.com and verified by
        # _make_github_raw_url — not user-controlled
        return _make_github_raw_url(repo, branch, file_path)
    is_file = any(path.endswith(ext) for ext in (".json", ".yaml", ".yml"))

    if is_file:
        return load_from_url(_raw_url(path), timeout=timeout)

    # Treat as directory — fetch index.json
    index_url = _raw_url(f"{path}/index.json")
    try:
        req = urllib.request.Request(
            index_url,
            headers={"User-Agent": "agenticbrowser-skills/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            index = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise SkillLoadError(
            f"Failed to fetch skill index from {index_url!r}: {exc}. "
            "Ensure the repo has an 'index.json' listing skill file names."
        ) from exc

    file_names: list[str] = index.get("skills", [])
    if not file_names:
        raise SkillLoadError(f"index.json from {repo!r} contains no 'skills' entries")

    skills: list[SkillDef] = []
    for fname in file_names:
        file_url = _raw_url(f"{path}/{fname}")
        try:
            skills.extend(load_from_url(file_url, timeout=timeout))
        except SkillLoadError as exc:
            import sys
            print(f"[skills] Warning: skipping {fname} from {repo} — {exc}", file=sys.stderr)

    return skills


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """
    A mutable collection of :class:`SkillDef` objects.

    Provides intent-matching so the TaskPlanner can check whether an intent
    matches a loaded skill before falling back to built-in templates or LLM.

    Usage::

        registry = SkillRegistry()
        registry.load_from_file("skills/github_search.json")
        registry.load_from_github("openclaw/skills")

        match = registry.match(intent)
        if match:
            skill, params = match
            steps = skill.resolve_steps(params)
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDef] = {}  # name → SkillDef

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def register(self, skill: SkillDef) -> None:
        """Add a skill to the registry (overwrites any existing skill with the same name)."""
        self._skills[skill.name] = skill

    def register_many(self, skills: list[SkillDef]) -> None:
        for s in skills:
            self.register(s)

    def unregister(self, name: str) -> bool:
        """Remove a skill by name.  Returns ``True`` if it existed."""
        return self._skills.pop(name, None) is not None

    def clear(self) -> None:
        self._skills.clear()

    # ------------------------------------------------------------------
    # Bulk loading — delegate to module-level helpers
    # ------------------------------------------------------------------

    def load_from_file(self, path: str | Path) -> list[SkillDef]:
        """Load skills from a local file and register them.  Returns loaded skills."""
        skills = load_from_file(path)
        self.register_many(skills)
        return skills

    def load_from_directory(self, directory: str | Path) -> list[SkillDef]:
        """Load all skills from a directory and register them."""
        skills = load_from_directory(directory)
        self.register_many(skills)
        return skills

    def load_from_url(self, url: str, *, timeout: int = 15) -> list[SkillDef]:
        """Fetch skills from a URL and register them."""
        skills = load_from_url(url, timeout=timeout)
        self.register_many(skills)
        return skills

    def load_from_github(
        self,
        repo: str,
        path: str = "skills/index.json",
        *,
        branch: str = "main",
        timeout: int = 15,
    ) -> list[SkillDef]:
        """Load skills from a GitHub repository and register them."""
        skills = load_from_github(repo, path, branch=branch, timeout=timeout)
        self.register_many(skills)
        return skills

    def load_from_source(self, source: str) -> list[SkillDef]:
        """
        Auto-detect and load skills from *source* (convenience method).

        Dispatch rules:

        * ``"gh:owner/repo[/path]"``  → :meth:`load_from_github`
        * ``"https?://..."``           → :meth:`load_from_url`
        * Existing local directory     → :meth:`load_from_directory`
        * Existing local file          → :meth:`load_from_file`

        Parameters
        ----------
        source:
            A file path, directory path, URL, or ``gh:owner/repo`` reference.
        """
        s = source.strip()
        if s.startswith("gh:") or (
            re.match(r"^[\w-]+/[\w.-]+", s) and not Path(s).exists()
        ):
            return self.load_from_github(s)
        if s.startswith("http://") or s.startswith("https://"):
            _validate_url_scheme(s)
            return self.load_from_url(s)
        p = Path(s).resolve()
        if p.is_dir():
            return self.load_from_directory(p)
        return self.load_from_file(p)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> SkillDef | None:
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDef]:
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)

    def match(self, intent: str) -> tuple[SkillDef, dict[str, str]] | None:
        """
        Check whether *intent* matches any loaded skill.

        Returns
        -------
        ``(SkillDef, params_dict)`` if a skill matches, otherwise ``None``.
        The ``params_dict`` maps parameter names to the values extracted from
        the intent string via trigger placeholders.
        """
        clean = intent.strip()
        for skill in self._skills.values():
            for pattern, param_names in skill._compiled:
                m = pattern.fullmatch(clean)
                if m:
                    params = dict(zip(param_names, m.groups()))
                    return skill, params
        return None


# ---------------------------------------------------------------------------
# SkillDef helper method (defined here to avoid circular import with registry)
# ---------------------------------------------------------------------------

def resolve_steps(skill: SkillDef, params: dict[str, str]) -> list[dict[str, Any]]:
    """
    Return the skill's step list with parameter placeholders substituted.

    Parameters
    ----------
    skill:
        The matched skill definition.
    params:
        Parameter values extracted from the intent string.
    """
    return _substitute_params(skill.steps, params)


# Attach as a method for convenience
SkillDef.resolve_steps = lambda self, params: resolve_steps(self, params)  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Module-level default registry (used by TaskPlanner if no registry provided)
# ---------------------------------------------------------------------------

_default_registry: SkillRegistry = SkillRegistry()


def get_default_registry() -> SkillRegistry:
    """Return the process-level default :class:`SkillRegistry`."""
    return _default_registry
