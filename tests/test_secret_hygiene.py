"""Secret-hygiene guarantees for the packaging repos.

The .env spec is strict: real credentials must never reach the public remote,
and any file a tool reads for credentials must be ignored *before* it exists in
the working tree. These tests encode that contract so a future edit cannot
quietly reintroduce a leak.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conftest import REPO_ROOT, git_tracked_files

SECRET_FILES = (".env",)


def _git_check_ignore(repo: Path, path: str) -> bool:
    """True when git would ignore ``path`` inside ``repo``."""
    result = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "--quiet", path],
        capture_output=True,
    )
    # 0 == ignored, 1 == not ignored, other == error.
    return result.returncode == 0


@pytest.mark.parametrize("secret", SECRET_FILES)
def test_env_file_is_ignored(secret: str) -> None:
    """.env must be ignored, or the token gets committed on the next `git add -A`."""
    assert _git_check_ignore(REPO_ROOT, secret), (
        f"{secret} is not ignored; a future 'git add -A' would publish the token"
    )


@pytest.mark.parametrize("secret", SECRET_FILES)
def test_env_file_is_never_tracked(secret: str) -> None:
    """The secret file must not be tracked, even if it was ignored later."""
    assert secret not in git_tracked_files(REPO_ROOT), (
        f"{secret} is tracked by git; the credential is already exposed"
    )


def test_env_example_exists_and_is_tracked() -> None:
    """A redacted template must exist and be committed for contributors."""
    example = REPO_ROOT / ".env.example"
    assert example.is_file(), ".env.example is missing"
    assert ".env.example" in git_tracked_files(REPO_ROOT), (
        ".env.example exists but is not tracked; it will not reach the remote"
    )


def test_env_example_contains_no_secret_values() -> None:
    """The template must not leak any real credential value."""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    # A GitHub token of any shape must never appear in the template.
    for marker in ("github_pat_", "ghp_", "gho_", "ghs_", "ghu_"):
        assert marker not in text, f".env.example leaks a token ({marker})"


def test_env_example_documents_required_keys() -> None:
    """The keys the tooling reads must be documented, with empty values."""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GITHUB_TOKEN=" in text, "GITHUB_TOKEN is not documented"
    for line in text.splitlines():
        if line.startswith("GITHUB_TOKEN="):
            assert line.strip() == "GITHUB_TOKEN=", (
                "GITHUB_TOKEN must be present but empty in the template"
            )


def test_gitattributes_export_ignores_secrets() -> None:
    """Defense in depth: secrets must be excluded from archives."""
    path = REPO_ROOT / ".gitattributes"
    assert path.is_file(), ".gitattributes is missing"
    text = path.read_text(encoding="utf-8")
    assert ".env export-ignore" in text, (
        ".env must be export-ignored so `git archive` cannot leak it"
    )


def test_makepkg_working_trees_are_ignored() -> None:
    """src/ and the bare clone are huge downstream trees, never committed."""
    for path in ("src", "pkg", "ollama"):
        assert _git_check_ignore(REPO_ROOT, path), (
            f"{path} is not ignored; a 'git add -A' would commit a full "
            "upstream checkout"
        )
