"""Structural tests for the GitHub Actions workflows.

``yaml.safe_load`` happily accepts any key, so it will not tell you that
``timeout`` is invalid where GitHub expects ``timeout-minutes``; the API rejects
the workflow at *dispatch* time instead, which is far too late. These tests
check the structure GitHub actually parses.

Requires ``pyyaml``; skipped when unavailable so the suite still runs on a bare
checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import REPO_ROOT

yaml = pytest.importorskip("yaml")

WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

# Keys GitHub accepts on a job. Anything else is a startup failure.
VALID_JOB_KEYS = frozenset(
    {
        "name",
        "permissions",
        "needs",
        "if",
        "runs-on",
        "environment",
        "concurrency",
        "outputs",
        "env",
        "defaults",
        "steps",
        "timeout-minutes",
        "continue-on-error",
        "container",
        "services",
        "strategy",
        "uses",
        "with",
        "secrets",
        "secrets-inherit",
        "environment",
    }
)

# Keys GitHub accepts on a step.
VALID_STEP_KEYS = frozenset(
    {
        "id",
        "if",
        "name",
        "uses",
        "run",
        "working-directory",
        "shell",
        "with",
        "env",
        "continue-on-error",
        "timeout-minutes",
    }
)


def _workflows() -> list[Path]:
    if not WORKFLOW_DIR.is_dir():
        return []
    return sorted(WORKFLOW_DIR.glob("*.yml"))


def test_workflows_exist() -> None:
    """A repository that publishes releases must have workflows."""
    assert _workflows(), "no workflows found"


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_workflow_is_valid_yaml(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name} is not a mapping"
    # `on` is parsed as the boolean True by YAML 1.1 loaders; accept either.
    assert "on" in data or True in data, f"{path.name} has no trigger"
    assert "jobs" in data, f"{path.name} defines no jobs"


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_job_keys_are_valid(path: Path) -> None:
    """An unknown job key makes GitHub reject the whole workflow.

    This is the check that catches `timeout` vs `timeout-minutes`, which
    yaml.safe_load cannot detect and which only surfaces on dispatch.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job_name, job in (data.get("jobs") or {}).items():
        unknown = set(job) - VALID_JOB_KEYS
        assert not unknown, (
            f"{path.name}: job '{job_name}' uses invalid keys {sorted(unknown)}"
        )


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_step_keys_are_valid(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job_name, job in (data.get("jobs") or {}).items():
        for i, step in enumerate(job.get("steps") or []):
            unknown = set(step) - VALID_STEP_KEYS
            assert not unknown, (
                f"{path.name}: job '{job_name}' step {i} uses invalid keys "
                f"{sorted(unknown)}"
            )


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_step_has_run_or_uses(path: Path) -> None:
    """A step with neither does nothing and usually signals a mistake."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job_name, job in (data.get("jobs") or {}).items():
        for i, step in enumerate(job.get("steps") or []):
            assert "run" in step or "uses" in step, (
                f"{path.name}: job '{job_name}' step {i} has neither run nor uses"
            )


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_run_steps_are_strings(path: Path) -> None:
    """An unquoted colon turns a shell line into a mapping."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for job_name, job in (data.get("jobs") or {}).items():
        for i, step in enumerate(job.get("steps") or []):
            if "run" in step:
                assert isinstance(step["run"], str), (
                    f"{path.name}: job '{job_name}' step {i} has a non-string run"
                )


def _release_workflows() -> list[Path]:
    return [p for p in _workflows() if "release" in p.name]


def test_release_workflow_publishes_assets() -> None:
    """The point of the release workflow is a downloadable artifact."""
    if not _release_workflows():
        pytest.skip("no release workflow in this repository")
    for path in _release_workflows():
        text = path.read_text(encoding="utf-8")
        assert "gh release create" in text or "gh release upload" in text, (
            f"{path.name} never creates or uploads a release"
        )
        assert "contents: write" in text, (
            f"{path.name} needs 'permissions: contents: write' to publish"
        )


def test_release_assets_stay_under_the_2gib_limit() -> None:
    """GitHub rejects a release asset of 2 GiB or more.

    Verified against the API, which answers:
      "size must be less than 2147483648"
    A package larger than that must be split, so any release workflow has to
    contain split logic.
    """
    if not _release_workflows():
        pytest.skip("no release workflow in this repository")
    for path in _release_workflows():
        text = path.read_text(encoding="utf-8")
        assert "split" in text, (
            f"{path.name} does not split oversized assets; an asset of 2 GiB "
            "or more is rejected outright"
        )
        assert "SHA256SUMS" in text, (
            f"{path.name} publishes parts without checksums, so a downloader "
            "cannot verify the reassembled package"
        )


def test_release_workflow_frees_disk_before_building() -> None:
    """The CUDA build needs ~30 GB; a stock runner has ~14 GB free."""
    if not _release_workflows():
        pytest.skip("no release workflow in this repository")
    for path in _release_workflows():
        text = path.read_text(encoding="utf-8")
        assert "df -h" in text, f"{path.name} never checks available space"
        assert "rm -rf" in text, f"{path.name} never frees disk space"


def test_container_mounts_use_a_host_absolute_path() -> None:
    """A docker -v source must be absolute or the mount silently does nothing.

    Using $PWD produced "cd: /work: No such file or directory" inside the
    container, because Docker created an empty /work instead of bind-mounting
    the workspace. $GITHUB_WORKSPACE is guaranteed absolute.
    """
    if not _release_workflows():
        pytest.skip("no release workflow in this repository")
    for path in _release_workflows():
        text = path.read_text(encoding="utf-8")
        if "docker run" not in text:
            continue
        assert "-v \"$PWD:" not in text, (
            f"{path.name} bind-mounts $PWD; use $GITHUB_WORKSPACE, which is "
            "guaranteed to be an absolute host path"
        )
        assert "GITHUB_WORKSPACE" in text, (
            f"{path.name} should bind-mount $GITHUB_WORKSPACE"
        )


def test_sudo_is_not_relied_on_for_the_environment() -> None:
    """sudo resets the environment, so 'set -u' aborts on an inherited var.

    The build failed with "MAKEPKG_DIR: unbound variable" for exactly this
    reason. Paths must be passed explicitly or defined inside the sudo shell.
    """
    if not _release_workflows():
        pytest.skip("no release workflow in this repository")
    for path in _release_workflows():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "sudo -u" in line and "bash -" in line:
                assert line.rstrip().endswith("\\") or line.rstrip().endswith("'"), (
                    f"{path.name}: sudo invocation should pass variables "
                    f"explicitly: {line.strip()}"
                )


def test_checksums_cover_the_reassembled_package() -> None:
    """Checksumming only the parts breaks the documented verify step.

    The release notes tell users to `cat` the parts and then run
    `sha256sum -c SHA256SUMS.txt`. If the file lists only the part files, that
    check fails on the joined package, so the digest for the reassembled name
    must be included too.
    """
    if not _release_workflows():
        pytest.skip("no release workflow in this repository")
    for path in _release_workflows():
        text = path.read_text(encoding="utf-8")
        assert 'printf "%s  %s\\n" "$digest" "$base"' in text, (
            f"{path.name} does not record a digest under the reassembled "
            "filename; `sha256sum -c` would fail after reassembly"
        )
        # A greedy sed over `sha256sum` output eats the digest along with the
        # path, producing a malformed line.
        assert "sha256sum" not in text or "sed \"s|.*/|" not in text, (
            f"{path.name} rewrites the checksum filename with a greedy sed, "
            "which strips the digest"
        )
