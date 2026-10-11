"""Structural tests for the GitLab CI configuration.

A broken ``.gitlab-ci.yml`` only fails after pushing, and a pipeline that never
runs is worse than one that fails loudly. These tests catch the mistakes that
are otherwise invisible until then:

  * a script step that YAML parsed into a mapping (the classic unquoted colon)
  * a ``!reference`` pointing at a hidden job that does not exist
  * a job the config never schedules because its stage is undefined
  * the Arch Linux image being dropped, and the build silently running on a
    distribution the package is not built for
  * makepkg being invoked as root, which it refuses to do

Requires ``pyyaml``; skipped when it is unavailable so the suite still runs on
a bare checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import REPO_ROOT

yaml = pytest.importorskip("yaml")

CI_FILE = REPO_ROOT / ".gitlab-ci.yml"
SCRIPT_KEYS = ("before_script", "script", "after_script")

# GitLab's `!reference` is a custom tag; register a no-op constructor so the
# file can be parsed the same way the GitLab linter does.
class _GitLabLoader(yaml.SafeLoader):
    pass


_GitLabLoader.add_constructor(
    "!reference", lambda loader, node: loader.construct_sequence(node)
)


@pytest.fixture(scope="module")
def ci() -> dict:
    if not CI_FILE.is_file():
        pytest.skip("no .gitlab-ci.yml in this repository")
    return yaml.load(CI_FILE.read_text(encoding="utf-8"), Loader=_GitLabLoader)


def _jobs(ci: dict) -> dict[str, dict]:
    """Return the scheduled jobs.

    GitLab treats a leading dot as a hidden template job: it is never run, so it
    legitimately has no stage and produces no artifacts.
    """
    return {
        name: cfg
        for name, cfg in ci.items()
        if name not in ("stages", "default", "variables", "include", "workflow")
        and not name.startswith(".")
        and isinstance(cfg, dict)
    }


def _all_steps(ci: dict):
    """Yield (job, key, index, step) for every script step in the config."""
    for job, cfg in _jobs(ci).items():
        for key in SCRIPT_KEYS:
            for i, step in enumerate(cfg.get(key) or []):
                yield job, key, i, step


def test_ci_file_is_well_formed(ci: dict) -> None:
    """The file must parse and declare stages."""
    assert "stages" in ci, ".gitlab-ci.yml declares no stages"
    assert ci["stages"], "stages list is empty"


def test_uses_an_arch_linux_image(ci: dict) -> None:
    """The whole point: build on Arch, not an approximation of it."""
    image = (ci.get("default") or {}).get("image")
    assert image, "default: image is not set"
    assert "archlinux" in image, (
        f"CI must run on an Arch Linux image, found: {image}"
    )


def test_every_job_uses_an_arch_image(ci: dict) -> None:
    """A job overriding the image must still use Arch."""
    for job, cfg in _jobs(ci).items():
        image = cfg.get("image")
        if image is None:
            continue
        assert "archlinux" in image, (
            f"job '{job}' runs on a non-Arch image: {image}"
        )


def test_no_script_step_parsed_into_a_mapping(ci: dict) -> None:
    """An unquoted colon turns a shell line into a YAML mapping.

    GitLab then receives a dict where it expects a string and rejects the job.
    """
    offenders = [
        (job, key, i, step)
        for job, key, i, step in _all_steps(ci)
        if isinstance(step, dict)
    ]
    assert not offenders, (
        "these script steps parsed as YAML mappings rather than strings "
        f"(quote them): {offenders}"
    )


def test_job_stages_are_declared(ci: dict) -> None:
    """A job in an undeclared stage is never scheduled."""
    stages = set(ci["stages"])
    for job, cfg in _jobs(ci).items():
        stage = cfg.get("stage")
        assert stage is not None, f"job '{job}' has no stage"
        assert stage in stages, f"job '{job}' uses undeclared stage '{stage}'"


def test_references_resolve_to_real_hidden_jobs(ci: dict) -> None:
    """``!reference`` must name a job that exists, or the pipeline is invalid."""
    for job, key, i, step in _all_steps(ci):
        if not isinstance(step, list) or len(step) != 2:
            continue
        target, target_key = step
        assert target in ci, (
            f"job '{job}' references undefined job '{target}'"
        )
        assert target_key in ci[target], (
            f"job '{job}' references '{target}.{target_key}', which is not defined"
        )


def test_makepkg_is_not_run_as_root(ci: dict) -> None:
    """makepkg refuses to run as root; an Arch container starts as root.

    Any job invoking makepkg must therefore drop privileges first.
    """
    for job, cfg in _jobs(ci).items():
        text = "\n".join(
            str(step) for key in SCRIPT_KEYS for step in (cfg.get(key) or [])
        )
        # Match a real invocation, not a passing mention in a comment or an
        # echo; those would otherwise produce false positives.
        invokes_makepkg = any(
            line.strip().startswith(("makepkg", "sudo -u builder", "bash -euo"))
            and "makepkg" in line
            for line in text.splitlines()
        )
        if not invokes_makepkg:
            continue
        assert "sudo -u builder" in text or "useradd" in text, (
            f"job '{job}' calls makepkg without dropping root privileges"
        )


def test_build_jobs_produce_artifacts(ci: dict) -> None:
    """A build job that discards its package is not useful."""
    build_jobs = [n for n in _jobs(ci) if "build" in n]
    assert build_jobs, "no build job found"
    for job in build_jobs:
        assert "artifacts" in ci[job], (
            f"build job '{job}' does not publish artifacts"
        )
        paths = ci[job]["artifacts"].get("paths") or []
        assert any("pkg.tar.zst" in p for p in paths), (
            f"build job '{job}' does not archive the built package"
        )


def test_heavy_jobs_are_bounded(ci: dict) -> None:
    """A CUDA build can run for hours; an unbounded job is a stuck runner."""
    for job in ("build-package", "cuda-toolkit"):
        if job not in ci:
            continue
        assert "timeout" in ci[job], (
            f"job '{job}' has no timeout; a hung build would occupy a runner"
        )
