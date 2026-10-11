"""Tests for the CUDA 11.8 ("cuda_v11") backend wiring.

Upstream ollama ships cuda_v12 and cuda_v13 backends only. CUDA 11.8 is needed
for hosts pinned to the legacy 520.x driver branch, but the CUDA 11 toolchain
cannot compile the arch list the 12/13 presets use (sm_90a/sm_100/sm_120 are
CUDA 12+). These tests pin down the three places a backend must be registered,
so the backend cannot be half-added.

The backend name matters: cmake/local.cmake derives the preset name from it via

    set(_preset "llama_${backend}_linux")

so "cuda_v11" must have a matching "llama_cuda_v11_linux" preset.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from conftest import (
    REPO_ROOT,
    pkgbuild_array,
    pkgbuild_scalar,
    read_pkgbuild,
)

PKGBUILD = read_pkgbuild()
UPSTREAM = REPO_ROOT / "src" / "ollama"
PRESETS_PATH = UPSTREAM / "llama" / "server" / "CMakePresets.json"
LOCAL_CMAKE = UPSTREAM / "cmake" / "local.cmake"
DEVELOPMENT_DOC = UPSTREAM / "docs" / "development.md"

BACKEND = "cuda_v11"

# Architectures the CUDA 11.8 toolchain can actually compile. sm_90a, sm_100,
# sm_103, sm_110, sm_120 and sm_121 all require CUDA 12 or newer.
CUDA_11_MAX_ARCH = 90


def _presets() -> dict:
    return json.loads(PRESETS_PATH.read_text(encoding="utf-8"))


def _upstream_available() -> bool:
    return PRESETS_PATH.is_file()


@pytest.fixture(scope="session", autouse=True)
def _ensure_upstream() -> None:
    """Fetch and patch upstream when the checkout is absent.

    Locally the makepkg tree is already present. In CI it is not, and skipping
    would mean the tests covering the backend wiring never run there - which
    defeats the purpose. Fetching is cheap next to a CUDA build, and the
    checkout is cached for the session.
    """
    if _upstream_available():
        return

    repo = "https://github.com/ollama/ollama.git"
    patch = REPO_ROOT / "0001-add-cuda-v11-backend.patch"
    UPSTREAM.parent.mkdir(parents=True, exist_ok=True)

    clone = subprocess.run(
        ["git", "clone", "--depth", "1", repo, str(UPSTREAM)],
        capture_output=True,
        text=True,
    )
    if clone.returncode != 0:
        pytest.skip(f"could not fetch upstream for verification: {clone.stderr.strip()[:200]}")

    # Apply the tracked patch, mirroring what the PKGBUILD does in prepare().
    applied = subprocess.run(
        ["patch", "-Np1", "-i", str(patch)],
        cwd=UPSTREAM,
        capture_output=True,
        text=True,
    )
    if applied.returncode != 0:
        pytest.fail(
            "the tracked backend patch does not apply to upstream HEAD:\n"
            + applied.stdout
            + applied.stderr
        )


pytestmark = pytest.mark.skipif(
    False,
    reason="upstream is fetched by the _ensure_upstream fixture",
)


def test_upstream_checkout_is_present() -> None:
    """The packaging expects a vendored upstream tree next to the PKGBUILD."""
    assert UPSTREAM.is_dir()


def test_presets_json_is_valid() -> None:
    """A malformed preset file breaks every configure invocation."""
    data = _presets()
    assert data["version"] == 3
    assert isinstance(data["configurePresets"], list)


def test_configure_preset_exists_for_the_backend() -> None:
    """cmake derives llama_<backend>_linux; it must exist or configure fails."""
    names = {p["name"] for p in _presets()["configurePresets"]}
    assert f"llama_{BACKEND}_linux" in names


def test_user_arch_preset_exists_for_the_backend() -> None:
    """Selecting CMAKE_CUDA_ARCHITECTURES switches to the _user_arch preset."""
    names = {p["name"] for p in _presets()["configurePresets"]}
    assert f"llama_{BACKEND}_user_arch" in names


def test_base_preset_is_hidden_and_enables_cuda() -> None:
    """The shared base must be hidden and actually turn CUDA on."""
    by_name = {p["name"]: p for p in _presets()["configurePresets"]}
    base = by_name[f"llama_{BACKEND}_base"]
    assert base.get("hidden") is True
    cache = base["cacheVariables"]
    assert cache["GGML_CUDA"] == "ON"
    assert cache["OLLAMA_GPU_BACKEND"] == "cuda"
    assert cache["OLLAMA_RUNNER_DIR"] == BACKEND


def test_linux_preset_inherits_base_and_has_architectures() -> None:
    """The linux preset must inherit the base and pin its arch list."""
    by_name = {p["name"]: p for p in _presets()["configurePresets"]}
    linux = by_name[f"llama_{BACKEND}_linux"]
    assert f"llama_{BACKEND}_base" in linux["inherits"]
    assert "CMAKE_CUDA_ARCHITECTURES" in linux["cacheVariables"]


def test_architecture_list_is_compilable_by_cuda_11() -> None:
    """CUDA 11.8 cannot compile sm_90a/sm_100/sm_120; they must not appear."""
    by_name = {p["name"]: p for p in _presets()["configurePresets"]}
    archs = by_name[f"llama_{BACKEND}_linux"]["cacheVariables"][
        "CMAKE_CUDA_ARCHITECTURES"
    ].split(";")

    assert archs, "the architecture list is empty"

    for arch in archs:
        if arch.endswith("-virtual"):
            continue
        # Accept "90a" style suffixes so the assertion explains itself.
        match = re.fullmatch(r"(\d+)([a-z]*)", arch)
        assert match, f"unparseable architecture entry: {arch}"
        number, suffix = int(match.group(1)), match.group(2)
        assert number <= CUDA_11_MAX_ARCH, (
            f"sm_{arch} needs CUDA 12+ but this backend targets CUDA 11.8"
        )
        assert not suffix, (
            f"sm_{arch} is a CUDA 12+ accelerated target and is unavailable "
            "from CUDA 11.8"
        )


def test_build_preset_exists_for_the_backend() -> None:
    """Without a build preset the backend cannot be built by preset name."""
    names = {p["name"] for p in _presets()["buildPresets"]}
    assert f"llama_{BACKEND}_linux" in names


def test_local_cmake_dispatches_the_backend() -> None:
    """local.cmake must recognise the backend or configure aborts."""
    text = LOCAL_CMAKE.read_text(encoding="utf-8")
    assert re.search(
        rf'if\s*\(\s*_backend\s+STREQUAL\s+"{BACKEND}"\s*\)', text
    ), f"cmake/local.cmake has no dispatch branch for {BACKEND}"


def test_local_cmake_builds_ggml_cuda_for_the_backend() -> None:
    """The backend branch must request the ggml-cuda target."""
    text = LOCAL_CMAKE.read_text(encoding="utf-8")
    start = text.find(f'if(_backend STREQUAL "{BACKEND}")')
    assert start != -1
    end = text.find("elseif", start)
    branch = text[start : end if end != -1 else len(text)]
    assert "ggml-cuda" in branch
    assert "ollama_add_llama_server_build" in branch
    assert f"RUNNER_DIR ${{_backend}}" in branch


def test_development_docs_list_the_backend() -> None:
    """Undocumented backends are effectively unusable for contributors."""
    text = DEVELOPMENT_DOC.read_text(encoding="utf-8")
    assert f"`{BACKEND}`" in text, (
        f"{BACKEND} is not listed in the supported backend values"
    )


def test_pkgbuild_selects_the_backend() -> None:
    """The package must actually request the CUDA 11.8 backend."""
    text = PKGBUILD
    assert "OLLAMA_LLAMA_BACKENDS" in text, (
        "the PKGBUILD does not select any GPU backend"
    )
    assert BACKEND in text, f"the PKGBUILD does not select {BACKEND}"


def test_pkgbuild_pins_the_cuda_toolkit_explicitly() -> None:
    """Relying on PATH can silently pick a newer, incompatible CUDA."""
    assert "CUDAToolkit_ROOT" in PKGBUILD, (
        "the PKGBUILD must pin CUDAToolkit_ROOT so nvcc is deterministic"
    )
    assert "CMAKE_CUDA_COMPILER" in PKGBUILD, (
        "the PKGBUILD must pin CMAKE_CUDA_COMPILER"
    )


def test_pkgbuild_does_not_use_the_removed_component_names() -> None:
    """--component CPU/CUDA no longer exists in the rewritten superbuild."""
    for stale in ("--component CPU", "--component CUDA"):
        assert stale not in PKGBUILD, (
            f"{stale} was removed upstream; the install step would fail"
        )


def test_pkgbuild_installs_the_current_components() -> None:
    """Install must target the components the superbuild actually defines."""
    assert "--component ollama-local" in PKGBUILD
    assert "--component llama-server" in PKGBUILD


def test_srcinfo_matches_pkgbuild_version() -> None:
    """AUR metadata must not drift from the PKGBUILD."""
    srcinfo = (REPO_ROOT / ".SRCINFO").read_text(encoding="utf-8")
    pkgver = pkgbuild_scalar(PKGBUILD, "pkgver")
    assert pkgver is not None
    assert f"pkgver = {pkgver}" in srcinfo


def test_srcinfo_lists_the_cuda_11_8_toolchain_dependency() -> None:
    """The build needs the side-by-side CUDA 11.8 toolkit, not `cuda`."""
    srcinfo = (REPO_ROOT / ".SRCINFO").read_text(encoding="utf-8")
    assert "makedepends = cuda11.8" in srcinfo, (
        "the package must build against cuda11.8, not a generic cuda"
    )


def test_go_binary_is_still_installed() -> None:
    """The Go binary is the actual executable the package ships."""
    assert "install -Dm755" in PKGBUILD
    assert "/usr/bin/$_pkgname" in PKGBUILD


def test_cuda_toolkit_args_are_published_to_the_caller() -> None:
    """Regression: ollama_append_cuda_toolkit_args must reach its caller.

    It used to delegate to ollama_append_cache_arg_if_set(), which writes via
    PARENT_SCOPE. That write landed in the wrapper's own scope and was
    discarded, so CUDAToolkit_ROOT silently never reached the nested build and
    nvcc was resolved from PATH instead of the pinned toolkit.
    """
    text = LOCAL_CMAKE.read_text(encoding="utf-8")
    start = text.find("function(ollama_append_cuda_toolkit_args")
    assert start != -1
    end = text.find("endfunction()", start)
    body = text[start:end]

    assert "PARENT_SCOPE" in body, (
        "the wrapper must publish its result to the caller"
    )
    assert "set(${output} ${_cuda_toolkit_args} PARENT_SCOPE)" in body, (
        "the wrapper must set ${output} in PARENT_SCOPE exactly once, from a "
        "local accumulator"
    )
    # Calling the PARENT_SCOPE helper directly with ${output} reintroduces the bug.
    assert "ollama_append_cache_arg_if_set(${output} CUDAToolkit_ROOT)" not in body, (
        "delegating to ollama_append_cache_arg_if_set(${output} ...) loses the "
        "value, because its PARENT_SCOPE write lands in this function's scope"
    )


def test_cuda_compiler_is_forwarded_to_nested_builds() -> None:
    """CMAKE_CUDA_COMPILER must be forwarded, or nvcc comes from PATH."""
    text = LOCAL_CMAKE.read_text(encoding="utf-8")
    start = text.find("function(ollama_append_cuda_toolkit_args")
    end = text.find("endfunction()", start)
    body = text[start:end]
    assert "CMAKE_CUDA_COMPILER" in body, (
        "CMAKE_CUDA_COMPILER is not forwarded to the nested CUDA build"
    )


def test_pkgbuild_applies_the_backend_patch() -> None:
    """Regression: the PKGBUILD selects cuda_v11 but must also enable it.

    cuda_v11 exists only in the tracked patch. Upstream cmake/local.cmake
    aborts with "Unknown OLLAMA_LLAMA_BACKENDS entry 'cuda_v11'" otherwise, so
    selecting the backend without applying the patch fails the build outright.
    """
    assert "0001-add-cuda-v11-backend.patch" in PKGBUILD, (
        "the patch must be listed in source=() or makepkg will not stage it"
    )
    assert "patch -Np1 -i" in PKGBUILD, (
        "the PKGBUILD must apply the patch, e.g. in prepare()"
    )

    # The patch must be a declared source, so makepkg copies it into $srcdir.
    source_block = PKGBUILD[PKGBUILD.index("source=("):PKGBUILD.index(")", PKGBUILD.index("source=("))]
    assert "0001-add-cuda-v11-backend.patch" in source_block, (
        "the patch is not listed inside source=()"
    )


def test_patch_file_exists_next_to_the_pkgbuild() -> None:
    """makepkg resolves source=() entries relative to the PKGBUILD."""
    assert (REPO_ROOT / "0001-add-cuda-v11-backend.patch").is_file(), (
        "the backend patch is missing from the repository root"
    )


def test_patch_actually_adds_the_backend() -> None:
    """The patch must contain all three edits, not a stale partial copy."""
    text = (REPO_ROOT / "0001-add-cuda-v11-backend.patch").read_text(encoding="utf-8")
    for target in ("cmake/local.cmake", "llama/server/CMakePresets.json", "docs/development.md"):
        assert target in text, f"the patch does not touch {target}"
    assert "cuda_v11" in text
    assert "PARENT_SCOPE" in text, (
        "the patch must also fix the CUDAToolkit_ROOT PARENT_SCOPE loss"
    )


def test_b2sums_match_the_shipped_files() -> None:
    """Editing a source without updating b2sums breaks the build.

    The sibling cuda11.8 package shipped this bug: its .pc files were edited
    for the versioned prefix but their checksums were not, so every build died
    with 'One or more files did not pass the validity check!'.
    """
    import hashlib

    source = pkgbuild_array(PKGBUILD, "source")
    sums = pkgbuild_array(PKGBUILD, "b2sums")
    assert len(sums) == len(source), "source and b2sums lengths differ"

    mismatches = []
    for name, expected in zip(source, sums):
        if name.startswith(("http://", "https://", "git+")):
            continue  # External, or SKIP by design for VCS sources.
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        if expected == "SKIP":
            continue
        actual = hashlib.blake2b(path.read_bytes(), digest_size=64).hexdigest()
        if actual != expected:
            mismatches.append(name)

    assert not mismatches, (
        f"these sources do not match their b2sums: {mismatches}. "
        "Regenerate with 'makepkg -g'."
    )


def test_container_script_fetches_source_before_probing() -> None:
    """The architecture probe reads the upstream presets, which makepkg has not
    fetched yet at that point in the script.

    The probe opens src/ollama/llama/server/CMakePresets.json. That path only
    exists after makepkg clones the VCS source, so running the probe first
    failed with FileNotFoundError. The script must clone and patch the source
    before probing.
    """
    script = (REPO_ROOT / ".github" / "scripts" / "build-in-container.sh")
    if not script.is_file():
        pytest.skip("container script not present")
    text = script.read_text(encoding="utf-8")

    if "verify_backend_consistency.py" not in text:
        pytest.skip("this repository does not run the probe")

    probe_at = text.index("verify_backend_consistency.py")
    # The source is fetched with an explicit --branch <tag>, so match the clone
    # rather than a specific URL form.
    clone_at = next(
        i for i, l in enumerate(text.splitlines())
        if "ollama.git" in l and "clone" in l
    )
    clone_at = len("\n".join(text.splitlines()[:clone_at]))
    assert clone_at < probe_at, (
        "the upstream source must be cloned before the architecture probe runs, "
        "otherwise the probe cannot read CMakePresets.json"
    )
    assert "patch -Np1" in text, (
        "the cloned source must be patched, so the probe inspects the presets "
        "the build will actually use"
    )


def test_source_is_pinned_to_a_release_tag() -> None:
    """An unpinned git source cannot be checksummed, so b2sums is SKIP.

    Pinning to the tag that the official Arch `ollama` package builds gives the
    source a real digest and makes two builds of the same pkgver identical.
    """
    source = pkgbuild_array(PKGBUILD, "source")
    git_sources = [s for s in source if s.startswith("git+")]
    assert git_sources, "no VCS source found"

    for entry in git_sources:
        assert "#tag=" in entry, (
            f"VCS source is not pinned to a tag, so its checksum cannot be "
            f"verified: {entry}"
        )
        # The tag must follow pkgver, or the digest and the version drift apart.
        assert "v$pkgver" in entry or "v${pkgver}" in entry, (
            f"the pinned tag must track $pkgver: {entry}"
        )


def test_pinned_source_has_a_real_checksum() -> None:
    """A pinned source must not be SKIPped; that is the point of pinning."""
    source = pkgbuild_array(PKGBUILD, "source")
    sums = pkgbuild_array(PKGBUILD, "b2sums")
    assert len(sums) == len(source), "source and b2sums lengths differ"

    for name, digest in zip(source, sums):
        if not name.startswith("git+"):
            continue
        assert digest != "SKIP", (
            f"the pinned VCS source {name} has b2sums=SKIP, so makepkg will "
            "not verify it"
        )
        # blake2b-512 is 128 hex characters.
        assert len(digest) == 128 and all(c in "0123456789abcdef" for c in digest), (
            f"checksum for {name} is not a valid blake2b-512 digest: {digest}"
        )


def test_pkgver_is_not_a_git_describe_string() -> None:
    """A pinned package must report the tag, not a commit count.

    The previous pkgver() appended `+r<count>+g<sha>`, which contradicts a
    pinned tag and changes with how the repository was cloned.
    """
    pkgver = pkgbuild_scalar(PKGBUILD, "pkgver")
    assert pkgver is not None
    assert "+r" not in pkgver and "+g" not in pkgver, (
        f"pkgver looks like a git-describe string: {pkgver}"
    )
    import re as _re
    assert _re.fullmatch(r"\d+\.\d+\.\d+.*", pkgver), (
        f"pkgver is not a release version: {pkgver}"
    )


def test_ci_probe_clones_the_pinned_tag() -> None:
    """The probe must inspect the revision that will actually be built.

    It previously cloned upstream HEAD, which drifts past the pinned tag; the
    probe would then validate presets that are never compiled.
    """
    script = REPO_ROOT / ".github" / "scripts" / "build-in-container.sh"
    if not script.is_file():
        pytest.skip("container script not present")
    text = script.read_text(encoding="utf-8")

    clone_line = next(
        (l for l in text.splitlines() if "ollama.git" in l and "clone" in l), ""
    )
    assert clone_line, "no upstream clone found in the container script"
    assert "--branch" in clone_line, (
        "the probe must clone a specific tag rather than HEAD: " + clone_line
    )
    assert "tag=" in text, (
        "the tag should be derived from the PKGBUILD so the two cannot drift"
    )


def test_no_source_is_skipped() -> None:
    """Every source can and should carry a real digest.

    A SKIP entry means makepkg downloads or clones without verifying anything.
    For a local patch file there is no excuse: it is right here in the
    repository.
    """
    source = pkgbuild_array(PKGBUILD, "source")
    sums = pkgbuild_array(PKGBUILD, "b2sums")
    assert len(sums) == len(source), "source and b2sums lengths differ"

    skipped = [n for n, d in zip(source, sums) if d == "SKIP"]
    assert not skipped, (
        f"these sources are not checksummed: {skipped}. Regenerate with "
        "'makepkg -g'."
    )
