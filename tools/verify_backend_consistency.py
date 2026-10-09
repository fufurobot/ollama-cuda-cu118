#!/usr/bin/env python3
"""Cross-check the cuda_v11 backend wiring against the installed toolchain.

The test suite asserts the backend is registered in the right places. This
script additionally checks the *installed* CUDA 11.8 toolkit can actually
compile for the architecture list the preset advertises, which is the failure
mode that only appears at build time.

Usage:
    tools/verify_backend_consistency.py [--cuda-root /opt/cuda-11.8]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRESETS = REPO_ROOT / "src" / "ollama" / "llama" / "server" / "CMakePresets.json"
BACKEND = "cuda_v11"

# sm_90a and everything above sm_90 need CUDA 12 or newer.
CUDA_11_MAX_ARCH = 90


def preset_architectures() -> list[str]:
    data = json.loads(PRESETS.read_text(encoding="utf-8"))
    by_name = {p["name"]: p for p in data["configurePresets"]}
    raw = by_name[f"llama_{BACKEND}_linux"]["cacheVariables"]["CMAKE_CUDA_ARCHITECTURES"]
    return [a for a in raw.split(";") if a]


def nvcc_version(nvcc: Path) -> str:
    out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"cannot run {nvcc}: {out.stderr.strip()}")
    match = re.search(r"release (\S+)", out.stdout)
    return match.group(1) if match else "unknown"


def probe_arch(nvcc: Path, cuda_root: Path, arch: str, host_cc: str | None) -> tuple[bool, str]:
    """Try to compile a trivial kernel for one architecture."""
    real = arch[: -len("-virtual")] if arch.endswith("-virtual") else arch
    sm = real if real.startswith("sm_") else f"sm_{real}"

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cu"
        src.write_text("__global__ void k(){} int main(){return 0;}\n")
        cmd = [
            str(nvcc),
            f"-arch={sm}",
            # glibc >= 2.42 declares cospi/sinpi/rsqrt with __THROW, which
            # conflicts with the CUDA 11.8 headers. The guard derives from
            # __USE_GNU and is #undef'd, so -D cannot override it; dropping
            # _GNU_SOURCE is the supported workaround.
            "-U_GNU_SOURCE",
            "-I", str(cuda_root / "include"),
            "-o", str(Path(tmp) / "probe"),
            str(src),
        ]
        if host_cc:
            cmd[1:1] = ["-ccbin", host_cc]
        out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode == 0:
        return True, ""
    first = next((l for l in out.stderr.splitlines() if "error" in l.lower()), out.stderr.strip()[:200])
    return False, first


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-root", default="/opt/cuda-11.8")
    parser.add_argument("--host-cc", default="/usr/bin/g++-10")
    args = parser.parse_args()

    cuda_root = Path(args.cuda_root)
    nvcc = cuda_root / "bin" / "nvcc"

    archs = preset_architectures()
    print(f"{BACKEND} advertises: {';'.join(archs)}")

    if not nvcc.is_file():
        print(f"SKIP: {nvcc} not present; build the cuda11.8 package first")
        return 0

    version = nvcc_version(nvcc)
    print(f"nvcc: {version}")

    failures = []
    for arch in archs:
        ok, err = probe_arch(nvcc, cuda_root, arch, args.host_cc)
        status = "ok  " if ok else "FAIL"
        print(f"  {status} {arch}" + (f"  <- {err}" if err else ""))
        if not ok:
            failures.append(arch)

    # Guard the upper bound explicitly, since nvcc accepts some future archs
    # only with a newer toolkit and the preset must never list them.
    for arch in archs:
        real = arch[: -len("-virtual")] if arch.endswith("-virtual") else arch
        match = re.fullmatch(r"(\d+)([a-z]*)", real)
        if match and (int(match.group(1)) > CUDA_11_MAX_ARCH or match.group(2)):
            failures.append(arch)

    if failures:
        print(f"FAIL: architectures not compilable by CUDA 11.8: {sorted(set(failures))}")
        return 1
    print("OK: every advertised architecture compiles with this toolkit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
