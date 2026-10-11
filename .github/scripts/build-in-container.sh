#!/bin/bash
# Builds ollama-cuda-git inside an Arch Linux container.
#
# The CUDA 11.8 toolkit is not in the official Arch repositories, so it is
# built and installed from the sibling repository first, exactly as a user
# would have to.
#
# This lives in its own file rather than inline in the workflow because the
# script needs nested quoting (docker run ... bash -c '... sudo ... bash -c
# "..."'). Nested single quotes terminate the outer block, so an inline version
# silently passes a truncated script to the container.
set -euo pipefail

CUDA11_8_REPO="${CUDA11_8_REPO:-https://github.com/fufurobot/cuda11.8.git}"

echo "== refreshing the keyring =="
pacman-key --init >/dev/null 2>&1 || true
pacman-key --populate archlinux >/dev/null 2>&1 || true
pacman -Sy --noconfirm --needed --noprogressbar archlinux-keyring >/dev/null
pacman -S --noconfirm --needed --noprogressbar base-devel sudo git patch >/dev/null
# Build-time dependencies that ARE in the official repositories. gcc10 is
# AUR-only and is not needed here: the PKGBUILD only symlinks gcc-10 inside the
# cuda11.8 package, so it is a runtime dependency of that package rather than a
# build requirement for ollama.
pacman -S --noconfirm --needed --noprogressbar cmake ninja go clang >/dev/null

# makepkg refuses to run as root; the image runs as root.
useradd -m -s /bin/bash builder
echo "builder ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/builder
chmod 0440 /etc/sudoers.d/builder
chown -R builder:builder /work

# --- 0. Build an old host compiler ---------------------------------------
# llama.cpp's CUDA sources go through nvcc, and CUDA 11.8 cannot use a modern
# host compiler: verified locally that GCC 10 compiles cleanly while GCC 16
# fails with "user-defined literal operator not found" and "__malloc__ does not
# take arguments". gcc10 is AUR-only, so it is built once and cached.
if [ -x /work/.makepkg/gcc10/usr/bin/g++-10 ]; then
  echo "== using cached gcc10 =="
else
  echo "== building gcc10 from the AUR (cached after this run) =="
  mkdir -p /work/.makepkg/gcc10
  chown builder:builder /work/.makepkg/gcc10
  builddir=$(mktemp -d)
  chown builder:builder "$builddir"
  # The gcc10 PKGBUILD verifies PGP signatures on the GCC release tarball with
  # validpgpkeys, and a fresh container has none of those keys, so makepkg
  # aborts with "One or more PGP signatures could not be verified!". Import them
  # into the builder's keyring before building.
  sudo -u builder -H bash -euo pipefail -c "
    cd $builddir
    curl -fsSL 'https://aur.archlinux.org/cgit/aur.git/snapshot/gcc10.tar.gz' -o gcc10.tar.gz
    tar xzf gcc10.tar.gz
    cd gcc10
    # Not optional: makepkg aborts if any validpgpkey is missing.
    gpg --keyserver hkps://keyserver.ubuntu.com --recv-keys \
      F3691687D867B81B51CE07D9BBE43771487328A9 \
      86CFFCA918CF3AF47147588051E8B148A9999C34 \
      13975A70E63C361C73AE69EF6EEB81F8981C74C7
    makepkg --noconfirm --nodeps
  "
  # Install the freshly built packages: the cuda11.8 PKGBUILD depends on
  # gcc10 and gcc10-libs, and makepkg verifies runtime dependencies before it
  # will build, so they must actually be registered with pacman rather than
  # merely unpacked.
  pacman -U --noconfirm --needed --noprogressbar "$builddir"/gcc10/*.pkg.tar.zst

  # Also unpack a copy where the build can find g++-10 directly.
  for pkg in "$builddir"/gcc10/*.pkg.tar.zst; do
    bsdtar -xf "$pkg" -C /work/.makepkg/gcc10 usr/bin usr/lib usr/lib64 usr/include 2>/dev/null || true
  done
fi
export CUDA_HOST_CC=/work/.makepkg/gcc10/usr/bin/g++-10
test -x "$CUDA_HOST_CC" || { echo "ERROR: no usable g++-10"; exit 1; }

# --- 1. Build and install the CUDA 11.8 toolkit ---------------------------
echo "== building the cuda11.8 toolkit =="
git clone --depth 1 "$CUDA11_8_REPO" /tmp/cuda11.8
chown -R builder:builder /tmp/cuda11.8
cd /tmp/cuda11.8
sudo -u builder -H bash -euo pipefail -c '
  export BUILDDIR=/work/.makepkg/cuda/build
  export SRCDEST=/work/.makepkg/cuda/src
  export PKGDEST=/work/.makepkg/cuda/pkg
  # --nodeps: cuda11.8 also depends on nvidia-utils and opencl-nvidia, which
  # require an NVIDIA driver stack that a CI runner has no use for. The toolkit
  # is only being built here to compile ollama, and the architecture probe
  # afterwards proves the resulting nvcc actually works.
  makepkg --noconfirm --force --nodeps
'
pacman -U --noconfirm --needed --noprogressbar --overwrite "*" \
  /work/.makepkg/cuda/pkg/*.pkg.tar.zst

cudaroot=$(grep -m1 '^_cudaroot=' /tmp/cuda11.8/PKGBUILD | cut -d= -f2)
echo "CUDA installed at: ${cudaroot}"
"${cudaroot}/bin/nvcc" --version | grep -i release

# --- 2. Verify the backend list against this toolkit ----------------------
# Proves the preset only advertises architectures this nvcc can compile, which
# is exactly what fails later in the build if it is wrong.
echo "== probing the cuda_v11 architecture list =="
pacman -S --noconfirm --needed --noprogressbar python >/dev/null
cd /work
python3 tools/verify_backend_consistency.py --cuda-root "${cudaroot}"

# --- 3. Build ollama ------------------------------------------------------
echo "== building ollama-cuda-git =="
# Put the cached GCC 10 first in PATH and tell CMake explicitly, so nvcc cannot
# fall back to the container's GCC 16.
sudo -u builder -H \
  PATH="/work/.makepkg/gcc10/usr/bin:$PATH" \
  CUDA_HOST_CC="$CUDA_HOST_CC" \
  bash -euo pipefail -c '
  cd /work
  export BUILDDIR=/work/.makepkg/ollama/build
  export SRCDEST=/work/.makepkg/ollama/src
  export PKGDEST=/work/.makepkg/ollama/pkg
  # --nodeps: gcc10 is AUR-only and only needed at runtime by cuda11.8, whose
  # package this script already installed. The verification step afterwards
  # checks the package actually carries cuda_v11 runners.
  makepkg --noconfirm --force --nodeps
'
echo "== produced =="
ls -la /work/.makepkg/ollama/pkg
