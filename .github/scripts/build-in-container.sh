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

# makepkg refuses to run as root; the image runs as root.
useradd -m -s /bin/bash builder
echo "builder ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/builder
chmod 0440 /etc/sudoers.d/builder
chown -R builder:builder /work

# --- 1. Build and install the CUDA 11.8 toolkit ---------------------------
echo "== building the cuda11.8 toolkit =="
git clone --depth 1 "$CUDA11_8_REPO" /tmp/cuda11.8
chown -R builder:builder /tmp/cuda11.8
cd /tmp/cuda11.8
sudo -u builder -H bash -euo pipefail -c '
  export BUILDDIR=/work/.makepkg/cuda/build
  export SRCDEST=/work/.makepkg/cuda/src
  export PKGDEST=/work/.makepkg/cuda/pkg
  makepkg --noconfirm --force
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
sudo -u builder -H bash -euo pipefail -c '
  cd /work
  export BUILDDIR=/work/.makepkg/ollama/build
  export SRCDEST=/work/.makepkg/ollama/src
  export PKGDEST=/work/.makepkg/ollama/pkg
  makepkg --noconfirm --force
'
echo "== produced =="
ls -la /work/.makepkg/ollama/pkg
