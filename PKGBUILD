# Maintainer: Christopher D. Degawa <ccom@randomderp.com>
# Contributor: envolution
# Contributor: Alexander F. Rødseth <xyproto@archlinux.org>
# Contributor: Sven-Hendrik Haase <svenstaro@archlinux.org>
# Contributor: Steven Allen <steven@stebalien.com>
# Contributor: Matt Harrison <matt@harrison.us.com>
# Contributor: Kainoa Kanter <kainoa@t1c.dev>
# shellcheck shell=bash disable=SC2034,SC2154

pkgname=ollama-cuda-git
_pkgname=ollama
pkgver=0.34.1.rc2+r1+g38fdb5dd5
pkgrel=1
pkgdesc='Create, run and share large language models (LLMs) with CUDA'
arch=(x86_64)
url='https://github.com/ollama/ollama'
license=(MIT)
options=('!lto')
makedepends=(cmake ninja git go clang)
# The cuda_v11 backend is built against the side-by-side CUDA 11.8 toolkit,
# which ships nvcc that accepts GCC 10 as a host compiler.
makedepends+=('cuda11.8' 'gcc10')
provides=("$_pkgname=$pkgver" "$_pkgname-cuda=$pkgver")
conflicts=("$_pkgname" "$_pkgname-cuda")
source=(git+https://github.com/ollama/ollama.git
  ollama-ld.conf
  ollama.service
  sysusers.conf
  tmpfiles.d)
b2sums=('SKIP'
        '121a7854b5a7ffb60226aaf22eed1f56311ab7d0a5630579525211d5c096040edbcfd2608169a4b6d83e8b4e4855dbb22f8ebf3d52de78a34ea3d4631b7eff36'
        '031e0809a7f564de87017401c83956d43ac29bd0e988b250585af728b952a27d139b3cad0ab1e43750e2cd3b617287d3b81efc4a70ddd61709127f68bd15eabd'
        '68622ac2e20c1d4f9741c57d2567695ec7b5204ab43356d164483cd3bc9da79fad72489bb33c8a17c2e5cb3b142353ed5f466ce857b0f46965426d16fb388632'
        'e8f2b19e2474f30a4f984b45787950012668bf0acb5ad1ebb25cd9776925ab4a6aa927f8131ed53e35b1c71b32c504c700fe5b5145ecd25c7a8284373bb951ed')

pkgver() {
  cd ollama
  _version=$(git describe --tags --abbrev=0 | tr - .)
  _commits=$(git rev-list --count HEAD)
  _short_commit_hash=$(git rev-parse --short=9 HEAD)
  echo "${_version#'v'}+r${_commits}+g${_short_commit_hash}"
}

build() {
  export CGO_CPPFLAGS="${CPPFLAGS}"
  export CGO_CFLAGS="${CFLAGS}"
  export CGO_CXXFLAGS="${CXXFLAGS}"
  export CGO_LDFLAGS="${LDFLAGS}"
  export GOPATH="${srcdir}"
  export GOFLAGS="-buildmode=pie -mod=readonly -modcacherw '-ldflags=-linkmode=external -compressdwarf=false -X=github.com/ollama/ollama/version.Version=$pkgver -X=github.com/ollama/ollama/server.mode=release'"

  cd ollama

  # The superbuild selects GPU backends by name. cuda_v11 is the CUDA 11.8
  # backend added by this packaging effort; upstream ships cuda_v12/cuda_v13
  # only, which need a 12.x or 13.x toolkit.
  local cmake_options=(
    -B build
    -G Ninja
    -W no-dev
    -D CMAKE_BUILD_TYPE=Release
    -D CMAKE_INSTALL_PREFIX=/usr
    -D OLLAMA_LLAMA_BACKENDS="cuda_v11"
    -D OLLAMA_VERSION="$pkgver"
    # Point CMake at the side-by-side CUDA 11.8 toolkit instead of relying on
    # PATH, which may resolve to a newer CUDA.
    -D CUDAToolkit_ROOT="/opt/cuda-11.8"
    -D CMAKE_CUDA_COMPILER="/opt/cuda-11.8/bin/nvcc"
    # sm_90a/sm_100/sm_120 are CUDA 12+ only; the cuda_v11 preset already
    # defaults to sm_61..sm_90, so only override when the user asks for it.
  )

  cmake "${cmake_options[@]}"
  cmake --build build
  go build .
}

check() {
  $_pkgname/$_pkgname --version >/dev/null
  cd $_pkgname
  go test .
}

package() {
  # The superbuild exposes one component per artifact group. `ollama-local`
  # carries the Go binary plus the CPU runners, and `llama-server` carries the
  # CUDA (cuda_v11) runners produced by the GPU backends.
  DESTDIR="$pkgdir" cmake --install ollama/build --component ollama-local
  DESTDIR="$pkgdir" cmake --install ollama/build --component llama-server

  install -Dm755 $_pkgname/$_pkgname "$pkgdir/usr/bin/$_pkgname"
  install -dm755 "$pkgdir/var/lib/ollama"
  install -Dm644 ollama.service "$pkgdir/usr/lib/systemd/system/ollama.service"
  install -Dm644 sysusers.conf "$pkgdir/usr/lib/sysusers.d/ollama.conf"
  install -Dm644 tmpfiles.d "$pkgdir/usr/lib/tmpfiles.d/ollama.conf"
  install -Dm644 $_pkgname/LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"

  ln -s /var/lib/ollama "$pkgdir/usr/share/ollama"
}
# vim:set ts=2 sw=2 et:
