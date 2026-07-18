#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_REPO="https://github.com/cersonsky-lab/AniSOAP.git"
UPSTREAM_COMMIT="02aa98a3d4f74c9f637ca166ebe8d6043e0e7b26"
WORKTREE="${ROOT_DIR}/.worktree/AniSOAP"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON:-python3}"

if [[ -e "${WORKTREE}" ]]; then
  echo "${WORKTREE} already exists." >&2
  echo "Move or remove that generated directory before running bootstrap again." >&2
  exit 1
fi

mkdir -p "$(dirname "${WORKTREE}")"
git clone --filter=blob:none --no-checkout "${UPSTREAM_REPO}" "${WORKTREE}"
git -C "${WORKTREE}" fetch --depth 1 origin "${UPSTREAM_COMMIT}"
git -C "${WORKTREE}" checkout --detach "${UPSTREAM_COMMIT}"

install -m 0644 "${ROOT_DIR}/Cargo.toml" "${WORKTREE}/Cargo.toml"
install -m 0644 "${ROOT_DIR}/anisoap/representations/radial_basis.py" "${WORKTREE}/anisoap/representations/radial_basis.py"
install -m 0644 "${ROOT_DIR}/anisoap/representations/ellipsoidal_density_projection.py" "${WORKTREE}/anisoap/representations/ellipsoidal_density_projection.py"
install -m 0644 "${ROOT_DIR}/rust/lib.rs" "${WORKTREE}/rust/lib.rs"
install -m 0644 "${ROOT_DIR}/rust/ellip_expansion/mod.rs" "${WORKTREE}/rust/ellip_expansion/mod.rs"
install -m 0644 "${ROOT_DIR}/rust/ellip_expansion/compute_moments.rs" "${WORKTREE}/rust/ellip_expansion/compute_moments.rs"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  VENV_PYTHON="${VENV_DIR}/bin/python"
elif [[ -x "${VENV_DIR}/Scripts/python.exe" ]]; then
  VENV_PYTHON="${VENV_DIR}/Scripts/python.exe"
else
  echo "Unable to locate the virtual environment Python interpreter." >&2
  exit 1
fi

"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install --editable "${WORKTREE}"

if [[ "$(uname -s)" == "Linux" ]]; then
  "${VENV_PYTHON}" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
else
  "${VENV_PYTHON}" -m pip install torch
fi
"${VENV_PYTHON}" -m pip install --upgrade pytest

SITE_PACKAGES="$("${VENV_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${WORKTREE}" > "${SITE_PACKAGES}/anisoap_torch_worktree.pth"

printf '\nAniSOAP-Torch is ready.\n\n'
printf 'Activate: source .venv/bin/activate\n'
printf 'Validate: python scripts/validate.py --frames 1 && pytest -q\n'
printf 'Benchmark: python scripts/benchmark.py --frames 50 --repeats 3\n'