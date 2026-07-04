#!/usr/bin/env bash
# Pre-download the SWAY Ollama models into a chosen directory so the serving
# container can "fast-track" — bind-mount the dir and skip the ~20 GB pull.
#
# Works whether or not Ollama is installed: uses the native `ollama` binary if
# present (via a private temporary server, leaving any system service alone),
# otherwise falls back to a throwaway Docker container.
#
# The target dir is laid out as <dir>/models/{blobs,manifests} — i.e. it is an
# ".ollama" directory, so point the container at it with:
#     SWAY_MODELS_DIR=<dir> docker compose up -d
#
# Usage:
#   scripts/download_models.sh [-d DIR] [-f FILE] [-m "tag1 tag2 ..."]
#
# Config (flags override env override defaults):
#   -d/--dir     OLLAMA_MODELS_DIR   where models land   (default: ./ollama-models)
#   -f/--file    MODELS_FILE         model list file     (default: scripts/models.list)
#   -m/--models  MODELS              inline tag list, skips the file
#   env OLLAMA_IMAGE                 Docker fallback image (default: ollama/ollama:0.31.1)
#   env OLLAMA_DL_PORT              private serve port    (default: 11435)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS_DIR="${OLLAMA_MODELS_DIR:-${SCRIPT_DIR}/../ollama-models}"
MODELS_FILE="${MODELS_FILE:-${SCRIPT_DIR}/models.list}"
MODELS_INLINE="${MODELS:-}"
OLLAMA_IMAGE="${OLLAMA_IMAGE:-ollama/ollama:0.31.1}"
DL_PORT="${OLLAMA_DL_PORT:-11435}"

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; s/^set -euo.*//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dir)    MODELS_DIR="$2";    shift 2 ;;
    -f|--file)   MODELS_FILE="$2";   shift 2 ;;
    -m|--models) MODELS_INLINE="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# ---- Resolve the model list ----
if [[ -n "${MODELS_INLINE}" ]]; then
  read -ra MODELS_ARR <<< "${MODELS_INLINE}"
else
  [[ -f "${MODELS_FILE}" ]] || { echo "model list not found: ${MODELS_FILE}" >&2; exit 1; }
  # Strip comments (# ...), then take the first token of each non-empty line.
  mapfile -t MODELS_ARR < <(sed 's/#.*//' "${MODELS_FILE}" | awk 'NF {print $1}')
fi
[[ ${#MODELS_ARR[@]} -gt 0 ]] || { echo "no models to download" >&2; exit 1; }

mkdir -p "${MODELS_DIR}"
MODELS_DIR="$(cd "${MODELS_DIR}" && pwd)"   # absolutize
echo "[dl] target dir : ${MODELS_DIR}"
echo "[dl] models     : ${MODELS_ARR[*]}"

# ---- Download ----
if command -v ollama >/dev/null 2>&1; then
  echo "[dl] using native ollama on a private server (system service untouched)"
  export OLLAMA_MODELS="${MODELS_DIR}/models"
  export OLLAMA_HOST="127.0.0.1:${DL_PORT}"
  mkdir -p "${OLLAMA_MODELS}"

  ollama serve >"/tmp/sway-dl-serve.$$.log" 2>&1 &
  SERVE_PID=$!
  trap 'kill "${SERVE_PID}" 2>/dev/null || true' EXIT
  echo "[dl] waiting for temporary server on ${OLLAMA_HOST}..."
  until ollama list >/dev/null 2>&1; do sleep 1; done

  for m in "${MODELS_ARR[@]}"; do
    echo "[dl] pulling ${m} ..."
    ollama pull "${m}"
  done
else
  echo "[dl] ollama not installed — using Docker (${OLLAMA_IMAGE})"
  command -v docker >/dev/null 2>&1 || { echo "need either ollama or docker installed" >&2; exit 1; }
  name="sway-ollama-dl-$$"
  docker run -d --name "${name}" -v "${MODELS_DIR}:/root/.ollama" "${OLLAMA_IMAGE}" >/dev/null
  trap 'docker rm -f "${name}" >/dev/null 2>&1 || true' EXIT
  echo "[dl] waiting for container server..."
  until docker exec "${name}" ollama list >/dev/null 2>&1; do sleep 1; done

  for m in "${MODELS_ARR[@]}"; do
    echo "[dl] pulling ${m} ..."
    docker exec "${name}" ollama pull "${m}"
  done
fi

echo
echo "[dl] done. Models are in ${MODELS_DIR}/models"
echo "[dl] fast-track the container with:"
echo "       SWAY_MODELS_DIR=${MODELS_DIR} docker compose up -d"
