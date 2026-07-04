#!/usr/bin/env bash
# Start Ollama, pull the exact SWAY model tags, and warm each one so the
# container only reports ready once every model has actually loaded onto the
# GPU — not merely downloaded. Idempotent: pulls into the mounted volume and
# skips models already present. Fails loudly if a model can't be pulled/loaded.
set -euo pipefail

echo "[entrypoint] starting ollama serve..."
ollama serve &
OLLAMA_PID=$!

# Wait for the API before pulling.
echo "[entrypoint] waiting for ollama API..."
until ollama list >/dev/null 2>&1; do sleep 1; done

for m in ${MODELS}; do
  if ollama list | awk '{print $1}' | grep -qx "${m}"; then
    echo "[entrypoint] ${m} already present, skipping pull."
  else
    echo "[entrypoint] pulling ${m}..."
    ollama pull "${m}"
  fi

  # Warm-up: force a load + tiny generation. think:false keeps reasoning models
  # (Gemma) from spending the warm-up budget on hidden reasoning.
  echo "[entrypoint] warming up ${m}..."
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:11434/api/generate \
    -d "{\"model\":\"${m}\",\"prompt\":\"hi\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":8}}")
  if [ "${code}" != "200" ]; then
    echo "[entrypoint] ERROR: warm-up for ${m} returned HTTP ${code}" >&2
    exit 1
  fi

  # Assert it landed on the GPU, not CPU (CPU fallback would be unusably slow).
  # The PROCESSOR column reads e.g. "100% GPU"; grep the model's line for it.
  psline=$(ollama ps | grep -F "${m}" || true)
  if echo "${psline}" | grep -qi 'gpu'; then
    echo "[entrypoint] ${m} loaded on GPU: ${psline}"
  else
    echo "[entrypoint] WARNING: ${m} not on GPU (check nvidia-container-toolkit / --gpus): ${psline}" >&2
  fi
done

echo "[entrypoint] all models ready:"
ollama list

# Hand the container's lifetime to the ollama server.
wait "${OLLAMA_PID}"
