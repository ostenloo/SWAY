# Reproducible Ollama server for the SWAY harness.
# Serves an OpenAI-compatible API on :11434 that sway_harness/client.py targets.
#
# Pinned to the Ollama version verified working on the RTX 5090 (Blackwell/sm_120)
# host. Bump OLLAMA_VERSION only after re-verifying GPU offload on target hardware.
ARG OLLAMA_VERSION=0.31.1
FROM ollama/ollama:${OLLAMA_VERSION}

# Exact model tags validated against MODELS_SPEC.md (all load 100% GPU, /v1 OK).
# Role assignments live in sway_harness/config.json (source of truth); kept here in sync:
#   qwen3:4b-instruct-2507  -> simulator / optimizer / reference_interlocutor
#   llama3.1:8b-instruct    -> fidelity_checker AND judge
#   gemma4:12b-it           -> MUT (reasoning disabled at request time in runner.py)
#   mistral:7b-instruct-v0.3-> pulled but currently UNASSIGNED (was fidelity_checker)
#   qwen2.5:14b-instruct    -> optimizer-capability candidate (non-reasoning; under evaluation)
ENV MODELS="qwen3:4b-instruct-2507-q4_K_M mistral:7b-instruct-v0.3-q4_K_M llama3.1:8b-instruct-q4_K_M gemma4:12b-it-q4_K_M qwen2.5:14b-instruct-q4_K_M"

# Listen on all interfaces so the published port works.
ENV OLLAMA_HOST=0.0.0.0:11434

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 11434

# Base image sets ENTRYPOINT ["ollama"]; override with our pull+warm-up script.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
