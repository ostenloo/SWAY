# SWAY 

An LLM benchmark for multi-turn affective conversations measuring an LLMs performance in terms of single session therapist drift. 

## Running

Models run on Ollama + RTX 5090 on the `fedora` host, not the Mac. The workflow is
**push from the Mac → pull on fedora → run there**. See [docs/RUNNING.md](docs/RUNNING.md)
for the full build/run runbook (including `python main.py build-all --all`).
