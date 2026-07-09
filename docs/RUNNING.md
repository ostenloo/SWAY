# Running the build (all patient profiles)

The optimizer/simulator/judge models run on **Ollama with an RTX 5090**, which lives on
the `fedora` box — **not** the Mac. The Mac is for editing only. So the loop is **not**
run locally against a tunnel; the workflow is **push from the Mac → pull on fedora → run
on fedora**, where the GPU and the model weights are.

> If you point `config.json` at `localhost:11434` from the Mac, you hit the Mac's own
> Ollama (which only has `llama3.2:latest`) and every model call 404s. Run on fedora.

## 0. One-time facts

- **Host:** `ssh fedora` (Tailscale `100.71.95.25`, user `austin`). Reachable only when
  fedora is powered on and on the tailnet.
- **Repo remote:** `origin` = https://github.com/ostenloo/SWAY.git — the sync point
  between Mac and fedora.
- **Profiles:** 6 backbone (`b1`–`b6`) + 3 probes (`p1`–`p3`), from
  `specs/sway_profile_roster.md`.
- **Models** (served by the Ollama container, see `Dockerfile` `ENV MODELS`):
  `qwen3:4b-instruct-2507-q4_K_M` (simulator/optimizer/reference),
  `llama3.1:8b-instruct-q4_K_M` (fidelity/judge),
  `gemma4:12b-it-q4_K_M` (MUT, run-time only).

## 1. Push from the Mac

```bash
cd ~/SWAY
git add -A && git commit -m "…"
git push origin main
```

## 2. Pull on fedora

```bash
ssh fedora
cd ~/SWAY          # clone once if absent: git clone https://github.com/ostenloo/SWAY.git
git pull origin main
```

## 3. Make sure the models are served on the GPU

The container pulls + warms every tag and only reports healthy once they're loaded on the
GPU (`docker-entrypoint.sh`). The host also runs a **native** Ollama systemd service on the
same port 11434 — **stop it first** or the container can't bind.

```bash
sudo systemctl stop ollama          # free port 11434 (native service)
cd ~/SWAY
docker compose up -d --build        # first run pulls ~20 GB; healthcheck waits for it
docker compose ps                   # wait for "healthy"
curl -s localhost:11434/api/tags | python3 -c 'import sys,json;print([m["name"] for m in json.load(sys.stdin)["models"]])'
```

All five tags in `Dockerfile`'s `MODELS` should appear. `ollama ps` should show each on
`100% GPU` (CPU fallback is unusably slow).

## 4. Run the loop over all profiles

Run from **inside** `sway_harness/` (imports are flat, not a package):

```bash
cd ~/SWAY/sway_harness
python main.py build-all --all        # backbone + probes (all 9)
```

Variants:

```bash
python main.py build-all              # backbone only (b1–b6)
python main.py build-all --probes     # probes only (p1–p3)
python main.py build-all --ids b1 p2  # a specific subset
python main.py build --cell b4        # a single cell
```

Knobs live in `sway_harness/config.json` under `"build"`: `n_samples` (arcs per
iteration, 10), `max_iterations` (10), `n_feedback` (5). `main.py build --cell … --max-iters N`
overrides iterations for a single cell.

**This is a long job.** Per profile ≈ `max_iterations × n_samples` arcs, each ~20 turns,
each patient turn separately annotated by the fidelity model — thousands of model calls
per profile, all nine is hours. Run it detached so an SSH drop doesn't kill it:

```bash
cd ~/SWAY/sway_harness
nohup python main.py build-all --all > ../results/build_all.log 2>&1 &
tail -f ../results/build_all.log
# live per-cell progress also at: results/build_artifacts/<cell>/progress.txt
```

## 5. Outputs

- **Frozen prompts:** `results/build/<cell>_prompt.txt`
- **Per-iteration artifacts:** `results/build_artifacts/<cell>/iter_*/`
  (`transcript_*.json` / `.txt`, `fidelity_results.json`, `optimizer_input.txt`,
  `optimizer_prompt.txt`, `summary.txt`, `timing.json`)
- **Score trajectory:** `results/build_artifacts/<cell>/scores_by_iteration.md`
- **Run summary:** `results/build/build_summary.json`

## 6. Get results back to the Mac

Either commit on fedora and pull on the Mac (results are tracked), or rsync:

```bash
# on the Mac
rsync -avz fedora:~/SWAY/results/ ~/SWAY/results/
```

## Notes

- **`gemma4:12b` is a reasoning model.** The MUT call passes `reasoning_effort="none"`
  (`runner.py`) so it answers directly; only Gemma needs this.
- **Guardrail breaks:** Qwen sometimes safety-refuses the depressed roleplay and flips to
  Mandarin mid-arc. `build.py` detects this (CJK), re-rolls with a fresh seed, and discards
  any arc that still breaks. Expect a few re-rolls in the log; a `LEAKY` warning in a
  cell's `summary.txt` means >10% of its arcs broke.
