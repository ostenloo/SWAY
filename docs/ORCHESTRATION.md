# SWAY Orchestration Guide

Manage the SWAY workflow with `sway-orchestrate`. See `./sway-orchestrate help` for full reference.

## Quick Start

```bash
# Full pipeline: build → run → score
./sway-orchestrate full --cell b4 --muts qwen,gemma,gpt-oss

# Or run phases separately
./sway-orchestrate build --cell b4
./sway-orchestrate run --cell b4 --muts qwen,gemma,gpt-oss
./sway-orchestrate score
```

## Common Commands

```bash
# Check server status
./sway-orchestrate status

# Stop all servers
./sway-orchestrate cleanup

# List available models
ssh fedora "cd ~/sway-runner && ./model-runner list"
```

## Workflow Patterns

**Single cell, single MUT:**
```bash
./sway-orchestrate full --cell b4 --muts qwen
```

**Single cell, multiple MUTs:**
```bash
./sway-orchestrate full --cell b4 --muts qwen,gemma,gpt-oss
```

**Multiple cells, reuse prompts:**
```bash
./sway-orchestrate build --cell b4
./sway-orchestrate run --cell b4 --muts qwen,gemma,gpt-oss
./sway-orchestrate score
```

**Iterative testing:**
```bash
./sway-orchestrate full --cell b4 --muts qwen  # Quick test
./sway-orchestrate full --cell b4 --muts qwen,gemma,gpt-oss  # Full sweep
```

## Model Assignments

| Role | Model | Port |
|------|-------|------|
| Simulator | Ministral-3-8B | 8003 |
| Fidelity Checker | Phi-4-mini | 8004 |
| MUT (Qwen) | Qwen3.6-35B-A3B-AWQ | 8000 |
| MUT (Gemma) | Gemma-4-31B-AWQ | 8001 |
| MUT (gpt-oss) | gpt-oss-20b | 8002 |
| Judge | Llama-4-Scout-W4A16 | 8005 |

Peak VRAM (Simulator + Fidelity + Qwen): ~36 GB

## Troubleshooting

**Server startup timeout (3 min):**
- Check models exist: `ssh fedora "du -sh /mnt/data/models/*"`
- Check GPU: `ssh fedora "nvidia-smi"`
- View logs: `ssh fedora "docker logs vllm-qwen"`

**Port conflict:**
```bash
ssh fedora "netstat -tlnp | grep 8000"
```

**Stuck servers:**
```bash
./sway-orchestrate cleanup
```

**Adjust model settings:**
Edit `models.json` to change port, `gpu_memory_util`, or `max_num_seqs`.

## Output

- **Build:** `results/build/{cell}_prompt.txt` (frozen prompt)
- **Run:** `results/run/` (transcripts)
- **Score:** `results/judge/` (scores)

See `./sway-orchestrate help` for full command reference.
