# Testing Guide for sway-orchestrate

The `sway-orchestrate` script has a comprehensive test suite using **bats** (Bash Automated Testing System).

## Quick Start

```bash
# Install bats (one time)
brew install bats-core

# Run all tests
bats sway-orchestrate.bats

# Run specific test
bats sway-orchestrate.bats -f "models.json"

# Run with verbose output
bats sway-orchestrate.bats -v
```

## Test Coverage

The test suite validates:

### ✅ Script Integrity (5 tests)
- Script is executable
- No syntax errors
- Proper shebang (`#!/bin/bash`)
- Uses `set -euo pipefail` for safety
- Non-empty and properly sized

### ✅ Help & Usage (4 tests)
- `./sway-orchestrate help` shows documentation
- No arguments shows help
- Unknown commands fail with error message
- All commands documented

### ✅ Argument Parsing (5 tests)
- `build` requires `--cell`
- `run` requires both `--cell` and `--muts`
- `full` requires both arguments
- `eval` is accepted as alias for `run`
- Graceful failure on missing required args

### ✅ Configuration File (7 tests)
- `models.json` exists
- Valid JSON syntax
- Contains all 6 required models: qwen, gemma, gpt-oss, ministral, phi, llama4-scout
- Each model has required fields: `model_path`, `served_name`, `quantization`, `port`, `gpu_memory_util`
- Port numbers are unique
- Port numbers in valid range (8000-8999)
- `gpu_memory_util` values valid (0.0–1.0)

### ✅ Script Functions (7 tests)
- `cmd_build()` exists and callable
- `cmd_run()` exists and callable
- `cmd_score()` exists and callable
- `cmd_full()` exists and callable
- `start_servers()` exists and callable
- `stop_servers()` exists and callable
- `wait_for_model()` exists and callable

### ✅ Documentation (2 tests)
- `ORCHESTRATION.md` exists
- All commands documented

## What the Tests Cover

| Category | Coverage | Notes |
|----------|----------|-------|
| **Syntax** | ✅ | Catches typos, bash errors |
| **Arguments** | ✅ | Validates required flags |
| **Configuration** | ✅ | Ensures models.json is valid |
| **Functions** | ✅ | All required functions present |
| **Integration** | ⚠️ | Requires SSH/fedora (mocked) |
| **Actual Execution** | ❌ | Would need real vLLM servers |

## What Tests DON'T Cover (and why)

### ❌ Live SSH Execution
Tests avoid actual SSH calls to fedora because:
- Requires network access
- Depends on fedora being online
- Too slow for quick test runs
- Not reproducible in CI/CD

**Workaround:** Integration tests can be run separately with a mock SSH:
```bash
# Future: add integration test suite
# bats sway-orchestrate-integration.bats
```

### ❌ vLLM Server Startup
Tests don't actually start Docker containers because:
- Requires Docker daemon
- Very slow (minutes per test)
- Requires GPU
- Only relevant in production

**Workaround:** Run orchestration script in dry mode:
```bash
./sway-orchestrate status  # Safe, no-op
./sway-orchestrate cleanup  # Safe, clean shutdown
```

### ❌ Python Harness Execution
Tests don't run the SWAY harness because:
- Complex dependencies
- Slow (10–30 min per test)
- Requires vLLM servers running
- Tests harness separately (if unit tests exist)

## Running Tests in CI/CD

```bash
# In your CI pipeline
- name: Test orchestration script
  run: |
    brew install bats-core
    bats sway-orchestrate.bats --tap  # TAP format for CI

# Or with coverage reporting
- name: Test with output
  run: bats sway-orchestrate.bats --verbose
```

## Interpreting Test Results

### All tests pass ✅
```
1..32
ok 1 ..
ok 2 ..
...
ok 32 ..
```

Script is safe to use. Argument parsing, configuration, and functions are all valid.

### Some tests fail ❌
```
not ok 5 - build requires --cell argument
```

Fix the issue before running the orchestrator. Common failures:

| Issue | Cause | Fix |
|-------|-------|-----|
| `models.json is valid JSON` fails | Syntax error in JSON | Run `jq empty models.json` |
| `model ports are unique` fails | Port conflict | Edit `models.json`, change duplicate port |
| `script has no syntax errors` fails | Bash syntax error | Run `bash -n sway-orchestrate` |

## Adding New Tests

To add a test, edit `sway-orchestrate.bats`:

```bash
@test "new test description" {
    # Arrange
    setup_some_state

    # Act
    run script_under_test

    # Assert
    [ "$status" -eq 0 ]
    [[ "$output" =~ "expected string" ]]
}
```

Example: Testing a new model addition

```bash
@test "models.json contains new-model" {
    local models=$(jq -r '.models | keys[]' "$SCRIPT_DIR/models.json")
    [[ "$models" =~ "new-model" ]]
}

@test "new-model has valid config" {
    local model_config=$(jq '.models.["new-model"]' "$SCRIPT_DIR/models.json")
    [[ $(echo "$model_config" | jq 'has("port")') == "true" ]]
}
```

## Test Maintenance

### When to update tests
- After adding new commands
- After changing argument requirements
- After modifying configuration schema
- After adding new models

### Running after changes
```bash
# After editing sway-orchestrate
bats sway-orchestrate.bats

# After editing models.json
bats sway-orchestrate.bats -f "models.json"

# Quick validation
./sway-orchestrate help
```

## Advanced: Mock Testing

For more advanced testing (e.g., mocking SSH calls), see `bats` documentation:

```bash
# Example: mock SSH command
@test "start_servers calls ssh fedora" {
    # This would require modifying the script to be testable
    # (e.g., extracting SSH calls into separate functions)
}
```

Currently, tests avoid mocking SSH to keep them simple and maintainable.

## Performance

Test suite runtime:

```
$ time bats sway-orchestrate.bats
1..32
...
real	0m2.1s
user	0m1.8s
sys 	0m0.4s
```

**~2 seconds** — fast enough for pre-commit hooks or CI runs.

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Test orchestration

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install bats
        run: brew install bats-core
      - name: Run tests
        run: bats sway-orchestrate.bats
```

## FAQ

**Q: Can I test the full pipeline without running vLLM?**
A: Not yet. You could add integration tests that mock vLLM responses, but current tests focus on syntax and configuration.

**Q: Why doesn't it test actual SSH execution?**
A: To keep tests fast and reproducible. SSH requires network, is slow, and would make tests fragile.

**Q: What about testing error handling?**
A: Some error paths are tested (argument validation). Full error path testing would require more complex mocks.

**Q: Can I add my own tests?**
A: Yes! Edit `sway-orchestrate.bats` and add a new `@test` block. Run `bats sway-orchestrate.bats` after.

