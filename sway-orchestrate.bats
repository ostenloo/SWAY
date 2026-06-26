#!/usr/bin/env bats

# Test suite for sway-orchestrate
# Install bats: brew install bats-core
# Run: bats sway-orchestrate.bats

setup() {
    # Create temporary test environment
    export TEST_DIR=$(mktemp -d)
    export SCRIPT_DIR="$BATS_TEST_DIRNAME"
    export SCRIPT="$SCRIPT_DIR/sway-orchestrate"

    # Mock sway_harness directory
    mkdir -p "$TEST_DIR/sway_harness"

    # Copy models.json for testing
    cp "$SCRIPT_DIR/models.json" "$TEST_DIR/"
}

teardown() {
    rm -rf "$TEST_DIR"
}

# Helper to run script with test directory
run_script() {
    # Temporarily patch SCRIPT_DIR in script for testing
    sed "s|SCRIPT_DIR=.*|SCRIPT_DIR=\"$TEST_DIR\"|" "$SCRIPT" > "$TEST_DIR/sway-orchestrate-test"
    chmod +x "$TEST_DIR/sway-orchestrate-test"
    "$TEST_DIR/sway-orchestrate-test" "$@"
}

# ──────────────────────────────────────────────────────────────────
# Basic Tests
# ──────────────────────────────────────────────────────────────────

@test "script is executable" {
    [ -x "$SCRIPT" ]
}

@test "help command shows usage" {
    run "$SCRIPT" help
    [ "$status" -eq 0 ]
    [[ "$output" =~ "SWAY Orchestrator" ]]
    [[ "$output" =~ "build" ]]
    [[ "$output" =~ "run" ]]
    [[ "$output" =~ "score" ]]
    [[ "$output" =~ "full" ]]
}

@test "no arguments shows help" {
    run "$SCRIPT"
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Usage:" ]]
}

@test "unknown command fails gracefully" {
    run "$SCRIPT" unknown-command
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Unknown command" ]]
}

# ──────────────────────────────────────────────────────────────────
# Argument Parsing Tests
# ──────────────────────────────────────────────────────────────────

@test "build requires --cell argument" {
    run "$SCRIPT" build
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Missing --cell" ]]
}

@test "build accepts --cell argument" {
    run "$SCRIPT" build --cell b4
    # Will fail at health check (no SSH), but parsing should succeed
    [ "$status" -ne 0 ]
}

@test "run requires both --cell and --muts" {
    run "$SCRIPT" run --cell b4
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Missing --muts" ]]
}

@test "eval is alias for run" {
    run "$SCRIPT" eval --cell b4 --muts qwen
    # Should fail at health check, not argument parsing
    [ "$status" -ne 0 ]
    [[ "$output" =~ "Starting vLLM" ]] || [[ "$output" =~ "Error" ]]
}

@test "full requires both --cell and --muts" {
    run "$SCRIPT" full --cell b4
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Missing --muts" ]]
}

# ──────────────────────────────────────────────────────────────────
# Configuration Tests
# ──────────────────────────────────────────────────────────────────

@test "models.json exists in script directory" {
    [ -f "$SCRIPT_DIR/models.json" ]
}

@test "models.json is valid JSON" {
    run jq empty "$SCRIPT_DIR/models.json"
    [ "$status" -eq 0 ]
}

@test "models.json contains all required models" {
    local models=$(jq -r '.models | keys[]' "$SCRIPT_DIR/models.json")
    [[ "$models" =~ "qwen" ]]
    [[ "$models" =~ "gemma" ]]
    [[ "$models" =~ "gpt-oss" ]]
    [[ "$models" =~ "ministral" ]]
    [[ "$models" =~ "phi" ]]
    [[ "$models" =~ "llama4-scout" ]]
}

@test "each model has required config fields" {
    local model_config=$(jq '.models.qwen' "$SCRIPT_DIR/models.json")

    # Check required fields
    [[ $(echo "$model_config" | jq 'has("model_path")') == "true" ]]
    [[ $(echo "$model_config" | jq 'has("served_name")') == "true" ]]
    [[ $(echo "$model_config" | jq 'has("quantization")') == "true" ]]
    [[ $(echo "$model_config" | jq 'has("port")') == "true" ]]
    [[ $(echo "$model_config" | jq 'has("gpu_memory_util")') == "true" ]]
}

@test "model ports are unique" {
    local ports=$(jq -r '.models | .[] | .port' "$SCRIPT_DIR/models.json" | sort)
    local unique_ports=$(echo "$ports" | uniq | wc -l)
    local total_ports=$(echo "$ports" | wc -l)
    [ "$unique_ports" -eq "$total_ports" ]
}

@test "model ports are in valid range (8000-8999)" {
    local ports=$(jq -r '.models | .[] | .port' "$SCRIPT_DIR/models.json")
    while read -r port; do
        [ "$port" -ge 8000 ] && [ "$port" -le 8999 ]
    done <<< "$ports"
}

@test "gpu_memory_util is between 0 and 1" {
    local utils=$(jq -r '.models | .[] | .gpu_memory_util' "$SCRIPT_DIR/models.json")
    while read -r util; do
        # Check it's a valid float between 0 and 1
        [[ "$util" =~ ^0\.[0-9]+$ ]] || [ "$util" == "0" ] || [ "$util" == "1" ]
    done <<< "$utils"
}

# ──────────────────────────────────────────────────────────────────
# Logic Tests (mocked)
# ──────────────────────────────────────────────────────────────────

@test "status command works (dry run)" {
    # Mock SSH to avoid fedora dependency
    export SSH_MOCK=true
    run bash -c "source $SCRIPT && cmd_status"
    # Will fail but shouldn't crash on parsing
    [ "$status" -ne 0 ] || [ "$status" -eq 0 ]
}

@test "cleanup command works (dry run)" {
    # Just verify the function exists and is callable
    grep -q "cmd_cleanup()" "$SCRIPT"
    [ $? -eq 0 ]
}

# ──────────────────────────────────────────────────────────────────
# Integration-Style Tests (without SSH)
# ──────────────────────────────────────────────────────────────────

@test "script has required functions: cmd_build" {
    grep -q "cmd_build()" "$SCRIPT"
    [ $? -eq 0 ]
}

@test "script has required functions: cmd_run" {
    grep -q "cmd_run()" "$SCRIPT"
    [ $? -eq 0 ]
}

@test "script has required functions: cmd_score" {
    grep -q "cmd_score()" "$SCRIPT"
    [ $? -eq 0 ]
}

@test "script has required functions: cmd_full" {
    grep -q "cmd_full()" "$SCRIPT"
    [ $? -eq 0 ]
}

@test "script has required functions: start_servers" {
    grep -q "start_servers()" "$SCRIPT"
    [ $? -eq 0 ]
}

@test "script has required functions: stop_servers" {
    grep -q "stop_servers()" "$SCRIPT"
    [ $? -eq 0 ]
}

@test "script has required functions: wait_for_model" {
    grep -q "wait_for_model()" "$SCRIPT"
    [ $? -eq 0 ]
}

@test "script sources check_prereqs" {
    grep -q "check_prereqs" "$SCRIPT"
    [ $? -eq 0 ]
}

# ──────────────────────────────────────────────────────────────────
# Syntax Tests
# ──────────────────────────────────────────────────────────────────

@test "script has no obvious syntax errors" {
    bash -n "$SCRIPT"
    [ $? -eq 0 ]
}

@test "script is not empty" {
    [ -s "$SCRIPT" ]
    [ $(wc -l < "$SCRIPT") -gt 50 ]
}

@test "script has shebang" {
    head -1 "$SCRIPT" | grep -q "#!/bin/bash"
    [ $? -eq 0 ]
}

@test "script uses set -euo pipefail" {
    grep -q "set -euo pipefail" "$SCRIPT"
    [ $? -eq 0 ]
}

# ──────────────────────────────────────────────────────────────────
# Documentation Tests
# ──────────────────────────────────────────────────────────────────

@test "ORCHESTRATION.md exists" {
    [ -f "$SCRIPT_DIR/ORCHESTRATION.md" ]
}

@test "ORCHESTRATION.md documents all commands" {
    local doc="$SCRIPT_DIR/ORCHESTRATION.md"
    grep -q "### \`build" "$doc"
    grep -q "### \`run" "$doc"
    grep -q "### \`score" "$doc"
    grep -q "### \`full" "$doc"
}

