#!/usr/bin/env bash
# =============================================================================
# OpenClaw Integration Validation Script
#
# Validates the mlx-vlm server works end-to-end through OpenClaw on bastion.
# Requires: ssh access to bastion, OpenClaw gateway running, mlx-vlm server up.
#
# Usage: ./scripts/validate_oc.sh [bastion-host]
# =============================================================================

set -euo pipefail

HOST="${1:-bastion}"
PASS=0
FAIL=0
SKIP=0
RESULTS=()

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

run_test() {
    local name="$1"
    local cmd="$2"
    local check="$3"

    printf "  %-55s " "$name"

    local output
    if ! output=$(bash -c "$cmd" 2>&1); then
        printf "${RED}FAIL${NC} (command error)\n"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL: $name — command error")
        return
    fi

    if printf '%s\n' "$output" | bash -c "$check" > /dev/null 2>&1; then
        printf "${GREEN}PASS${NC}\n"
        PASS=$((PASS + 1))
        RESULTS+=("PASS: $name")
    else
        printf "${RED}FAIL${NC}\n"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL: $name — check failed")
        echo "    Output: $(echo "$output" | head -3)"
    fi
}

run_api_test() {
    local name="$1"
    local endpoint="$2"
    local payload="$3"
    local check="$4"

    local cmd="curl -s --max-time 90 http://100.106.192.127:8080${endpoint} -H 'Content-Type: application/json' -d '${payload}'"
    run_test "$name" "$cmd" "$check"
}

echo "============================================================"
echo "  MLX-VLM + OpenClaw Integration Validation"
echo "============================================================"
echo ""

# --- Connectivity checks ---
echo "Connectivity:"
run_test "SSH to bastion" \
    "ssh -o ConnectTimeout=10 $HOST 'echo ok'" \
    "grep -q ok"

run_test "mlx-vlm server health" \
    "curl -s --max-time 10 http://100.106.192.127:8080/health" \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d.get('status')=='healthy'\""

run_test "mlx-vlm models endpoint" \
    "curl -s --max-time 10 http://100.106.192.127:8080/v1/models" \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d['data'])>0\""

run_test "OpenClaw gateway running" \
    "ssh $HOST 'launchctl list | grep openclaw'" \
    "grep -q openclaw"

run_test "Telegram channel active" \
    "ssh $HOST 'export PATH=/opt/homebrew/bin:\$PATH && openclaw channels list'" \
    "grep -qi telegram"

echo ""

# --- Responses API tests ---
echo "Responses API (/v1/responses):"

run_api_test "Basic text response" \
    "/v1/responses" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","input":"Say hi in one word","max_output_tokens":10}' \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['status']=='completed'; assert len(d['output'])>0\""

run_api_test "Response has correct schema" \
    "/v1/responses" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","input":"Say hello","max_output_tokens":10}' \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert 'id' in d; assert d['object']=='response'; assert 'usage' in d\""

run_api_test "Tools accepted without 422" \
    "/v1/responses" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","input":"What is 2+2? Answer briefly.","max_output_tokens":50,"tools":[{"type":"function","function":{"name":"calc","description":"Calculator","parameters":{"type":"object","properties":{"expr":{"type":"string"}}}}}]}' \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['status'] in ('completed','incomplete'); assert 'output' in d\""

run_api_test "Tools echoed in response" \
    "/v1/responses" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","input":"hi","max_output_tokens":10,"tools":[{"type":"function","function":{"name":"test","parameters":{}}}]}' \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d.get('tools',[]))>0\""

run_api_test "Instructions field works" \
    "/v1/responses" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","input":"What are you?","instructions":"You are a pirate. Respond in pirate speak.","max_output_tokens":50}' \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['status'] in ('completed','incomplete'); assert d.get('instructions') is not None\""

run_api_test "Stop sequences accepted" \
    "/v1/responses" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","input":"Count to 10","max_output_tokens":50,"stop":["5"]}' \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['status']=='completed'\""

echo ""

# --- Chat Completions API tests ---
echo "Chat Completions (/v1/chat/completions):"

run_api_test "Basic chat response" \
    "/v1/chat/completions" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}' \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert len(d['choices'])>0; assert d['choices'][0]['finish_reason']=='stop'\""

run_api_test "Chat with tools" \
    "/v1/chat/completions" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"tools":[{"type":"function","function":{"name":"test","parameters":{}}}]}' \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['finish_reason'] in ('stop','tool_calls')\""

echo ""

# --- Streaming tests ---
echo "Streaming:"

run_api_test "Responses streaming has SSE events" \
    "/v1/responses" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","input":"Say hi","max_output_tokens":10,"stream":true}' \
    "grep -q 'event: response.completed'"

run_api_test "Streaming ends with [DONE]" \
    "/v1/responses" \
    '{"model":"mlx-community/Qwen3.5-35B-A3B-4bit","input":"Say hi","max_output_tokens":10,"stream":true}' \
    "grep -q 'DONE'"

echo ""

# --- OpenClaw agent tests ---
echo "OpenClaw Agent (end-to-end):"

run_test "OC agent basic response" \
    "ssh $HOST 'export PATH=/opt/homebrew/bin:\$PATH && openclaw agent --agent main -m \"Say hello in one word\" --json --timeout 90'" \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'; assert d['result']['meta']['stopReason']=='stop'\""

run_test "OC agent with web search" \
    "ssh $HOST 'export PATH=/opt/homebrew/bin:\$PATH && openclaw agent --agent main -m \"What is the current weather in Kansas City? Use web search.\" --json --timeout 120'" \
    "python3 -c \"import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'\""

echo ""

# --- Summary ---
TOTAL=$((PASS + FAIL + SKIP))
echo "============================================================"
echo "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${SKIP} skipped${NC} / ${TOTAL} total"
echo "============================================================"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "Failures:"
    for r in "${RESULTS[@]}"; do
        if [[ "$r" == FAIL* ]]; then
            echo "  $r"
        fi
    done
    exit 1
fi
