#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workload="${repo_root}/docker/gpuq-workload.sh"
test_root="$(mktemp -d)"
trap 'rm -rf -- "$test_root"' EXIT
fail() { echo "gpuq workload test: $*" >&2; exit 1; }
mkdir -p "$test_root/bin"
cat >"$test_root/bin/s5cmd" <<'FAKE'
#!/usr/bin/env bash
echo 'unexpected S3 access' >&2
exit 99
FAKE
cat >"$test_root/bin/speakrs-bm" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
for name in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL AWS_REGION HF_TOKEN; do
    [[ ! -v "$name" ]]
done
printf '%s\n' "$@" >"$FAKE_BM_ARGS"
mkdir -p "$SPEAKRS_MODELS_DIR" "$SPEAKRS_DATASETS_DIR/voxconverse-dev/"{wav,rttm}
printf model >"$SPEAKRS_MODELS_DIR/segmentation-3.0.safetensors"
printf wav >"$SPEAKRS_DATASETS_DIR/voxconverse-dev/wav/canary.wav"
printf rttm >"$SPEAKRS_DATASETS_DIR/voxconverse-dev/rttm/canary.rttm"
[[ "${FAKE_NO_RESULTS:-false}" != true ]] || exit 0
mkdir -p "$SPEAKRS_RESULTS_DIR/run"
printf '{"total_audio_minutes":2,"implementations":{"speakrs-cuda":{"status":"%s","measurement":{"der":3.5,"time_seconds":4}}}}\n' "${FAKE_BM_STATUS:-completed}" >"$SPEAKRS_RESULTS_DIR/run/results.json"
mkdir -p "$SPEAKRS_RESULTS_DIR/second-run"
cp "$SPEAKRS_RESULTS_DIR/run/results.json" "$SPEAKRS_RESULTS_DIR/second-run/results.json"
FAKE
chmod +x "$test_root/bin/"*
export PATH="$test_root/bin:$PATH"
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL AWS_REGION HF_TOKEN
export FAKE_BM_ARGS="$test_root/args"
export SPEAKRS_WORKSPACE="$test_root/success" SPEAKRS_RUN_ID=canary
injection_marker="$test_root/injected"
literal_dollar='$'
description="${literal_dollar}(touch ${injection_marker})"
"$workload" --dataset voxconverse-dev --impls speakrs --max-files 1 --description "$description" >"$test_root/output"
[[ ! -e "$injection_marker" ]] || fail 'argv evaluated'
grep -Fxq "$description" "$FAKE_BM_ARGS" || fail 'argv changed'
[[ -s "$SPEAKRS_WORKSPACE/models/segmentation-3.0.safetensors" ]] || fail 'models not fetched'
sed -n '/^SPEAKRS_RESULTS_BEGIN$/,/^SPEAKRS_RESULTS_END$/p' "$test_root/output" | sed '1d;$d' >"$test_root/summary"
jq -se 'length == 2 and all(.[]; .implementations["speakrs-cuda"] == {der:3.5,rtfx:30} and (.file | endswith("results.json")))' "$test_root/summary" >/dev/null || fail 'summary invalid'
jq -e --arg path "$SPEAKRS_WORKSPACE/gpuq-results/canary" '.status == "completed" and .results_path == $path and (has("results_prefix") | not)' "$SPEAKRS_WORKSPACE/gpuq-results/canary/completion.json" >/dev/null || fail 'manifest invalid'
for scenario in failed empty; do
    export SPEAKRS_WORKSPACE="$test_root/$scenario"
    export FAKE_BM_STATUS=failed FAKE_NO_RESULTS=false
    [[ "$scenario" != empty ]] || export FAKE_BM_STATUS=completed FAKE_NO_RESULTS=true
    if "$workload" --dataset voxconverse-dev --impls speakrs --max-files 1 >"$test_root/$scenario.log" 2>&1; then
        fail "$scenario unexpectedly succeeded"
    fi
    [[ ! -e "$SPEAKRS_WORKSPACE/gpuq-results/canary/completion.json" ]] || fail 'invalid completion'
done
grep -q 'produced no results.json' "$test_root/empty.log" || fail 'missing-result error absent'
echo 'gpuq workload tests passed'
