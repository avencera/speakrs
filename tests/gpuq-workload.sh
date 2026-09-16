#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly repo_root
readonly workload="${repo_root}/docker/gpuq-workload.sh"
test_root="$(mktemp -d "${TMPDIR:-/tmp}/speakrs-gpuq-test.XXXXXX")"
readonly test_root

cleanup() {
    [[ "$test_root" == "${TMPDIR:-/tmp}/speakrs-gpuq-test."* ]] ||
        return
    rm -rf -- "$test_root"
}
trap cleanup EXIT

fail() {
    echo "gpuq workload test: $*" >&2
    exit 1
}

mkdir -p "${test_root}/bin"

cat >"${test_root}/bin/s5cmd" <<'FAKE_S5CMD'
#!/usr/bin/env bash
set -euo pipefail

[[ "$1" == "--endpoint-url" ]]
shift 2
command="$1"
shift
printf '%s' "$command" >>"$FAKE_S5_LOG"
printf ' %q' "$@" >>"$FAKE_S5_LOG"
printf '\n' >>"$FAKE_S5_LOG"

case "$command" in
    cp)
        source_arg="${*: -2:1}"
        destination="${*: -1}"
        case "$source_arg" in
            s3://speakrs/models/*)
                mkdir -p "$destination"
                printf 'model\n' >"${destination}/segmentation-3.0.onnx"
                ;;
            s3://speakrs/datasets/*)
                mkdir -p "${destination}/wav" "${destination}/rttm"
                printf 'wav\n' >"${destination}/wav/canary.wav"
                printf 'rttm\n' >"${destination}/rttm/canary.rttm"
                ;;
            */completion.json)
                cp "$source_arg" "$FAKE_REMOTE_MANIFEST"
                ;;
        esac
        ;;
    sync)
        source_arg="${*: -2:1}"
        source_dir="${source_arg%/*}"
        count="$(find "$source_dir" -type f | wc -l | tr -d '[:space:]')"
        printf '%s\n' "$count" >"$FAKE_S5_STATE"
        ;;
    ls)
        if [[ -f "$FAKE_S5_STATE" ]]; then
            count="$(<"$FAKE_S5_STATE")"
            if [[ "${FAKE_VERIFY_MISMATCH:-false}" == true ]]; then
                ((count -= 1))
            fi
            for ((i = 0; i < count; i++)); do
                printf '2026/07/24 00:00:00 1 object-%s\n' "$i"
            done
        fi
        ;;
    *)
        echo "unexpected fake s5cmd command: $command" >&2
        exit 1
        ;;
esac
FAKE_S5CMD

cat >"${test_root}/bin/speakrs-bm" <<'FAKE_SPEAKRS_BM'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$@" >"$FAKE_BM_ARGS"
mkdir -p "${SPEAKRS_RESULTS_DIR}/20260724-000000"
printf '{"results":{"speakrs CUDA":{"status":"%s"}}}\n' "${FAKE_BM_STATUS:-completed}" \
    >"${SPEAKRS_RESULTS_DIR}/20260724-000000/results.json"
printf 'results\n' >"${SPEAKRS_RESULTS_DIR}/20260724-000000/results.txt"
FAKE_SPEAKRS_BM

chmod +x "${test_root}/bin/s5cmd" "${test_root}/bin/speakrs-bm"

export PATH="${test_root}/bin:${PATH}"
export AWS_ACCESS_KEY_ID="provider-injected"
export AWS_SECRET_ACCESS_KEY="provider-injected"
export AWS_ENDPOINT_URL="https://t3.storage.dev"
export AWS_REGION="auto"
export HF_TOKEN="provider-injected"
export FAKE_S5_LOG="${test_root}/s5cmd.log"
export FAKE_S5_STATE="${test_root}/s5cmd.state"
export FAKE_BM_ARGS="${test_root}/speakrs-bm.args"
export FAKE_REMOTE_MANIFEST="${test_root}/remote-completion.json"
export SPEAKRS_WORKSPACE="${test_root}/workspace-success"
export SPEAKRS_RUN_ID="canary-safe-argv"

readonly injection_marker="${test_root}/unsafe-argv-executed"
readonly literal_dollar='$'
readonly description="${literal_dollar}(touch ${injection_marker})"
"$workload" \
    --dataset voxconverse-dev \
    --impls speakrs \
    --max-files 1 \
    --description "$description"

[[ ! -e "$injection_marker" ]] || fail "benchmark argv was evaluated by a shell"
grep -Fxq "$description" "$FAKE_BM_ARGS" || fail "benchmark argv was not preserved"
[[ -f "$FAKE_REMOTE_MANIFEST" ]] || fail "completion manifest was not uploaded"
jq -e '
    .schema_version == 1
    and .status == "completed"
    and .run_id == "canary-safe-argv"
    and .result_object_count == 2
' "$FAKE_REMOTE_MANIFEST" >/dev/null || fail "completion manifest is invalid"
tail -n 1 "$FAKE_S5_LOG" | grep -q 'completion.json' ||
    fail "completion manifest was not the final upload"

export FAKE_S5_LOG="${test_root}/s5cmd-benchmark-failure.log"
export FAKE_S5_STATE="${test_root}/s5cmd-benchmark-failure.state"
export FAKE_BM_ARGS="${test_root}/speakrs-bm-failure.args"
export FAKE_REMOTE_MANIFEST="${test_root}/remote-completion-benchmark-failure.json"
export SPEAKRS_WORKSPACE="${test_root}/workspace-benchmark-failure"
export SPEAKRS_RUN_ID="canary-benchmark-failure"
export FAKE_BM_STATUS=failed

if "$workload" --dataset voxconverse-dev --impls speakrs --max-files 1; then
    fail "failed benchmark result unexpectedly succeeded"
fi
[[ ! -e "$FAKE_REMOTE_MANIFEST" ]] ||
    fail "completion manifest was uploaded for a failed benchmark"

export FAKE_S5_LOG="${test_root}/s5cmd-mismatch.log"
export FAKE_S5_STATE="${test_root}/s5cmd-mismatch.state"
export FAKE_BM_ARGS="${test_root}/speakrs-bm-mismatch.args"
export FAKE_REMOTE_MANIFEST="${test_root}/remote-completion-mismatch.json"
export SPEAKRS_WORKSPACE="${test_root}/workspace-mismatch"
export SPEAKRS_RUN_ID="canary-mismatch"
export FAKE_BM_STATUS=completed
export FAKE_VERIFY_MISMATCH=true

if "$workload" --dataset voxconverse-dev --impls speakrs --max-files 1; then
    fail "upload verification mismatch unexpectedly succeeded"
fi
[[ ! -e "$FAKE_REMOTE_MANIFEST" ]] ||
    fail "completion manifest was uploaded after failed verification"

echo "gpuq workload tests passed"
