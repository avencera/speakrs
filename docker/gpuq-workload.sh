#!/usr/bin/env bash
set -euo pipefail

readonly bucket="s3://speakrs"
readonly workspace="${SPEAKRS_WORKSPACE:-/workspace}"
readonly models_dir="${workspace}/models"
readonly datasets_dir="${workspace}/datasets"
readonly results_root="${workspace}/gpuq-results"
readonly endpoint="${AWS_ENDPOINT_URL:?AWS_ENDPOINT_URL must be injected by the provider secret store}"

die() {
    echo "gpuq workload: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_secret_env() {
    local name
    for name in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION HF_TOKEN; do
        [[ -n "${!name:-}" ]] || die "${name} must be injected by the provider secret store"
    done
}

dataset_from_args() {
    local dataset="voxconverse-dev"
    local found=false
    local -a argv=("$@")
    local i

    for ((i = 0; i < ${#argv[@]}; i++)); do
        case "${argv[$i]}" in
            --dataset)
                "$found" && die "--dataset may only be specified once"
                ((i + 1 < ${#argv[@]})) || die "--dataset requires a value"
                dataset="${argv[$((i + 1))]}"
                found=true
                ((i += 1))
                ;;
            --dataset=*)
                "$found" && die "--dataset may only be specified once"
                dataset="${argv[$i]#--dataset=}"
                found=true
                ;;
            --models-dir | --models-dir=* | --datasets-dir | --datasets-dir=* | --results-dir | --results-dir=* | --root | --root=*)
                die "gpuq owns benchmark input and output paths"
                ;;
        esac
    done

    [[ "$dataset" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || die "invalid dataset id: $dataset"
    [[ "$dataset" != all ]] || die "gpuq workloads must select one dataset"
    printf '%s\n' "$dataset"
}

run_id() {
    local id="${SPEAKRS_RUN_ID:-}"
    if [[ -z "$id" ]]; then
        id="$(date -u +%Y%m%dT%H%M%SZ)-${HOSTNAME:-container}-$$"
    fi

    [[ "$id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "invalid SPEAKRS_RUN_ID"
    printf '%s\n' "$id"
}

s5cmd_run() {
    s5cmd --endpoint-url "$endpoint" "$@"
}

stage_models() {
    mkdir -p "$models_dir"
    s5cmd_run cp --concurrency 20 "${bucket}/models/*" "${models_dir}/"
    [[ -s "${models_dir}/segmentation-3.0.onnx" ]] ||
        die "model staging did not produce segmentation-3.0.onnx"
}

stage_dataset() {
    local dataset="$1"
    local dataset_dir="${datasets_dir}/${dataset}"

    mkdir -p "$dataset_dir"
    s5cmd_run cp --concurrency 20 "${bucket}/datasets/${dataset}/*" "${dataset_dir}/"
    find "${dataset_dir}/wav" -type f -name '*.wav' -print -quit 2>/dev/null | grep -q . ||
        die "dataset staging did not produce a wav file"
    find "${dataset_dir}/rttm" -type f -name '*.rttm' -print -quit 2>/dev/null | grep -q . ||
        die "dataset staging did not produce an RTTM file"
}

ensure_empty_remote_prefix() {
    local prefix="$1"
    local listing

    if ! listing="$(s5cmd_run ls "${prefix}/")"; then
        die "could not inspect result prefix: $prefix"
    fi
    [[ -z "$listing" ]] || die "result prefix already exists: $prefix"
}

upload_and_verify_results() {
    local local_dir="$1"
    local prefix="$2"
    local expected_count="$3"
    local listing
    local remote_count

    s5cmd_run sync --concurrency 20 --part-size 25 "${local_dir}/*" "${prefix}/"
    if ! listing="$(s5cmd_run ls "${prefix}/")"; then
        die "could not verify uploaded results"
    fi
    remote_count="$(printf '%s\n' "$listing" | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
    [[ "$remote_count" == "$expected_count" ]] ||
        die "result verification failed: expected ${expected_count} objects, found ${remote_count}"
}

verify_benchmark_results() {
    local run_dir="$1"
    local result_file
    local -a result_files=()

    mapfile -d '' result_files < <(find "$run_dir" -type f -name results.json -print0)
    ((${#result_files[@]} > 0)) || die "speakrs-bm produced no results.json"
    for result_file in "${result_files[@]}"; do
        jq -e '
            (.results | length) > 0
            and all(.results[]; .status == "completed")
        ' "$result_file" >/dev/null || die "benchmark result is incomplete: $result_file"
    done
}

write_completion_manifest() {
    local manifest="$1"
    local id="$2"
    local prefix="$3"
    local result_count="$4"
    local completed_at

    completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{\n  "schema_version": 1,\n  "status": "completed",\n  "run_id": "%s",\n  "results_prefix": "%s",\n  "result_object_count": %s,\n  "completed_at": "%s"\n}\n' \
        "$id" "$prefix" "$result_count" "$completed_at" >"$manifest"
}

main() {
    require_command s5cmd
    require_command speakrs-bm
    require_command jq
    require_secret_env

    local dataset
    local id
    local run_dir
    local remote_prefix
    local result_count
    local manifest

    dataset="$(dataset_from_args "$@")"
    id="$(run_id)"
    run_dir="${results_root}/${id}"
    remote_prefix="${bucket}/benchmarks/gpuq/${id}"
    manifest="${run_dir}/completion.json"

    [[ ! -e "$run_dir" ]] || die "local result directory already exists: $run_dir"
    mkdir -p "$run_dir"
    ensure_empty_remote_prefix "$remote_prefix"
    stage_models
    stage_dataset "$dataset"

    SPEAKRS_MODELS_DIR="$models_dir" \
        SPEAKRS_DATASETS_DIR="$datasets_dir" \
        SPEAKRS_RESULTS_DIR="$run_dir" \
        SPEAKRS_ROOT="$workspace" \
        speakrs-bm "$@"

    verify_benchmark_results "$run_dir"
    result_count="$(find "$run_dir" -type f ! -name completion.json | wc -l | tr -d '[:space:]')"
    [[ "$result_count" -gt 0 ]] || die "speakrs-bm produced no result files"
    upload_and_verify_results "$run_dir" "$remote_prefix" "$result_count"

    write_completion_manifest "$manifest" "$id" "$remote_prefix" "$result_count"
    s5cmd_run cp "$manifest" "${remote_prefix}/completion.json"
    echo "gpuq workload completed: ${remote_prefix}/completion.json"
}

main "$@"
