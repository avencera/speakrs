#!/usr/bin/env bash
set -euo pipefail

readonly workspace="${SPEAKRS_WORKSPACE:-/workspace}"
readonly models_dir="${workspace}/models"
readonly datasets_dir="${workspace}/datasets"
readonly results_root="${workspace}/gpuq-results"

die() {
    echo "gpuq workload: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
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

verify_dataset() {
    local dataset_dir="${datasets_dir}/$1"
    find "${dataset_dir}/wav" -type f -name '*.wav' -print -quit 2>/dev/null | grep -q . ||
        die "dataset acquisition did not produce a wav file"
    find "${dataset_dir}/rttm" -type f -name '*.rttm' -print -quit 2>/dev/null | grep -q . ||
        die "dataset acquisition did not produce an RTTM file"
}

verify_benchmark_results() {
    local run_dir="$1"
    local result_file
    local -a result_files=()

    mapfile -d '' result_files < <(find "$run_dir" -type f -name results.json -print0)
    ((${#result_files[@]} > 0)) || die "speakrs-bm produced no results.json"
    for result_file in "${result_files[@]}"; do
        jq -e '
            (.implementations | length) > 0
            and all(.implementations[]; .status == "completed")
        ' "$result_file" >/dev/null || die "benchmark result is incomplete: $result_file"
    done
}

print_results_summary() {
    local run_dir="$1"
    local result_file
    local summary="${run_dir}/summary.jsonl"
    while IFS= read -r -d '' result_file; do
        jq -c --arg file "$result_file" '
            .total_audio_minutes as $minutes |
            {file: $file, implementations: (.implementations | with_entries(
                .value = {der: .value.measurement.der,
                    rtfx: (if .value.measurement.time_seconds > 0 then
                        ($minutes * 60 / .value.measurement.time_seconds) else null end)}))}
        ' "$result_file" >>"$summary"
    done < <(find "$run_dir" -type f -name results.json -print0)
    [[ "$(wc -c <"$summary")" -lt 262144 ]] || die "results summary exceeds 256 KiB"
    echo SPEAKRS_RESULTS_BEGIN
    cat "$summary"
    echo SPEAKRS_RESULTS_END
}

main() {
    require_command speakrs-bm
    require_command jq

    local dataset id run_dir result_count manifest
    dataset="$(dataset_from_args "$@")"
    id="$(run_id)"
    run_dir="${results_root}/${id}"
    manifest="${run_dir}/completion.json"

    [[ ! -e "$run_dir" ]] || die "local result directory already exists: $run_dir"
    mkdir -p "$run_dir"

    SPEAKRS_MODELS_DIR="$models_dir" \
        SPEAKRS_DATASETS_DIR="$datasets_dir" \
        SPEAKRS_RESULTS_DIR="$run_dir" \
        SPEAKRS_ROOT="$workspace" \
        speakrs-bm "$@"

    verify_dataset "$dataset"
    verify_benchmark_results "$run_dir"
    print_results_summary "$run_dir"
    result_count="$(find "$run_dir" -type f ! -name completion.json | wc -l | tr -d '[:space:]')"
    jq -n --arg id "$id" --arg path "$run_dir" \
        --arg completed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --argjson count "$result_count" \
        '{schema_version: 1, status: "completed", run_id: $id,
          results_path: $path, result_object_count: $count, completed_at: $completed_at}' >"$manifest"
    echo "gpuq workload completed: $manifest"
}

main "$@"
