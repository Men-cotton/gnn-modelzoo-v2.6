#!/usr/bin/env bash
# Submit all supported PyG Pegasus PBS benchmark jobs.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
pbs_script="${script_dir}/run_pyg_nqsv.pbs"
hwinfo_pbs_script="${script_dir}/collect_hwinfo_nqsv.pbs"
configs_dir="${project_root}/src/cerebras/modelzoo/models/gnn/configs"

dry_run=0
no_compile="${PYG_NO_COMPILE:-0}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] [--no-compile]

Submits the eight PyG configs supported by run_pyg_nqsv.pbs.
Hardware info is collected once through a separate GPU PBS job.
Set PYG_NO_COMPILE=1 or pass --no-compile to disable torch.compile.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --no-compile)
            no_compile=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${pbs_script}" ]]; then
    echo "[ERROR] PBS script not found: ${pbs_script}" >&2
    exit 2
fi

if [[ ! -f "${hwinfo_pbs_script}" ]]; then
    echo "[ERROR] hardware info PBS script not found: ${hwinfo_pbs_script}" >&2
    exit 2
fi

if [[ "${dry_run}" -eq 0 ]] && ! command -v qsub >/dev/null 2>&1; then
    echo "[ERROR] qsub command not found. Use --dry-run to print commands." >&2
    exit 1
fi

validate_num_workers() {
    "${project_root}/benchmark_scripts/validate_num_workers.sh" \
        "${configs_dir}"
}

submit_hwinfo_job() {
    if [[ "${dry_run}" -eq 1 ]]; then
        printf "qsub '%s'\n" "${hwinfo_pbs_script}"
        return 0
    fi

    echo "[submit] hardware info"
    qsub "${hwinfo_pbs_script}"
}

submit_job() {
    local benchmark="$1"
    local profile="$2"
    local config_prefix="$3"
    local config_path="${configs_dir}/${config_prefix}_pyg_${profile}.yaml"
    local vars="PYG_BENCHMARK=${benchmark},PYG_RUN_PROFILE=${profile},PYG_NO_COMPILE=${no_compile}"

    if [[ ! -f "${config_path}" ]]; then
        echo "[ERROR] config not found: ${config_path}" >&2
        exit 2
    fi

    if [[ "${dry_run}" -eq 1 ]]; then
        printf "qsub -v '%s' '%s'\n" "${vars}" "${pbs_script}"
        return 0
    fi

    echo "[submit] ${benchmark} ${profile}"
    qsub -v "${vars}" "${pbs_script}"
}

validate_num_workers

submit_hwinfo_job

submit_job graphsage_ogbn_arxiv throughput_nocache params_graphsage_ogbn_arxiv
submit_job graphsage_ogbn_arxiv throughput_cache params_graphsage_ogbn_arxiv
submit_job graphsage_ogbn_arxiv accuracy_nocache params_graphsage_ogbn_arxiv
submit_job graphsage_ogbn_products throughput_nocache params_graphsage_ogbn_products
submit_job graphsage_ogbn_products throughput_cache params_graphsage_ogbn_products
submit_job graphsage_ogbn_products accuracy_nocache params_graphsage_ogbn_products
submit_job gcn_ogbn_arxiv throughput_nocache params_gcn_ogbn_arxiv
submit_job gcn_ogbn_arxiv accuracy_nocache params_gcn_ogbn_arxiv
