#!/bin/bash
# Pegasus GPU environment helpers. Source common.sh before this file.

if [[ "${BASH_SOURCE[0]}" -ef "$0" ]]; then
    echo "Error: This script is meant to be sourced, not executed directly." >&2
    exit 1
fi

load_cuda_module() {
    local cuda_module="${CUDA_MODULE_NAME:-cuda}"
    if ! command -v module &> /dev/null; then
        log_warn "module command not found; cannot load CUDA module '${cuda_module}'."
        return 1
    fi

    log_info "Loading CUDA module: ${cuda_module}"
    if module load "${cuda_module}"; then
        log_info "CUDA module loaded: ${cuda_module}"
        return 0
    fi

    log_warn "failed to load CUDA module '${cuda_module}'."
    return 1
}

detect_cuda_toolkit_root() {
    local root=""
    if [[ -n "${CUDA_TOOLKIT_ROOT_DIR:-}" ]]; then
        root="${CUDA_TOOLKIT_ROOT_DIR}"
    elif [[ -n "${CUDA_HOME:-}" ]]; then
        root="${CUDA_HOME}"
    elif command -v nvcc &> /dev/null; then
        root="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi

    if [[ -n "${root}" && -x "${root}/bin/nvcc" ]]; then
        echo "${root}"
        return 0
    fi

    echo ""
}

require_cuda_toolkit() {
    local cuda_root
    cuda_root="$(detect_cuda_toolkit_root)"
    if [[ -z "${cuda_root}" ]]; then
        log_warn "CUDA toolkit is not visible after loading the CUDA module."
        log_warn "Expected nvcc, CUDA_HOME, or CUDA_TOOLKIT_ROOT_DIR. Refusing to start a production GPU run."
        return 1
    fi

    export CUDA_HOME="${cuda_root}"
    export CUDA_TOOLKIT_ROOT_DIR="${cuda_root}"
    log_info "CUDA toolkit root: ${cuda_root}"
    if [[ -x "${cuda_root}/bin/nvcc" ]]; then
        log_info "nvcc: $("${cuda_root}/bin/nvcc" --version | tail -n 1)"
    fi
    return 0
}
