#!/usr/bin/env bash
# Workspace-local launcher: keep downloaded runtimes/caches outside outputs/.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$project_root/../.." && pwd)"
export UV_CACHE_DIR="$workspace_root/work/uv-cache-linux"
export UV_PYTHON_INSTALL_DIR="$workspace_root/work/python-linux"
export UV_PROJECT_ENVIRONMENT="$workspace_root/work/ngc-314t-linux"
export UV_LINK_MODE=copy
uv_bin="$workspace_root/work/uv-linux/bin/uv"
if [[ ! -x "$uv_bin" ]]; then
    echo "Workspace uv is missing. For a standalone checkout, use uv sync --locked --extra test in Linux." >&2
    exit 1
fi
cd "$project_root"
case "${1:-check}" in
    sync) "$uv_bin" sync --locked --extra test --extra control --extra hydra --extra gym ;;
    check) "$uv_bin" run --locked --extra test --extra control --extra hydra --extra gym python scripts/check_runtime.py --output artifacts/runtime-314t.json ;;
    test) "$uv_bin" run --locked --extra test --extra control --extra hydra --extra gym python -m pytest -q ;;
    examples)
        for example in pitch_tracking world_pitch hybrid_propulsion graph_world compiled_modules batch_rollout instrumented_rollout linear_control control_toolbox f16_level_flight rigid_body geography_atmosphere rendering_data renderer_backend gym_world; do
            "$uv_bin" run --locked --extra test --extra control --extra hydra --extra gym python "examples/$example.py"
        done
        ;;
    *) echo "Expected sync, check, test or examples" >&2; exit 2 ;;
esac
