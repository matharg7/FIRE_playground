# rorqual: H100 nodes. Inherits the Alliance defaults, pinning the module
# versions this project was built and tested against.
source "$FIRE_REPO_ROOT/bash_scripts/clusters/drac.sh"
FIRE_MODULES="${FIRE_MODULES:-StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 cuda/13.2}"
