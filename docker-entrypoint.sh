#!/usr/bin/env bash
set -euo pipefail

# docker-entrypoint.sh
# Runs as root so it can chown volumes that Docker mounts as root,
# then drops to the unprivileged `lrs` user before executing the real command.

# Ensure runtime directories exist and are writable by the lrs user.
mkdir -p /app/data "${HF_HOME:-/app/data/hf_cache}"
chown -R lrs:lrs /app/data /home/lrs

exec gosu lrs "$@"
