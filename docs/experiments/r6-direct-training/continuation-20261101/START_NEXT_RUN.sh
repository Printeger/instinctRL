#!/usr/bin/env bash
set -euo pipefail
printf "%s\n" "WAITING_ADMIN_STORAGE: no runnable confirmation inputs have been issued." >&2
printf "%s\n" "Run storage-confirmation-01/CREATE_STORAGE.sh with sudo in your terminal, then resume preparation." >&2
exit 2
