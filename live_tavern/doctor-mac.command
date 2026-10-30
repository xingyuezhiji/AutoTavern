#!/bin/bash
set -euo pipefail
exec /bin/bash "$(dirname "$0")/run-mac.sh" ui-doctor
