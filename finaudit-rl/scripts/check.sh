#!/bin/sh
# Single verification command: checks named specs, behavioral invariants,
# on-disk episodes and the unit test-suite.
set -eu
cd "$(dirname "$0")/.."
exec python3 -m finaudit check "$@"
