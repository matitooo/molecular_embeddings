#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' 'This cleanup helper is disabled: the previous version recursively deleted training outputs.' >&2
printf '%s\n' 'Review the exact results directory before removing any generated files.' >&2
exit 2
