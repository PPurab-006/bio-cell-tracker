#!/usr/bin/env bash
set -e

# Run test suite with isolated environment (disabling external host entrypoint plugins)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. .venv/bin/pytest "$@"
