#!/bin/env bash

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

export POETRY_HOME="/app/.local/ "
export PATH="$POETRY_HOME/bin:$PATH"

# handle environment variables with defaults
BIND_HOST=${BIND_HOST:-0.0.0.0}
BIND_PORT=${BIND_PORT:-8888}
WORKERS=${WORKERS:-1}

# change to path for app
cd /app

# output environment variable settings
echo "BIND_HOST: ${BIND_HOST}"
echo "BIND_PORT: ${BIND_PORT}"
echo "WORKERS: ${WORKERS}"

# run the app
PYTHONPATH=. /app/.local/bin/poetry run hypercorn \
    --bind ${BIND_HOST}:${BIND_PORT} \
    --workers ${WORKERS} \
    --log-level info \
    lmss_suggestion_api.api:app
