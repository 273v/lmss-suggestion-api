#!/usr/bin/env bash

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

# get bind port or default
BIND_PORT=${BIND_PORT:-8888}

# set to first CLI argument if sent, else check for environment variable, else empty string
OPENAI_API_KEY=${1:-${OPENAI_API_KEY:-""}}

# set default model name
OPENAI_MODEL_NAME=${OPENAI_MODEL_NAME:-"gpt-3.5-turbo-0301"}


# if OPENAI_API_KEY key is set, use it via -e
if [ -z "$OPENAI_API_KEY" ]; then
  echo "No OpenAI API key provided through environment variable or CLI.  Using .docker.env file."
  docker run --publish 8888:${BIND_PORT} \
    --name lmss-suggestion-api \
    --env-file .docker.env \
    lmss-suggestion-api:latest
else
  docker run --publish 8888:${BIND_PORT} \
    --name lmss-suggestion-api \
    -e OPENAI_API_KEY=${OPENAI_API_KEY} \
    -e OPENAI_MODEL_NAME=${OPENAI_MODEL_NAME} \
    lmss-suggestion-api:latest
fi
