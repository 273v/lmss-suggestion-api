#!/bin/env bash
# N.B.: must be run from project root

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

# check if we have an .env.json set and copy from template if not
if [ -f ".env.json" ]; then
  echo "Found .env.json file.  Using it to set environment variables."
else
  echo "No .env.json file found.  Creating one from template."
  cp .env.json.template .env.json
fi

# build with a hard-coded name since we're not using compose/swarm/k8s or registry here
docker build -t 273ventures/lmss-suggestion-api:0.1.0 -f docker/Dockerfile .

# add tags
docker tag 273ventures/lmss-suggestion-api:0.1.0 273ventures/lmss-suggestion-api:latest


