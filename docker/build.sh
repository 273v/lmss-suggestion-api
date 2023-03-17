#!/bin/env bash
# N.B.: must be run from project root

# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

# build with a hard-coded name since we're not using compose/swarm/k8s or registry here
docker build -t lmss-suggestion-api -f docker/Dockerfile .
