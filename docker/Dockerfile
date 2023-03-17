# Copyright (C) 2023 273 Ventures LLC (https://273ventures.com)
# SPDX-License-Identifier: MIT

# Base LTS image.
FROM ubuntu:22.04

# NOTE: You may need to set a proxy for apt and pip to work in your environment.
# RUN echo 'Acquire::http { Proxy "host.docker.internal"; };' >> /etc/apt/apt.conf.d/01proxy

# Set basic docker environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=UTC

# init apt and cleanup locale first
RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends locales locales-all apt-utils

# set locale and language values in environment
ENV LC_CTYPE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# set locale and language values permanently on image
RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    update-locale LANGUAGE=en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    update-locale LC_CTYPE=en_US.UTF-8

# set timezone to UTC explicitly
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && \
    apt-get install -y --no-install-recommends tzdata \
    && \
    dpkg-reconfigure --frontend noninteractive tzdata

# update and install required packages
RUN apt-get update -y --fix-missing \
       && \
       apt-get dist-upgrade -y \
       && \
       apt-get install -y --no-install-recommends \
            apt-transport-https \
            build-essential \
            ca-certificates \
            curl \
            python3 \
            python3-pip \
            python3-virtualenv \
            python3-wheel \
            python3-dev \
            libxml2-dev \
      && \
      python3 -m pip install --upgrade pip \
      && \
      python3 -m pip install --upgrade setuptools \
      && \
      python3 -m pip install --upgrade wheel

# cleanup  for smallest footprint
RUN apt-get autoremove -y \
    && \
    apt-get clean -y \
    && \
    apt-get purge -y --auto-remove \
    && \
    rm -rf /tmp/* \
    && \
    rm -rf /var/tmp/* \
    && \
    rm -rf /var/log/* \
    && \
    rm -rf /var/lib/apt/lists/* \
    && \
    rm -rf /var/cache/apt/archives/* \
    && \
    rm -rf /var/cache/debconf/*-old

# NOTE: You can enable unattended upgrades, at your own risk, here.
#    && \
#    apt-get install -y --no-install-recommends unattended-upgrades \
#    && \
#    dpkg-reconfigure --frontend noninteractive unattended-upgrades

# create user with name app,  home path /app, and default shell /bin/bash
# make sure to run bash once as the user via sudo -u app bash
RUN useradd -m -s /bin/bash -d /app app \
    && \
    chown -R app:app /app \
    && \
    chmod -R 755 /app

# copy project files
COPY pyproject.toml /app/pyproject.toml
COPY poetry.lock /app/poetry.lock
RUN mkdir /app/lmss_suggestion_api

# poetry install as app user
USER app
WORKDIR /app/

# install poetry
ENV POETRY_HOME "/app/.local/"
ENV PATH "$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.3.2 \
    && \
    chmod -R 755 $POETRY_HOME \
    && \
    cd /app/ \
    && \
    /app/.local/bin/poetry install --only main --no-root \
    && \
    /app/.local/bin/poetry env info

COPY lmss_suggestion_api /app/lmss_suggestion_api

# copy env and run
COPY .env.json /app/.env.json
COPY lmss_suggestion_api/run.sh /app/run.sh

# expose port since we're not using compose/k8s for this release
EXPOSE 8888

# run the api via hypercorn
CMD ["bash", "/app/run.sh"]