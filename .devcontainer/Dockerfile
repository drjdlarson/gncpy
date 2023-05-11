FROM debian:bullseye-slim

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openssh-client \
        software-properties-common \
        python3-pip \
        python-is-python3 \
        python3-dev \
        g++ \
        make \
        cmake \
        git \
        texlive-base \
        latexmk \
        texlive-latex-extra \
        tex-gyre \
        librsvg2-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -y tox

