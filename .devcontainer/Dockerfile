FROM debian:bullseye-slim

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openssh-client \
        software-properties-common \
        libgl1-mesa-glx \
        locales \
        python3-pip \
        python-is-python3 \
        python3-dev \
        python3-tk \
        g++ \
        gdb \
        make \
        cmake \
        git \
        texlive-base \
        latexmk \
        texlive-latex-extra \
        tex-gyre \
        librsvg2-bin \
    && rm -rf /var/lib/apt/lists/*


# to fix locale errors
RUN sed -i 's/# \(en_US\.UTF-8 .*\)/\1/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# install some python dependencies
RUN pip install tox genbadge[tests,coverage] numpy matplotlib pytest tornado "setuptools_scm[toml]>=6.2"

