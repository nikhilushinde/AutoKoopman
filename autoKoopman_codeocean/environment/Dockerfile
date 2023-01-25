# hash:sha256:88d936f963f37142c04307c92ee12215acf0618b5bbd7eaca37e185a0ce8043d
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.8ubuntu1.1 \
        unzip=6.0-25ubuntu1.1 \
        wget=1.20.3-1ubuntu2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    ortools==9.4.1874

COPY postInstall /
RUN /postInstall
