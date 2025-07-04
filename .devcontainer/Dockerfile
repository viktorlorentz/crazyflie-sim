# .devcontainer/Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1) System build tools + ARM toolchain + OpenGL offscreen for mujoco-py + Ninja
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      swig \
      python3 \
      python3-dev \
      python3-pip \
      gcc-arm-none-eabi \
      binutils-arm-none-eabi \
      libnewlib-arm-none-eabi \
      libosmesa6-dev \
      libgl1-mesa-dev \
      ninja-build \
 && rm -rf /var/lib/apt/lists/*

# 2) Upgrade pip/setuptools/wheel, then install Python dependencies
#    Copy only requirements.txt so layer caching works when code changes but deps don't
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
 && pip3 install --no-cache-dir -r requirements.txt

# 3) Create a non-root 'vscode' user (for Remote-Containers)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
 && useradd -m -s /bin/bash --uid $USER_UID --gid $USER_GID $USERNAME

# 4) Switch to non-root user
USER $USERNAME

# 5) Default workdir inside container (will be mounted to your project root)
WORKDIR /workspaces/crazyflie-sim