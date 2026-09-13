# AscendC CPU twin debug environment (Linux)
# Build:
#   # 把 toolkit .run 放到 ./cann-packages/ 并命名，或传 build-arg
#   docker build -t ascendc-cpu-dev \
#     --build-arg CANN_RUN=Ascend-cann-toolkit_xxx_linux-aarch64.run \
#     -f docker/Dockerfile .
#
# Run:
#   docker run --rm -it -v "$PWD":/workspace -w /workspace ascendc-cpu-dev bash

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl wget \
      gcc g++ make cmake net-tools \
      python3 python3-dev python3-pip \
      gdb vim less \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir \
      numpy decorator sympy cffi 'protobuf<5' scipy requests absl-py

# 可选：构建时安装 CANN（.run 需自行下载）
ARG CANN_RUN=
ARG CANN_INSTALL_PATH=/usr/local/Ascend
COPY docker/cann-packages/ /tmp/cann-packages/

RUN if [ -n "$CANN_RUN" ] && [ -f "/tmp/cann-packages/$CANN_RUN" ]; then \
      chmod +x "/tmp/cann-packages/$CANN_RUN" && \
      "/tmp/cann-packages/$CANN_RUN" --quiet --install --install-path="${CANN_INSTALL_PATH}" ; \
    else \
      echo "NOTE: CANN toolkit .run not installed in image." ; \
      echo "Place the .run under docker/cann-packages/ and pass --build-arg CANN_RUN=..." ; \
    fi

# CANN 9.x uses cann/set_env.sh; older toolkit used ascend-toolkit/latest
ENV ASCEND_INSTALL_PATH=${CANN_INSTALL_PATH}/ascend-toolkit/latest

RUN echo 'if [ -f /usr/local/Ascend/cann/set_env.sh ]; then source /usr/local/Ascend/cann/set_env.sh; elif [ -f "$ASCEND_INSTALL_PATH/set_env.sh" ]; then source "$ASCEND_INSTALL_PATH/set_env.sh"; fi' \
      >> /etc/bash.bashrc

WORKDIR /workspace
CMD ["bash"]
