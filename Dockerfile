FROM python:3.10-slim

ARG USERNAME
ARG USER_UID
ARG USER_GID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

ENV PATH=/home/$USERNAME/.local/bin:$PATH

RUN sudo apt update \
    && sudo apt install -y \
        git \
        make \
    && sudo apt clean

ARG POETRY_INSTALL_PATH=/home/$USERNAME/.poetry
RUN python -m venv $POETRY_INSTALL_PATH \
    && $POETRY_INSTALL_PATH/bin/pip install -U pip setuptools \
    && $POETRY_INSTALL_PATH/bin/pip install poetry \
    && mkdir -p /home/$USERNAME/.local/bin \
    && ln -s $POETRY_INSTALL_PATH/bin/poetry /home/$USERNAME/.local/bin/poetry \
    && poetry about \
    && poetry self add poetry-plugin-export
