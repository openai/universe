FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y libav-tools \
    python3-numpy \
    python3-scipy \
    python3-setuptools \
    python3-pip \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
    swig \
    python3-opengl \
    libboost-all-dev \
    libsdl2-dev \
    wget \
    unzip \
    git \
    golang \
    net-tools \
    iptables \
    libvncserver-dev \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && pip install -U pip

# Install gym
RUN pip install gym[all]

# Get the faster VNC driver
RUN pip install go-vncdriver>=0.4.0

# Force the container to use the go vnc driver
ENV UNIVERSE_VNCDRIVER='go'

WORKDIR /usr/local/universe/

# Cachebusting
COPY ./setup.py ./
COPY ./tox.ini ./

RUN pip install -e .

# Upload our actual code
COPY . ./
