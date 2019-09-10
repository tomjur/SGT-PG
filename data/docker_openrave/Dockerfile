# to build this docker use:
# docker build -t supervised_traj_split -f Dockerfile .

FROM tensorflow/tensorflow:1.12.0-gpu

RUN apt-get update && apt-get install -q -y \
    build-essential \
    git \
    g++ \
    cmake \
    libassimp-dev \
    libxml2-dev \
    libboost-dev \
    libboost-python-dev \
    libeigen3-dev \
    libbullet-dev \
    libode-dev \
    liblapack-dev \
    python-pip \
    python-scipy \
    libfcl-dev \
    "collada-dom2.4-dp*" \
    libsoqt4-dev \
    coreutils \
    nano
    # nano \
    # && rm -rf /var/lib/apt/lists/*

# RUN pip uninstall mpmath
# RUN pip install sympy==0.7.1
RUN pip install sympy==0.7.1 h5py numpy scipy six matplotlib
RUN pip install pyyaml Pillow shapely descartes
RUN pip install tensorflow-probability==0.5.0
# RUN git clone https://github.com/rdiankov/openrave.git
RUN git clone https://github.com/EdsterG/openrave.git

RUN mkdir -p openrave/build
WORKDIR openrave/build
RUN cmake .. && make -j
RUN make install

WORKDIR /root
