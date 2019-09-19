FROM tensorflow/tensorflow:1.12.0-gpu

RUN apt-get update
RUN apt-get install -y apt-utils wget

RUN pip install sympy==0.7.1 h5py numpy scipy six matplotlib
RUN pip install pyyaml Pillow shapely descartes
RUN pip install tensorflow-probability==0.5.0 
RUN pip install pybullet==2.5.5


