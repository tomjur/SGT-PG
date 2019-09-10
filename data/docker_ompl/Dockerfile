FROM tensorflow/tensorflow:1.12.0-gpu

RUN apt-get update
RUN apt-get install -y apt-utils wget
RUN curl -O https://ompl.kavrakilab.org/install-ompl-ubuntu.sh
RUN sed -i 's/ sudo -H //' install-ompl-ubuntu.sh
RUN sed -i 's/ sudo//' install-ompl-ubuntu.sh
RUN sed -i 's/PYTHONV=3/echo $PYTHONV/' install-ompl-ubuntu.sh
# RUN sed -i 's/PYTHONV=3/PYTHONV=2/' install-ompl-ubuntu.sh
# RUN sed -i 's/wget/curl/' install-ompl-ubuntu.sh
RUN chmod u+x install-ompl-ubuntu.sh
# RUN ./install-ompl-ubuntu.sh -p
RUN ./install-ompl-ubuntu.sh --python

RUN pip install sympy==0.7.1 h5py numpy scipy six matplotlib
RUN pip install pyyaml Pillow shapely descartes
RUN pip install tensorflow-probability==0.5.0

RUN apt-get install -y python-tk


