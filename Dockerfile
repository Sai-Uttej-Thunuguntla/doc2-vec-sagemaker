# Build an image that can do inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack

FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>
		 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
         build-essential \
         git \
         curl \
         python-qt4 &&\
         rm -rf /var/lib/apt/lists/*
		 
RUN apt-get clean

ENV PYTHON_VERSION=3.6

RUN curl -o ~/miniconda.sh  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build

COPY environment-cpu.yml .
RUN /opt/conda/bin/conda env create -f environment-cpu.yml
RUN /opt/conda/bin/conda clean -ya


ENV PATH /opt/conda/envs/doc2-vec/bin:$PATH
ENV USER doc2-vec

# set working directory to /doc2-vec
WORKDIR /doc2-vec

CMD source activate doc2-vec ~/.bashrc

# Here we install the extra python packages to run the inference code
RUN python -m pip install flask gevent gunicorn && \
        rm -rf /root/.cache

RUN python -m pip install gensim

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY /src /opt/program

RUN chmod 755 /opt/program
WORKDIR /opt/program
RUN chmod 755 serve
