FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN apt-get -y update && apt-get install -y libzbar-dev
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         git \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
         poppler-utils -y \
         python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN pip install poppler-utils
RUN pip --no-cache-dir install flask gunicorn 
RUN pip install setuptools_rust opencv-python==4.2.0.34 pycryptodome==3.0.0 crypto layoutparser torchvision==0.8.0 boto3
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
RUN pip install "layoutparser[paddledetection]"
RUN pip install "layoutparser[ocr]"

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
RUN pip install awscli
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY src /opt/program
WORKDIR /opt/program
RUN mkdir model
RUN ls
RUN aws s3 cp s3://miap-gme-ocr-integration-bucket/model model/ --recursive
ENTRYPOINT [ "python", "-m" , "predictor", "run", "--host=0.0.0.0","--port=5000"]