FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN mkdir /circus && \
    mkdir /apps && \
    apt-get update && \
    apt-get install -y\
            software-properties-common\
            curl\
            python3\
            python3-dev && \
    apt-get clean

ENV CUDA_PATH /usr/local/cuda-10.0
ENV PATH $PATH:/usr/local/cuda-10.0/bin:/apps:/circus
ENV LIBRARY_PATH $LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/lib:/usr/local/cuda-10.0/lib64/stubs
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/lib:/usr/local/cuda-10.0/lib64/stubs

RUN ln -s /usr/bin/python3.6 /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --no-cache-dir\
        configparser\
        numpy\
        pillow\
        scipy\
        scikit-image

RUN pip install --no-cache-dir\
        https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl\
        https://download.pytorch.org/whl/cu100/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl

# Add plugin manifest file.
COPY plugin.json /

# Add main script for this CAD.
COPY apps/ /apps/

ENV PATH $PATH:/apps:/circus
ENV PYTHONPATH /apps
WORKDIR /apps

CMD ["python","mr_fat_volumetry.py","-i", "/circus/in","-o","/circus/out", "-m", "/apps/model_random_search_best.pth", "-n", "2", "-g", "0"]