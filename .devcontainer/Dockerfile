FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update
RUN apt -y update

##
RUN apt -y install curl wget vim
# RUN apt -y install python3.9
RUN apt -y install ffmpeg
RUN echo "alias python=python3.8" >> ~/.bashrc
RUN echo "alias python3=python3.8" >> ~/.bashrc
RUN source ~/.bashrc
RUN apt -y install python3-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

##
RUN apt -y install libopencv-dev
RUN apt -y install git x11-apps xserver-xorg
RUN apt -y install python3-dev

##
RUN pip3 install -U pip setuptools
RUN pip3 install matplotlib ipython seaborn jupyter
RUN pip3 install Pillow scikit-learn scikit-image
RUN pip3 install tqdm pandas
RUN pip3 install black flake8 prospector mypy bandit vulture dodgy pylint autopep8
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install torchmetrics captum torchray
RUN pip3 install cmapy ffmpeg-python h5py openmim

##
RUN mim install mmcv-full
RUN pip3 install mmflow

##
RUN mkdir /root/org3dresnet_installer
COPY org3dresnet_installer/* /root/org3dresnet_installer/
RUN chmod 764 /root/org3dresnet_installer/get_org3dresnet.sh
RUN cd /root \
    && ./org3dresnet_installer/get_org3dresnet.sh \
    && cp -r org3dresnet/org3dresnet.egg-info /usr/local/lib/python3.8/dist-packages/ \
    && cp -r org3dresnet /usr/local/lib/python3.8/dist-packages/ \
    && rm -r org3dresnet \
    && rm -r org3dresnet_installer

WORKDIR /root