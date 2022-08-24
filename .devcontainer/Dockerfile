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
RUN pip3 install -U pip setuptools==57.5.0
RUN pip3 install matplotlib ipython seaborn jupyter
RUN pip3 install Pillow scikit-learn scikit-image
RUN pip3 install tqdm pandas
RUN pip3 install black flake8 prospector mypy bandit vulture dodgy
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install torchmetrics captum torchray
RUN pip3 install cmapy ffmpeg-python h5py pyclustering openmim wandb

RUN mim install mmcv-full
RUN mim install mmaction2 -f https://github.com/open-mmlab/mmaction2.git
RUN pip3 install mmflow

## opencv Optical_Flow_SDK
# RUN pip3 install numpy
# RUN pip3 uninstall -y opencv-python
# RUN apt -y install libfreeimage3 libfreeimage-dev cmake 
# RUN apt -y install qtbase5-dev qtdeclarative5-dev python3.8-dev
# COPY opencv_installer/* /root/opencv_installer/
# RUN chmod 764 /root/opencv_installer/install_cuda_opencv.sh
# RUN cd /root/opencv_installer && ./install_cuda_opencv.sh

##
RUN mkdir /root/resnet3d_installer
COPY resnet3d_installer/* /root/resnet3d_installer/
RUN chmod 764 /root/resnet3d_installer/get_resnet3d.sh
RUN cd /root \
    && ./resnet3d_installer/get_resnet3d.sh \
    && cp -r resnet3d/resnet3d.egg-info /usr/local/lib/python3.8/dist-packages/ \
    && cp -r resnet3d /usr/local/lib/python3.8/dist-packages/ \
    && rm -r resnet3d \
    && rm -r resnet3d_installer

##
RUN pip3 install cython
RUN cd /root
COPY pyflow_installer/* /root/pyflow_installer/
RUN git clone https://github.com/pathak22/pyflow.git
RUN python3.8 /root/pyflow_installer/modify_orgpyflowsource.py
RUN pip3 install ./pyflow
RUN rm -r pyflow
RUN rm -r /root/pyflow_installer

WORKDIR /root
