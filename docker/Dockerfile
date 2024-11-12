# Usage:
# git clone https://github.com/RosettaCommons/RFdiffusion.git
# cd RFdiffusion
# docker build -f docker/Dockerfile -t rfdiffusion .
# mkdir $HOME/inputs $HOME/outputs $HOME/models
# bash scripts/download_models.sh $HOME/models
# wget -P $HOME/inputs https://files.rcsb.org/view/5TPN.pdb

# docker run -it --rm --gpus all \
#   -v $HOME/models:$HOME/models \
#   -v $HOME/inputs:$HOME/inputs \
#   -v $HOME/outputs:$HOME/outputs \
#   rfdiffusion \
#   inference.output_prefix=$HOME/outputs/motifscaffolding \
#   inference.model_directory_path=$HOME/models \
#   inference.input_pdb=$HOME/inputs/5TPN.pdb \
#   inference.num_designs=3 \
#   'contigmap.contigs=[10-40/A163-181/10-40]'
ARG CUDA_VERSION=11.7.1
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG IMAGE_TYPE=runtime

# Construct base image string
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-${IMAGE_TYPE}-ubuntu${UBUNTU_VERSION}
FROM ${BASE_IMAGE}

# Install system dependencies
RUN apt-get -q update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        git \
        python3 \
        python3-pip \
    && apt-get clean

WORKDIR /src

# Install base Python tooling and requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -U pip wheel setuptools \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# Install SE3Transformer
COPY env/SE3Transformer/se3_transformer se3transformer/se3_transformer
COPY env/SE3Transformer/setup.py se3transformer/
RUN python3 -m pip install --no-cache-dir --no-deps se3transformer/

# Install RFdiffusion
COPY rfdiffusion rfdiffusion/rfdiffusion
COPY scripts rfdiffusion/scripts
COPY setup.py README.md rfdiffusion/
RUN python3 -m pip install --no-cache-dir --no-deps rfdiffusion/

COPY config /usr/local/config
ENV DGLBACKEND="pytorch"

ENTRYPOINT ["run_inference.py"]
