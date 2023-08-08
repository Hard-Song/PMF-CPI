# 使用NVIDIA的官方CUDA镜像作为基础镜像
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
# FROM pmf1
# docker pull nvidia/cuda:11.7.1-runtime-ubuntu20.04
# 安装一些必要的软件包
ADD . /app
RUN apt-get update && apt-get install -y \
  python3-pip python3.8 python3.8-dev
# 更新pip版本
RUN pip3 install --upgrade pip
# 安装PyTorch
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
RUN pip install torch_geometric
RUN pip install rdkit
RUN pip install networkx
RUN pip install tape_proteins
RUN pip install pandas
# 将工作目录设置为/app
WORKDIR /app
