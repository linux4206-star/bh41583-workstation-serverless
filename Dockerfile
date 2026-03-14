FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y libgl1-mesa-glx curl git wget

WORKDIR /app

# 2. 预下载 Forge 引擎 (避免启动时下载)
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git reforge


# 1. 确保安装了 unzip 和 curl
RUN apt-get update && apt-get install -y libgl1-mesa-glx curl git wget unzip

# 1. 环境准备 (补全所有可能用到的工具)
RUN apt-get update && apt-get install -y libgl1-mesa-glx curl git wget unzip

# 3. 物理搬运：直接将本地已有的仓库文件复制进镜像
WORKDIR /app/reforge/repositories
COPY ./repositories /app/reforge/repositories

# 1. 环境准备：安装 Forge 运行必须的系统级图形库
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 4. 回到工作目录安装依赖
WORKDIR /app

# 优先安装你自定义的依赖，开启镜像源加速 (针对国内构建环境建议加，RunPod 节点可选)
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# 额外安装 Forge 核心依赖，使用 --prefer-binary 避免耗时的现场编译
# 增加 --ignore-installed 确保不会因为与基础镜像冲突而报错
RUN pip install --no-cache-dir --prefer-binary --ignore-installed -r reforge/requirements.txt

# 5. 复制你的 handler 和 启动脚本
COPY . .
RUN chmod +x start.sh

CMD ["bash", "start.sh"]