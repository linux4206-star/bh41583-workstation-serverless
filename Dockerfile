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

# 4. 回到工作目录安装 Python 依赖
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 额外安装 Forge 自己的依赖
RUN pip install --no-cache-dir -r reforge/requirements.txt

# 5. 复制你的 handler 和 启动脚本
COPY . .
RUN chmod +x start.sh

CMD ["bash", "start.sh"]