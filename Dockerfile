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

# 3. 【绝对通关版】采用“先尝试克隆，失败则暴力下载”的双重逻辑
WORKDIR /app/reforge/repositories

# 针对 stability-ai，我们先设置 Git 缓冲区，如果克隆失败，再尝试用 curl 强拉
RUN (git config --global http.postBuffer 1048576000 && \
    git clone --depth 1 https://github.com/Stability-AI/stablediffusion.git stable-diffusion-stability-ai) || \
    (curl -fL https://github.com/Stability-AI/stablediffusion/archive/refs/heads/main.zip -o sd.zip && \
    unzip sd.zip && \
    mv stablediffusion-main stable-diffusion-stability-ai && \
    rm sd.zip)

# 其他轻量级仓库继续执行
RUN git clone --depth 1 https://github.com/salesforce/BLIP.git BLIP
RUN git clone --depth 1 https://github.com/sczhou/CodeFormer.git CodeFormer
RUN git clone --depth 1 https://github.com/CompVis/taming-transformers.git taming-transformers
RUN git clone --depth 1 https://github.com/openai/CLIP.git CLIP

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