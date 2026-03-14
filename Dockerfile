FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y libgl1-mesa-glx curl git wget

WORKDIR /app

# 2. 预下载 Forge 引擎 (避免启动时下载)
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git reforge


# 3. 【极致抗造版】增加 Git 缓冲区并单行运行，防止大文件导致连接中断
WORKDIR /app/reforge/repositories
RUN git config --global http.postBuffer 524288000 && \
    git clone --depth 1 https://github.com/stability-ai/stablediffusion.git stable-diffusion-stability-ai || \
    (sleep 5 && git clone --depth 1 https://github.com/stability-ai/stablediffusion.git stable-diffusion-stability-ai)

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