FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 1. 安装系统依赖
RUN apt-get update && apt-get install -y libgl1-mesa-glx curl git wget

WORKDIR /app

# 2. 预下载 Forge 引擎 (避免启动时下载)
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git reforge

# 3. 【超级加速+抗抖动版】只拉取最新层，减少下载量
WORKDIR /app/reforge/repositories
RUN git clone --depth 1 https://github.com/salesforce/BLIP.git BLIP && \
RUN git clone --depth 1 https://github.com/sczhou/CodeFormer.git CodeFormer && \
RUN git clone --depth 1 https://github.com/CompVis/taming-transformers.git taming-transformers && \
RUN git clone --depth 1 https://github.com/openai/CLIP.git CLIP && \
RUN git clone --depth 1 https://github.com/stability-ai/stablediffusion.git stable-diffusion-stability-ai

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