FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 1. 系统依赖：合并安装，减少镜像层数
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libsm6 libxrender1 libxext6 ffmpeg \
    curl git wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 这一步会把本地 Backend/repositories 下的所有东西拷进 Forge 的对应目录
COPY ./repositories /app/reforge/repositories

# 4. 【解决你的报错】将项目根目录下的所有文件（requirements.txt, handler.py, start.sh 等）拷贝到 /app
COPY . /app

# 5. 安装 Python 依赖 (现在 requirements.txt 已经在屋子里了)
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt
RUN pip install --no-cache-dir --prefer-binary --ignore-installed -r reforge/requirements.txt

# 6. 赋予启动脚本权限
RUN chmod +x /app/start.sh

# 7. 总控开关
CMD ["/app/start.sh"]