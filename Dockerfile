FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 1. 系统基础环境 (必须包含 git，pip 里的某些包安装时需要它)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libsm6 libxrender1 libxext6 ffmpeg \
    curl git wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. 物理搬运：这一行会把你本地整个 Backend 文件夹(含 reforge)全部塞进容器
COPY . .

# 3. 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt
RUN pip install --no-cache-dir --prefer-binary --ignore-installed -r reforge/requirements.txt

# 4. 权限与启动 (注意：这里使用 ./ 相对路径)
RUN chmod +x start.sh
CMD ["./start.sh"]