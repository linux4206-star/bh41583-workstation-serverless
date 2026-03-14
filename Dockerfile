FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 1. 一次性安装所有系统依赖 (增加 git 以支持 pip 安装)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libsm6 libxrender1 libxext6 ffmpeg \
    curl git wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. 预下载 Forge (如果这一步 128 报错，请参考之前的“物理搬运”也把它拷上去)
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git reforge

# 3. 物理搬运核心组件 (确保你本地有 repositories 文件夹)
COPY ./repositories /app/reforge/repositories

# 4. 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt
RUN pip install --no-cache-dir --prefer-binary --ignore-installed -r reforge/requirements.txt

# 5. 搬入管家和总开关
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh

# 6. 关键：赋予启动脚本执行权限
RUN chmod +x /app/start.sh

# 7. 终极修正：启动指令必须指向 start.sh
CMD ["/app/start.sh"]