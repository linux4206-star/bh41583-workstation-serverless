# 使用官方 2026 稳定版基础镜像
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 1. 安装系统依赖 (比如处理图片需要的库)
RUN apt-get update && apt-get install -y libgl1-mesa-glx curl git

# 2. 设置工作目录并复制你的 GitHub 代码
WORKDIR /app
COPY . .

# 3. 安装依赖包
RUN pip install --no-cache-dir -r requirements.txt

# 4. 下载 reForge 引擎 (如果还没在 Volume 里，我们可以直接打进镜像)
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git reforge

# 5. 给启动脚本执行权限
RUN chmod +x start.sh

# 6. 运行！
CMD ["bash", "start.sh"]