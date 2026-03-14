#!/bin/bash

# 1. 启动 WebUI 引擎 (reForge)，指向你 Volume 里的模型路径
# --nowebui 代表只启动后端 API，不启动那个占内存的网页界面
python /app/reforge/launch.py --nowebui --port 7860 --ckpt-dir /workspace/models --skip-torch-cuda-test &

# 2. 循环检查 WebUI 是否启动成功
echo "正在唤醒 4090 算力引擎..."
while ! curl -s http://127.0.0.1:7860/sdapi/v1/sd-models > /dev/null; do
  sleep 2
done

# 3. 引擎就绪后，启动你的接线员
echo "引擎就绪，接线员已上岗！"
python -u /workspace/backend/handler.py