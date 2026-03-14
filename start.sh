#!/bin/bash

echo "🚀 正在唤醒 4090 算力引擎..."

# 1. 启动 WebUI 后端 (路径已在镜像里固定)
python /app/reforge/launch.py --nowebui --port 7860 --ckpt-dir /workspace/models --skip-torch-cuda-test &

# 2. 循环检查 WebUI 是否启动成功 (探测 7860 端口)
echo "⌛ 正在加载 Realism_Yogi 模型，请稍候..."
while ! curl -s http://127.0.0.1:7860/sdapi/v1/sd-models > /dev/null; do
  sleep 2
done

echo "✅ 引擎就绪，BH41583 的云端分身上岗了！"

# 3. 启动 RunPod 接线员
python -u handler.py