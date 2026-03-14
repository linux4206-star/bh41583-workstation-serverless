import runpod
import requests
import time
import os
import base64

# --- 核心逻辑：当有人点“生成”时，RunPod 会调用这个函数 ---
def handler(job):
    # 1. 获取输入参数 (从你的 Streamlit 传过来)
    job_input = job['input']
    prompt = job_input.get("prompt")
    # ... 其他你 main.py 里的参数
    
    # 2. 这里的逻辑相当于你原来的 draw 接口
    # 我们直接调用容器内即将启动的 WebUI (127.0.0.1:7860)
    sd_payload = {
        "prompt": f"Realism_Illustrious_Positive_Embedding, {prompt}",
        "steps": job_input.get("steps", 25),
        "width": job_input.get("width", 1024),
        "height": job_input.get("height", 1024),
        "sampler_name": job_input.get("sampler", "DPM++ 2M Karras"),
        # 加上你引以为傲的 ADetailer 自动修脸
        "alwayson_scripts": {"ADetailer": {"args": [{"ad_model": "face_yolov8n.pt"}]}}
    }

    try:
        # 等待 WebUI 就绪
        response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=sd_payload, timeout=300)
        
        if response.status_code == 200:
            return response.json() # 将图片 B64 返回给前端
        else:
            return {"error": f"WebUI Error: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# 启动 Serverless 引擎
runpod.serverless.start({"handler": handler})