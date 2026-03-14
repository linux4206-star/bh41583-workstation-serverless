import runpod
import sqlite3
import requests
import time
import os

# 1. 核心持久化路径：必须放在挂载的云硬盘上，关机也不丢数据
DB_PATH = "/workspace/workstation.db"

def init_db():
    """初始化云端数据库：如果硬盘里没有账本，就新建一个"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 用户表：账号、密码（建议以后加密）、权限等级
    cursor.execute('''CREATE TABLE IF NOT EXISTS users 
                      (username TEXT PRIMARY KEY, password TEXT, level INTEGER)''')
    # 日志表：记录谁在什么时候生了什么图
    cursor.execute('''CREATE TABLE IF NOT EXISTS logs 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, action TEXT, time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    # 注入管理员账号 (BH41583)
    cursor.execute("INSERT OR IGNORE INTO users VALUES ('BH41583', 'admin123', 99)")
    conn.commit()
    conn.close()

def handler(job):
    """云端大脑主循环：接收并执行指令"""
    input_data = job['input']
    action = input_data.get("action") # 动作类型：login, register, generate
    data = input_data.get("data")     # 携带的具体参数
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # --- 逻辑 A：新成员注册 ---
        if action == "register":
            if data.get("invite") != "admin123": # 校验你的邀请码
                return {"success": False, "error": "邀请码无效"}
            cursor.execute("INSERT INTO users VALUES (?, ?, 1)", (data['username'], data['password']))
            conn.commit()
            return {"success": True, "message": "欢迎加入云端算力中心"}

        # --- 逻辑 B：用户登录 ---
        elif action == "login":
            cursor.execute("SELECT level FROM users WHERE username=? AND password=?", (data['username'], data['password']))
            user = cursor.fetchone()
            if user:
                return {"success": True, "level": user[0], "token": "CLOUD_TOKEN_41583"}
            return {"success": False, "error": "身份核验失败"}

        # --- 逻辑 C：4090 暴力生图 ---
        elif action == "generate":
            # 这里的 7860 是你在 Docker 内部启动的 Forge 端口
            # 此时 4090 的所有显存都为你所用
            sd_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
            response = requests.post(sd_url, json=data, timeout=300)
            
            # 顺便记个日志，你是管理员，能随时查谁在用你的算力
            cursor.execute("INSERT INTO logs (user, action) VALUES (?, ?)", 
                           (input_data.get("username", "Guest"), f"Draw: {data.get('prompt')[:20]}..."))
            conn.commit()
            return response.json()

    except Exception as e:
        return {"success": False, "error": f"大脑运行异常: {str(e)}"}
    finally:
        conn.close()

# 启动前确保持久化数据库已就绪
init_db()
runpod.serverless.start({"handler": handler})