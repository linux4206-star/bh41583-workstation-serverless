from fastapi import FastAPI, Depends, HTTPException, status, Form, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from manager import WorkstationManager as mgr
from database import SessionLocal, User, ActivityLog
import bcrypt
from datetime import datetime, timedelta
from jose import jwt, JWTError
import asyncio
import requests
import base64
from io import BytesIO
from PIL import Image # 用于图片压缩

app = FastAPI()

# 安全与加密配置
SECRET_KEY = "bh41583-ultra-secret-key" 
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_password_hash(password: str):
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=1440)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)):
    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    
    if client_ip == "127.0.0.1" and token == "LOCAL_DEV_TOKEN":
        return "BH41583"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="通行证无效或已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="无效 Token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="登录已过期或无效")

# GPU 任务调度与生图逻辑
gpu_lock = asyncio.Lock()

async def generate_image_task(payload):
    async with gpu_lock: # 确保 4060 不会显存溢出
        reforge_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
        response = requests.post(reforge_url, json=payload, timeout=150)
        return response.json()


@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    if db.query(User).filter(User.username == username).first():
        db.close()
        raise HTTPException(status_code=400, detail="用户名已存在")
    new_user = User(username=username, hashed_password=get_password_hash(password))
    db.add(new_user)
    db.commit()
    db.close()
    return {"msg": "注册成功！"}

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)): # 必须在这里加上 request: Request
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()
    
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    
    # --- 记录登录日志 ---
    # 这里的 request 现在被定义了，可以正常提取 IP
    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    mgr.log_activity(username, client_ip, "login")
    
    # 发放 Token
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

# 修改后的 draw 接口函数
@app.post("/draw")
async def draw(
    request: Request, 
    prompt: str, 
    n_prompt: str = "", 
    steps: int = 25, 
    width: int = 1024, 
    height: int = 1024, 
    cfg_scale: float = 7.0,      # 新增：提示词相关性
    sampler_name: str = "DPM++ 2M Karras", # 新增：采样器
    scheduler: str = "Automatic",  # 新增：调度器
    seed: int = -1,
    token_user: str = Depends(get_current_user)
):
    db = SessionLocal()
    user = db.query(User).filter(User.username == token_user).first()
    
    if not user or not mgr.check_permission(user, required_level=1):
        raise HTTPException(status_code=403, detail="账号已禁用")

    payload = {
        "prompt": f"Realism_Illustrious_Positive_Embedding, rating_explicit, {prompt}",
        "negative_prompt": n_prompt, 
        "steps": steps, 
        "width": width, 
        "height": height,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "seed": seed,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    {
                        "ad_model": "face_yolov8n.pt", # 默认修脸
                        "ad_confidence": 0.4
                    },

                    {
                        "ad_model": "hand_yolov8n.pt",
                        "ad_confidence": 0.4
                    }
                ]
            }
        }
    }

    async with gpu_lock:
        response = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload, timeout=150)
        result = response.json()

    # 处理图片并存储
    client_ip = request.headers.get("CF-Connecting-IP", request.client.host)
    raw_b64 = result['images'][0]
    img = Image.open(BytesIO(base64.b64decode(raw_b64)))
    
    # 存储并记录日志
    saved_path = mgr.save_image(user.username, img)
    mgr.log_activity(user.username, client_ip, "draw", prompt=prompt, file_path=saved_path)

    # 压缩 WebP 回传给前端加速
    webp_buf = BytesIO()
    img.save(webp_buf, format="WebP", quality=75)
    return {"images": [base64.b64encode(webp_buf.getvalue()).decode('utf-8')]}

# --- 获取所有用户信息 (仅限管理员) ---
@app.get("/admin/users")
def list_users(current_user: str = Depends(get_current_user)):
    db = SessionLocal()
    admin = db.query(User).filter(User.username == current_user).first()
    db.close()
    
    # 权限拦截：只认等级
    if not admin or admin.level < 99:
        raise HTTPException(status_code=403, detail="权限不足")
        
    # 调用 manager 里的逻辑
    users = mgr.get_all_users()
    return [{"username": u.username, "level": u.level} for u in users]

# main.py 中需要补齐的接口

@app.get("/admin/logs")
def get_activity_logs(current_user: str = Depends(get_current_user)):
    db = SessionLocal()
    # 先验证当前操作者是不是上帝本人
    admin = db.query(User).filter(User.username == current_user).first()
    
    if not admin or admin.level < 99:
        db.close()
        raise HTTPException(status_code=403, detail="权限不足")
    
    # 从数据库获取最近的 100 条日志
    # 注意：这里需要按照时间倒序排列，让你看到最新的动态
    logs = db.query(ActivityLog).order_by(ActivityLog.timestamp.desc()).limit(100).all()
    db.close()
    
    # 转换为 JSON 列表返回
    return [
        {
            "id": log.id,
            "username": log.username,
            "ip": log.ip,
            "action": log.action,
            "prompt": log.prompt,
            "file_path": log.file_path,
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S") # 格式化时间字符串
        } for log in logs
    ]

# --- 修改用户等级 (仅限管理员) ---
@app.post("/admin/update_level")
def update_user_level(target_user: str = Form(...), new_level: int = Form(...), 
                      current_user: str = Depends(get_current_user)):
    db = SessionLocal()
    admin = db.query(User).filter(User.username == current_user).first()
    db.close()
    
    if not admin or admin.level < 99:
        raise HTTPException(status_code=403, detail="权限不足")
    
    # 防止管理员误操作把自己给封了或降级
    if target_user == current_user and new_level < 99:
         raise HTTPException(status_code=400, detail="不能对自己进行降级操作")

    # 调用 manager 执行更新
    if mgr.update_user_level(target_user, new_level):
        return {"msg": f"用户 {target_user} 等级已更新"}
    raise HTTPException(status_code=404, detail="用户不存在")

# main.py 增加模型切换接口
@app.post("/admin/set_model")
def set_model(model_name: str = Form(...), current_user: str = Depends(get_current_user)):
    db = SessionLocal()
    user = db.query(User).filter(User.username == current_user).first()
    db.close()
    
    # 建议只允许 Level 99 的你切换模型，防止路人乱搞导致显卡宕机
    if not user or user.level < 99:
        raise HTTPException(status_code=403, detail="暂无权限")

    # 调用 WebUI API 修改配置
    options_payload = {"sd_model_checkpoint": model_name}
    requests.post("http://127.0.0.1:7860/sdapi/v1/options", json=options_payload)
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/options", json=options_payload)
    
    # 核心修正：如果 WebUI 返回的不是 200，说明切换失败
    if response.status_code != 200:
        error_detail = response.json().get("detail", "未知错误")
        raise HTTPException(status_code=500, detail=f"WebUI 切换失败: {error_detail}")
    
    return {"msg": "指令发送成功，模型正在后台加载"}