import os
import time
from database import SessionLocal, ActivityLog, User
from PIL import Image
from io import BytesIO

# 确保图片目录存在
SAVE_DIR = "outputs"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class WorkstationManager:
    @staticmethod
    def log_activity(username, ip, action, prompt=None, file_path=None):
        db = SessionLocal()
        log = ActivityLog(username=username, ip=ip, action=action, prompt=prompt, file_path=file_path)
        db.add(log)
        db.commit()
        db.close()

    @staticmethod
    def save_image(username, img_obj):
        # 按照“用户名_时间戳”命名，防止重名
        filename = f"{username}_{int(time.time())}.webp"
        path = os.path.join(SAVE_DIR, filename)
        img_obj.save(path, format="WebP", quality=75)
        return path

    @staticmethod
    def check_permission(db_user, required_level):
        # 基于等级的访问控制：等级不足直接打回
        if db_user.level < required_level:
            return False
        return True
    
    @staticmethod
    def get_all_users():
        """从数据库获取所有用户信息"""
        db = SessionLocal()
        users = db.query(User).all()
        db.close()
        return users

    @staticmethod
    def update_user_level(target_username, new_level):
        """更新指定用户的等级"""
        db = SessionLocal()
        user = db.query(User).filter(User.username == target_username).first()
        if user:
            user.level = new_level
            db.commit()
            success = True
        else:
            success = False
        db.close()
        return success