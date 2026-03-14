import streamlit as st
import requests
import base64
import time
import json
from io import BytesIO
from PIL import Image

# ==========================================
# 1. 核心云端配置 (已根据你的信息填入)
# ==========================================
RUNPOD_ENDPOINT_ID = "xm7pxh9qkurjh8"
# 建议在 Streamlit Cloud 部署时将此 Key 放入 Secrets，目前保留你提供的 Key
RUNPOD_API_KEY = "rpa_SWX7F1KQX166KM4BL6RW3PEM0GHWRBONBRW8B0IO36q08d"

RUN_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status"

st.set_page_config(page_title="Illustrious Workstation", layout="wide")

# ==========================================
# 2. 身份识别 (完美保留 BH41583 的原创逻辑)
# ==========================================
def get_client_ip():
    headers = st.context.headers
    cf_ip = headers.get("Cf-Connecting-Ip")
    forwarded_ip = headers.get("X-Forwarded-For")
    if cf_ip: return cf_ip
    elif forwarded_ip: return forwarded_ip.split(",")[0]
    return "127.0.0.1"

client_ip = get_client_ip()

# 初始化 Session 状态
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""
if 'user_level' not in st.session_state: st.session_state['user_level'] = 1
if 'selected_model_path' not in st.session_state:
    st.session_state['selected_model_path'] = "sd/Realism_Yogi.safetensors"

# 管理员免密直达逻辑
if client_ip == "127.0.0.1" and not st.session_state['logged_in']:
    st.session_state.update({"logged_in": True, "username": "BH41583", "user_level": 99, "token": "LOCAL_ADMIN_TOKEN"})
    st.rerun()

# ==========================================
# 3. 核心：云端统一调度函数 (替换原本的 Backend 请求)
# ==========================================
def call_runpod_workstation(action, data):
    """
    统一将 action (login/register/generate) 发往云端 4090 处理
    """
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    
    # 构造标准请求体，带上 IP 用于云端日志审计
    payload = {
        "input": {
            "action": action,
            "data": data,
            "client_ip": client_ip
        }
    }
    
    try:
        resp = requests.post(RUN_URL, json=payload, headers=headers)
        if resp.status_code != 200: return None, f"云端握手失败: {resp.text}"
        
        job_id = resp.json().get("id")
        # 轮询逻辑
        with st.empty():
            while True:
                poll = requests.get(f"{STATUS_URL}/{job_id}", headers=headers).json()
                status = poll.get("status")
                if status == "COMPLETED":
                    return poll.get("output"), None
                elif status == "FAILED":
                    return None, poll.get("error")
                time.sleep(1.5)
    except Exception as e:
        return None, str(e)

# ==========================================
# 4. 登录与注册界面 (重构为云端通信)
# ==========================================
def show_auth_page():
    st.markdown("<br><br><h1 style='text-align: center;'>IMAGE-GENERATION</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>made by BH41583 | Cloud Edition</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>当前接入 IP: {client_ip}</p>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        tab_login, tab_reg = st.tabs(["登录", "注册"])
        with tab_login:
            u = st.text_input("账号", key="l_u")
            p = st.text_input("密码", type="password", key="l_p")
            if st.button("登录账号", use_container_width=True, type="primary"):
                output, err = call_runpod_workstation("login", {"username": u, "password": p})
                if output and output.get("success"):
                    st.session_state.update({
                        "logged_in": True, "username": u, 
                        "user_level": output.get("level", 1), "token": output.get("token")
                    })
                    st.rerun()
                else: st.error(err or "账号密码不匹配")

        with tab_reg:
            new_u = st.text_input("新账号", key="r_u")
            new_p = st.text_input("密码", type="password", key="r_p")
            invite = st.text_input("管理员邀请码", type="password")
            if st.button("创建账号", use_container_width=True):
                output, err = call_runpod_workstation("register", {"username": new_u, "password": new_p, "invite": invite})
                if output and output.get("success"):
                    st.success("注册成功！数据已持久化至云硬盘，请切换至登录页。")
                else: st.error(err or "注册失败")

# ==========================================
# 5. 主生图界面 (完全保留你的参数设计)
# ==========================================
def show_main_ui():
    col1, col2 = st.columns([1, 1.2])
    with col1:
        prompt = st.text_area("您想要看到的 (Prompt):", height=150, placeholder="1girl, solo, masterpiece...")
        n_prompt = st.text_area("您不希望看到的 (Negative):", height=100)
        
        with st.expander("高级参数调节", expanded=False):
            c1, c2 = st.columns(2)
            sampler = c1.selectbox("采样方法", ["DPM++ 2M Karras", "Euler a", "DPM++ SDE Karras"])
            sch = c2.selectbox("调度器", ["Automatic", "Karras", "Exponential"])
            
            c3, c4 = st.columns(2)
            steps = c3.slider("迭代步数", 10, 50, 25)
            cfg = c4.slider("引导系数", 1.0, 15.0, 7.0, 0.5)
            
            size_option = st.selectbox("图片尺寸", ["1024x1024", "832x1216", "1216x832", "自定义"])
            if size_option == "自定义":
                cc1, cc2 = st.columns(2)
                w, h = cc1.number_input("W", 512, 2048, 1024), cc2.number_input("H", 512, 2048, 1024)
            else: w, h = map(int, size_option.split('x'))

            seed = st.number_input("随机种子", -1, 999999999, -1)

        if st.button("Generate now", use_container_width=True, type="primary"):
            params = {
                "prompt": prompt, "n_prompt": n_prompt, "steps": steps, 
                "width": w, "height": h, "cfg_scale": cfg,
                "sampler_name": sampler, "scheduler": sch, "seed": seed,
                "model_name": st.session_state['selected_model_path']
            }
            with st.spinner("🚀 4090 正在暴力输出中..."):
                output, err = call_runpod_workstation("generate", params)
                if output and output.get("images"):
                    st.session_state['last_img'] = Image.open(BytesIO(base64.b64decode(output["images"][0])))
                else: st.error(f"算力分配失败: {err}")

    with col2:
        if 'last_img' in st.session_state:
            st.image(st.session_state['last_img'], use_container_width=True)
            # ... 保存逻辑 ...
        else: st.info("生成的图片将在此实时显示。")

# ==========================================
# 6. 侧边栏：CSS 压缩与模型管理 (保留灵魂)
# ==========================================
st.sidebar.markdown("""<style>.stSidebar .block-container { padding-top: 1rem !important; } hr { margin: 0.5rem 0 !important; }</style>""", unsafe_allow_html=True)
with st.sidebar:
    is_admin = st.session_state.get('user_level', 1) == 99
    st.subheader(st.session_state.get('username', 'Guest'))
    st.markdown(f"<small style='color:{'#FF4B4B' if is_admin else '#00C853'};'>● {'Administrator' if is_admin else 'Standard User'}</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("底模管理")
    model_options = {
        "Realism Yogi(写实)": "sd/Realism_Yogi.safetensors", 
        "Illustrious v1.6(动漫)": "sd/waiIllustriousSDXL_v160.safetensors"
    }
    selected = st.selectbox("当前算力载体", options=list(model_options.keys()), disabled=not is_admin)
    if st.button("确认切换", use_container_width=True, disabled=not is_admin):
        st.session_state['selected_model_path'] = model_options[selected]
        st.success(f"云端路径已对齐: {selected}")

    st.markdown("<div style='height: 180px;'></div>", unsafe_allow_html=True)
    if st.button("退出系统", use_container_width=True):
        st.session_state.clear(); st.rerun()

# 入口
if not st.session_state['logged_in']: show_auth_page()
else: show_main_ui()