import streamlit as st
import trimesh
import numpy as np
import cv2 
from io import BytesIO

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (3D Rotatable)")
st.info("パースのない正射影で、Z値に基づいた正しい深度マップを生成します。サイドバーのボタンでモデルを回転できます。")


# --- 2. セッションステートの初期化と回転ボタン (十字キー配置) ---

# Z軸回転 (左右) のためのヨー角
if 'yaw_angle' not in st.session_state:
    st.session_state['yaw_angle'] = 0
# X軸回転 (上下) のためのピッチ角
if 'pitch_angle' not in st.session_state:
    st.session_state['pitch_angle'] = 0

def rotate_yaw(degrees):
    """Y軸周りの回転 (左右に回り込む)"""
    st.session_state['yaw_angle'] = (st.session_state['yaw_angle'] + degrees) % 360

def rotate_pitch(degrees):
    """X軸周りの回転 (上下に傾ける)"""
    st.session_state['pitch_angle'] = (st.session_state['pitch_angle'] + degrees) % 360

st.sidebar.subheader("モデル回転 (十字キー)")

# 1. 上下回転（上ボタン）: 中央に配置 (X軸)
col_p_up, col_p_mid, col_p_down = st.sidebar.columns([1, 1, 1])
with col_p_mid:
    st.button("上へ 90°", on_click=rotate_pitch, args=(-90,), use_container_width=True, key="pitch_up", help="X軸周りに回転 (モデルが上へ傾く)")

# 2. 左右回転: 中央の行に配置 (Y軸)
col_y_left, col_y_mid, col_y_right = st.sidebar.columns([1, 1, 1])
with col_y_left:
    st.button("左へ 90°", on_click=rotate_yaw, args=(90,), use_container_width=True, key="yaw_left", help="Y軸周りに回転 (カメラが左に回り込む)")
with col_y_right:
    st.button("右へ 90°", on_click=rotate_yaw, args=(-90,), use_container_width=True, key="yaw_right", help="Y軸周りに回転 (カメラが右に回り込む)")

# 3. 上下回転（下ボタン）: 中央に配置 (X軸)
col_p_up_2, col_p_mid_2, col_p_down_2 = st.sidebar.columns([1, 1, 1])
with col_p_mid_2:
    st.button("下へ 90°", on_click=rotate_pitch, args=(90,), use_container_width=True, key="pitch_down", help="X軸周りに回転 (モデルが下へ傾く)")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Y軸角度 (左右): {st.session_state['yaw_angle']}°**")
st.sidebar.markdown(f"**X軸角度 (上下): {st.session_state['pitch_angle']}°**")

# 🔥 修正点 1: 解像度設定をサイドバーに追加
st.sidebar.markdown("---")
st.sidebar.subheader("解像度設定 (ピクセル)")

# デフォルト値は512x512
W = st.sidebar.number_input("幅 (Width)", min_value=100, max_value=2048, value=512, step=100)
H = st.sidebar.number_input("高さ (Height)", min_value=100, max_value=2048, value=512, step=100)

if W * H > 4000000: # 例: 2000x2000以上の処理負荷を制限
    st.sidebar.warning("警告: 高解像度は処理に時間がかかる場合があります。")


# --- 3. ファイルアップロード ---
uploaded_
