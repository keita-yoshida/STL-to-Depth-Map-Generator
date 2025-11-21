import streamlit as st
import trimesh
import numpy as np
import cv2 
from io import BytesIO

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (3D Rotatable)")
st.info("パースのない正射影で、Z値に基づいた正しい深度マップを生成します。サイドバーのボタンでモデルを回転できます。")

# 深度マップの解像度
W, H = 512, 512

# --- 2. セッションステートの初期化と回転ボタン (十字キー配置) ---

# Z軸回転 (左右) のためのヨー角
if 'yaw_angle' not in st.session_state:
    st.session_state['yaw_angle'] = 0
# X軸回転 (上下) のためのピッチ角
if 'pitch_angle' not in st.session_state:
    st.session_state['pitch_angle'] = 0

def rotate_yaw(degrees):
    """Z軸周りの回転 (ヨー)"""
    st.session_state['yaw_angle'] = (st.session_state['yaw_angle'] + degrees) % 360

def rotate_pitch(degrees):
    """X軸周りの回転 (ピッチ)"""
    st.session_state['pitch_angle'] = (st.session_state['pitch_angle'] + degrees) % 360

st.sidebar.subheader("モデル回転 (十字キー)")

# 1. 上下回転（上ボタン）: 中央に配置
col_p_up, col_p_mid, col_p_down = st.sidebar.columns([1, 1, 1])
with col_p_mid:
    st.button("上へ 90°", on_click=rotate_pitch, args=(-90,), use_container_width=True, key="pitch_up", help="X軸周りに回転 (モデルが上へ傾く)")

# 2. 左右回転: 中央の行に配置
col_y_left, col_y_mid, col_y_right = st.sidebar.columns([1, 1, 1])
with col_y_left:
    st.button("左へ 90°", on_click=rotate_yaw, args=(-90,), use_container_width=True, key="yaw_left", help="Z軸周りに回転 (反時計回り)")
with col_y_right:
    st.button("右へ 90°", on_click=rotate_yaw, args=(90,), use_container_width=True, key="yaw_right", help="Z軸周りに回転 (時計回り)")

# 3. 上下回転（下ボタン）: 中央に配置
col_p_up_2, col_p_mid_2, col_p_down_2 = st.sidebar.columns([1, 1, 1])
with col_p_mid_2:
    st.button("下へ 90°", on_click=rotate_pitch, args=(90,), use_container_width=True, key="pitch_down", help="X軸周りに回転 (モデルが下へ傾く)")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Z軸角度 (左右
