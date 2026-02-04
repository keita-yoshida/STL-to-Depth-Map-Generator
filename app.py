import streamlit as st
import trimesh
import numpy as np
import cv2 
from io import BytesIO

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (Pro Version)")
st.info("正射影深度マップ生成。反転を修正し、縦横比の自動調整、余白設定、十字キー回転を完備しました。")

# --- 2. セッションステートの初期化 ---
if 'yaw_angle' not in st.session_state:
    st.session_state['yaw_angle'] = 0
if 'pitch_angle' not in st.session_state:
    st.session_state['pitch_angle'] = 0

def rotate_yaw(degrees):
    """Y軸周りの回転 (左右に回り込む)"""
    st.session_state['yaw_angle'] = (st.session_state['yaw_angle'] + degrees) % 360

def rotate_pitch(degrees):
    """X軸周りの回転 (上下に傾ける)"""
    st.session_state['pitch_angle'] = (st.session_state['pitch_angle'] + degrees) % 360

# --- 3. サイドバー設定 (十字キー) ---
st.sidebar.subheader("モデル回転 (十字キー)")
col_p_up, col_p_mid, col_p_down = st.sidebar.columns([1, 1, 1])
with col_p_mid:
    st.button("上へ", on_click=rotate_pitch, args=(-90,), use_container_width=True, key="p_up")

col_y_left, col_y_mid, col_y_right = st.sidebar.columns([1, 1, 1])
with col_y_left:
    st.button("左へ", on_click=rotate_yaw, args=(90,), use_container_width=True, key="y_left")
with col_y_right:
    st.button("右へ", on_click=rotate_yaw, args=(-90,), use_container_width=True, key="y_right")

col_p_up_2, col_p_mid_2, col_p_down_2 = st.sidebar.columns([1, 1, 1])
with col_p_mid_2:
    st.button("下へ", on_click=rotate_pitch, args=(90,), use_container_width=True, key="p_down")

st.sidebar.markdown("---")
st.sidebar.subheader("出力設定")

W = st.sidebar.number_input("出力幅 (px)", min_value=100, max_value=4096, value=512, step=128)
auto_aspect = st.sidebar.checkbox("縦横比を自動調整", value=True)

if auto_aspect:
    H_val = 512 # プレースホルダー
    st.sidebar.caption("高さはモデル形状に合わせて自動計算されます")
else:
    H_val = st.sidebar.number_input("出力高さ (px)", min_value=100, max_value=4096, value=512, step=128)

margin_percent = st.sidebar.slider("余白 (%)", min_value=0, max_value=100, value=10)
padding_factor = 1.0 + (margin_percent / 100.0)

# --- 4. ファイルアップロード ---
uploaded_file = st.file_uploader("STLファイルをアップロード", type=["stl"])

if uploaded_file is not None:
    file_bytes = BytesIO(uploaded_file.getvalue())
    try:
        mesh = trimesh.load_mesh(file_bytes, file_type='stl')
        mesh.vertices -= mesh.centroid

        # 回転の適用 (Y軸左右、X軸上下)
        yaw_rad, pitch_rad = np.radians(st.session_state['yaw_angle']), np.radians(st.session_state['pitch_angle'])
        yaw_matrix = trimesh.transformations.rotation_matrix(yaw_rad, [0, 1, 0])
        pitch_matrix = trimesh.transformations.rotation_matrix(pitch_rad, [1, 0, 0])
        mesh.apply_transform(trimesh.transformations.concatenate_matrices(pitch_matrix, yaw_matrix))

        # --- 5. 解像度とビューポートの計算 ---
        bounds = mesh.bounds 
        min_xyz, max_xyz = bounds[0], bounds[1]
        view_size_x = max_xyz[0] - min_xyz[0]
        view_size_y = max_xyz[1] - min_xyz[1]
        
        # 縦横比の自動調整
        if auto_aspect:
            H = int(W * (view_size_y / view_size_x)) if view_size_x > 0 else W
        else:
            H = H_val

        aspect_ratio_mesh = view_size_x / view_size_y
        aspect_ratio_image = W / H

        if aspect_ratio_mesh > aspect_ratio_image:
            view_width = view_size_x * padding_factor
            view_height = view_width / aspect_ratio_image
        else:
            view_height = view_size_y * padding_factor
            view_width = view_height * aspect_ratio_image

        # --- 6. レイ生成と実行 ---
        camera_origin_z = max_xyz[2] + (max_xyz[2] - min_xyz[2]) + 10 # 十分な高さを確保
        
        x_coords = np.linspace(-view_width / 2, view_width / 2, W)
        # 🔥 ここが反転修正のポイント：上から下へ配置
        y_coords = np.linspace(view_height / 2, -view_height / 2, H)
        
        X, Y = np.meshgrid(x_coords, y_coords)
        
        origins = np.stack((X.flatten(), Y.flatten(), np.full(W * H, camera_origin_z)), axis=1).astype(np.float64)
        directions = np.tile(np.array([0.0, 0.0, -1.0]), (W * H, 1)).astype(np.float64)
        
        locations, index_ray, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        
        # --- 7. 深度マップ生成 ---
        depth_map = np.full(W
