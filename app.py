import streamlit as st
import trimesh
import numpy as np
import cv2 
from io import BytesIO

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (Final Stable Version)")
st.info("パースのない正射影で、Z値に基づいた正しい深度マップを生成します。サイドバーのボタンでモデルを回転できます。")

# 深度マップの解像度
W, H = 512, 512

# --- 2. セッションステートの初期化と回転ボタン ---
if 'rotation_angle' not in st.session_state:
    st.session_state['rotation_angle'] = 0

def rotate_model(degrees):
    st.session_state['rotation_angle'] = (st.session_state['rotation_angle'] + degrees) % 360

st.sidebar.subheader("モデルの回転")
col1, col2 = st.sidebar.columns(2)

with col1:
    st.button("左へ 90°", on_click=rotate_model, args=(-90,), use_container_width=True)
with col2:
    st.button("右へ 90°", on_click=rotate_model, args=(90,), use_container_width=True)

st.sidebar.markdown(f"**現在の角度: {st.session_state['rotation_angle']}°**")
st.sidebar.markdown("---")


# --- 3. ファイルアップロード ---
uploaded_file = st.file_uploader("STLファイルをアップロードしてください", type=["stl"])

if uploaded_file is not None:
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    try:
        # --- 4. STLの読み込みと前処理 (trimesh) ---
        mesh = trimesh.load_mesh(file_bytes, file_type='stl')
        
        if not isinstance(mesh, trimesh.Trimesh):
            st.error("アップロードされたファイルは有効なメッシュデータではありません。")
            st.stop() 

        mesh.vertices -= mesh.centroid

        # 回転処理の適用
        angle_rad = np.radians(st.session_state['rotation_angle'])
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
        mesh.apply_transform(rotation_matrix)

        # --- 5. 仮想カメラと正射影の設定 ---
        bounds = mesh.bounds 
        min_xyz = bounds[0]
        max_xyz = bounds[1]
        
        view_size_x = max_xyz[0] - min_xyz[0]
        view_size_y = max_xyz[1] - min_xyz[1]
        
        aspect_ratio_mesh = view_size_x / view_size_y
        aspect_ratio_image = W / H

        if aspect_ratio_mesh > aspect_ratio_image:
            view_width = view_size_x * 1.2 
            view_height = view_width / aspect_ratio_image
        else:
            view_height = view_size_y * 1.2
            view_width = view_height * aspect_ratio_image

        camera_origin_z = max_xyz[2] + view_size_y * 2 
        
        # --- 6. レイトレーシングのためのレイを生成 (正射影) ---
        
        x_coords = np.linspace(-view_width / 2, view_width / 2, W)
        y_coords = np.linspace(-view_height / 2, view_height / 2, H)
        
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # レイの始点を生成 (Line 93を含むブロック。括弧は完璧に閉じられています)
        origins_stack = np.stack((X.flatten(), Y.flatten(), np.full(W * H, camera_origin_z)), axis=1)
        ray_origins = origins_stack.astype(np.float64)
        
        # レイの方向は全てZ軸マイナス方向
        ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (W * H, 1)).astype(np.float64)
        
        # --- 7. レイトレーシングを実行 ---
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=False
        )
        
        # --- 8. 深度マップの生成 ---
        
        depth_map = np.full(W * H, min_xyz[2], dtype=np.float32) 
        hit_depths = locations[:, 2] 
        depth_map[index_ray] = hit_depths
        depth_map = depth_map.reshape((H, W))

        # --- 9. 深度値の正規化と
