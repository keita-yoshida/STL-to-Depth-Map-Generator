import streamlit as st
import trimesh
import numpy as np
import cv2 # 深度マップの正規化と画像化に必要
from io import BytesIO

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (Final Stable Version)")
st.info("パースのない正射影で、Z値に基づいた正しい深度マップを生成します。")

# 深度マップの解像度
W, H = 512, 512

# --- 2. ファイルアップロード ---
uploaded_file = st.file_uploader("STLファイルをアップロードしてください", type=["stl"])

if uploaded_file is not None:
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    try:
        # --- 3. STLの読み込み (trimesh) ---
        mesh = trimesh.load_mesh(file_bytes, file_type='stl')
        
        if not isinstance(mesh, trimesh.Trimesh):
            st.error("アップロードされたファイルは有効なメッシュデータではありません。")
            st.stop() 

        # モデルの中心を原点に移動し、Z軸方向を上面に固定
        mesh.vertices -= mesh.centroid

        # --- 4. 仮想カメラと正射影の設定 ---
        
        # モデルのX/Y/Zの範囲を取得
        bounds = mesh.bounds 
        min_xyz = bounds[0]
        max_xyz = bounds[1]
        
        # モデルを画面全体に収めるためのビューポートサイズを決定
        view_size_x = max_xyz[0] - min_xyz[0]
        view_size_y = max_xyz[1] - min_xyz[1]
        
        aspect_ratio_mesh = view_size_x / view_size_y
        aspect_ratio_image = W / H

        # ビューポートのサイズ調整
        if aspect_ratio_mesh > aspect_ratio_image:
            view_width = view_size_x * 1.2 
            view_height = view_width / aspect_ratio_image
        else:
            view_height = view_size_y * 1.2
            view_width = view_height * aspect_ratio_image

        # カメラの位置 (Z軸の非常に遠い位置から正対する)
        camera_origin_z = max_xyz[2] + view_size_y * 2 
        
        # --- 5. レイトレーシングのためのレイを生成 (正射影) ---
        
        # ピクセルグリッドの座標を生成 (XとYの範囲をカバー)
        x_coords = np.linspace(-view_width / 2, view_width / 2, W)
        y_coords = np.linspace(-view_height / 2, view_height / 2, H)
        
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # レイの始点は投影平面上の各点と、Z軸上のカメラ位置
        ray_origins = np.stack((X.flatten(), Y.flatten(), np.full(W * H, camera_origin_z)), axis=1).astype(np.float64)
        
        # レイの方向は
