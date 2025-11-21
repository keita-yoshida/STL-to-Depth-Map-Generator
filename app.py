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
# 現在の回転角度をセッションステートで記憶
if 'rotation_angle' not in st.session_state:
    st.session_state['rotation_angle'] = 0

def rotate_model(degrees):
    """現在の角度に指定された角度を追加するコールバック関数"""
    # 角度を0-359度の間に保つ
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
    
    # 全体の処理を try-except で囲む (構文エラー対策)
    try:
        # --- 4. STLの読み込みと前処理 (trimesh) ---
        mesh = trimesh.load_mesh(file_bytes, file_type='stl')
        
        if not isinstance(mesh, trimesh.Trimesh):
            st.error("アップロードされたファイルは有効なメッシュデータではありません。")
            st.stop() 

        # モデルの中心を原点に移動
        mesh.vertices -= mesh.centroid

        # 🔥 回転処理の適用
        angle_rad = np.radians(st.session_state['rotation_angle'])
        # Z軸を中心に回転させる変換行列を作成し、メッシュに適用
        rotation_matrix = trimesh.transformations.rotation_matrix(angle_rad, [0, 0, 1])
        mesh.apply_transform(rotation_matrix)

        # --- 5. 仮想カメラと正射影の設定 ---
        
        # モデルのX/Y/Zの範囲を取得 (回転後のboundsを使用)
        bounds = mesh.bounds 
        min_xyz = bounds[0]
        max_xyz = bounds[1]
        
        # モデルを画面全体に収めるためのビューポートサイズを決定
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

        # カメラの位置 (Z軸の非常に遠い位置から正対する)
        camera_origin_z = max_xyz[2] + view_size_y * 2 
        
        # --- 6. レイトレーシングのためのレイを生成 (正射影) ---
        
        # ピクセルグリッドの座標を生成 (XとYの範囲をカバー)
        x_coords = np.linspace(-view_width / 2, view_width / 2, W)
        y_coords = np.linspace(-view_height / 2, view_height / 2, H)
        
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Line 93: レイの始点は投影平面上の各点と、Z軸上のカメラ位置
        # 括弧は完全に
