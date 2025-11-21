import streamlit as st
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO

# 以前のOpenGL関連のインポートと環境変数設定は全て削除（pyrenderが不要になったため）
# import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# ... などは削除

# --- 1. STLファイルアップロード ---
st.title("STL to Depth Map Generator (Trimesh Only)")
uploaded_file = st.file_uploader("STLファイルをアップロードしてください", type=["stl"])

if uploaded_file is not None:
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    try:
        # --- 2. STLの読み込み (trimesh) ---
        mesh = trimesh.load_mesh(file_bytes, file_type='stl')
        
        # モデルがカメラの視野に入るように正規化（必須ではないが推奨）
        T = np.eye(4)
        T[:3, 3] = -mesh.centroid
        mesh.apply_transform(T)
        
        # --- 3. 深度マップのシミュレーション (TrimeshのRaycastingを使用) ---
        # 深度マップの解像度
        W, H = 512, 512
        
        # 仮想カメラと投影の設定 (簡単な正面投影)
        # カメラはZ軸方向から見下ろす
        camera_transform = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, mesh.extents.max() * 2],  # メッシュから十分離れた位置
            [0, 0, 0, 1]
        ])
        
        # ビューポートの設定
        resolution = (W, H)
        fov = np.arctan(mesh.extents.max() / (mesh.extents.max() * 2)) * 2 # フィールド・オブ・ビュー

        # 深度マップの取得 (Trimeshの内部機能を使用)
        # ray_origins: スクリーンに対応するレイの始点
        # ray_directions: スクリーンに対応するレイの方向
        
        # ここでは、Trimeshのray.vpt_ray_generatorを使用して、より制御されたビューを取得します。
        
        # レイトレーシングのためのレイを生成
        rays = mesh.ray.camera_rays(
            camera_transform, 
            resolution=resolution, 
            fov=fov
        )

        # レイトレーシングを実行
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            rays[0], rays[1], multiple_hits=False
        )

        # 全てのピクセルに対応する深度配列を初期化 (遠い点を最大値として設定)
        max_dist = mesh.extents.max() * 3
        depth_map = np.full(W * H, max_dist, dtype=np.float32)

        # 交点までの距離を計算 (カメラ位置から交点まで)
        # カメラ位置
        camera_origin = camera_transform[:3, 3]
        
        # 交点までの距離
        distances = np.linalg.norm(locations - camera_origin, axis=1)
        
        # 深度マップに距離を書き込む
        depth_map[index_ray] = distances
        depth_map = depth_map.reshape((H, W))

        # --- 4. 深度値の正規化と画像化 (OpenCV) ---
        # 深度値を0-255の範囲に正規化 (8bitグレースケール)
        depth_normalized = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # PNGファイルとしてメモリに書き出し
        is_success, buffer = cv2.imencode(".png", depth_normalized)
        png_bytes = BytesIO(buffer.tobytes())

        # --- 5. 結果の表示とダウンロード ---
        st.subheader("生成された深度マップ")
        st.image(png_bytes, caption="Depth Map (近: 黒, 遠: 白)")
        
        st.download_button(
            label="深度マップ (.png) をダウンロード",
            data=png_bytes,
            file_name="depth_map.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
        st.info("Pyrenderの問題を回避するため、trimeshとレイトレーシングによる深度マップ生成に切り替えました。")
