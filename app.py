import streamlit as st
import trimesh
import numpy as np
import cv2
from io import BytesIO

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (Trimesh Raycasting)")
st.info("Pyrenderの問題を回避するため、trimeshのレイトレーシング機能で深度マップを生成します。")

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
            return

        # --- 4. 仮想カメラと投影の設定 ---
        
        # モデルの境界ボックスの対角線の長さ
        max_extents = mesh.extents.max()
        
        # カメラはZ軸方向からメッシュ全体が見えるように配置
        camera_distance = max_extents * 2.5
        
        # モデルの中心をカメラが向くように変換行列を構築
        camera_transform = np.array([
            [1, 0, 0, mesh.centroid[0]],
            [0, 1, 0, mesh.centroid[1]],
            [0, 0, 1, mesh.centroid[2] + camera_distance], # Z方向に離す
            [0, 0, 0, 1]
        ])

        # 視錐台の視野角 (FOV) を計算 (モデルが画面に収まるように)
        tan_half_fov = (max_extents / 2.0) / camera_distance
        fov = np.arctan(tan_half_fov) * 2

        # レイトレーシングのためのレイの始点と方向を手動で生成
        ray_origins, ray_directions = trimesh.util.create_perspective_rays(
            resolution=[W, H],
            transform=camera_transform,
            fov=fov
        )
        
        # レイトレーシングを実行
        # locations: 交点の座標, index_ray: どのレイがヒットしたか
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=False
        )
        
        # --- 5. 深度マップの生成 ---
        
        # 全てのピクセルに対応する深度配列を初期化 (遠い点を最大値として設定)
        max_dist = camera_distance * 3
        depth_map = np.full(W * H, max_dist, dtype=np.float32)

        # 交点までの距離を計算 (カメラ位置から交点まで)
        camera_origin = camera_transform[:3, 3]
        
        # ヒットしたレイの、カメラ位置から交点までのユークリッド距離
        distances = np.linalg.norm(locations - camera_origin, axis=1)
        
        # 深度マップに距離を書き込む
        depth_map[index_ray] = distances
        depth_map = depth_map.reshape((H, W))

        # --- 6. 深度値の正規化と画像化 (OpenCV) ---
        
        # 深度値を0-255の範囲に正規化 (8bitグレースケール)
        # NORM_MINMAX: 最小値と最大値を指定した範囲に正規化
        depth_normalized = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # PNGファイルとしてメモリに書き出し
        is_success, buffer = cv2.imencode(".png", depth_normalized)
        png_bytes = BytesIO(buffer.tobytes())

        # --- 7. 結果の表示とダウンロード ---
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
        st.info("コードまたはSTLファイルに問題がある可能性があります。")
