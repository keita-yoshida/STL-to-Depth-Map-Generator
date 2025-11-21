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
        
        # レイの方向は全てZ軸マイナス方向（[0, 0, -1]）
        ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (W * H, 1)).astype(np.float64)
        
        # --- 6. レイトレーシングを実行 ---
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=False
        )
        
        # --- 7. 深度マップの生成 ---
        
        # 深度マップを初期化 (ヒットしなかったピクセルはモデルのZ軸最低値で初期化)
        depth_map = np.full(W * H, min_xyz[2], dtype=np.float32) 

        # ヒットしたレイの、Z座標（深度）を取得
        hit_depths = locations[:, 2] 
        
        # 深度マップにZ座標を書き込む
        depth_map[index_ray] = hit_depths
        
        # 深度マップを2D配列にリシェイプ
        depth_map = depth_map.reshape((H, W))

        # --- 8. 深度値の正規化と画像化 (OpenCV) ---
        
        actual_z_range = max_xyz[2] - min_xyz[2]
        
        if actual_z_range <= 1e-6: # ほぼ平面の場合の対策
            depth_normalized = np.full((H, W), 128, dtype=np.uint8) 
        else:
            # 深度値を0-255の範囲に正規化
            depth_normalized = cv2.normalize(
                src=depth_map, 
                dst=None, 
                alpha=0,  
                beta=255, 
                norm_type=cv2.NORM_MINMAX, 
                dtype=cv2.CV_8U 
            )
            
        # PNGファイルとしてメモリに書き出し
        is_success, buffer = cv2.imencode(".png", depth_normalized)
        png_bytes = BytesIO(buffer.tobytes())

        # --- 9. 結果の表示とダウンロード ---
        st.subheader("生成された上面図深度マップ（正射影）")
        st.image(png_bytes, caption="Depth Map (Z値が低い: 黒, Z値が高い: 白)")
        
        st.download_button(
            label="深度マップ (.png) をダウンロード",
            data=png_bytes,
            file_name="depth_map_ortho.png",
            mime="image/png"
        )

    except Exception as e:
        # if uploaded_file is not None の直後の try に対応する except ブロック
        st.error(f"処理中にエラーが発生しました: {e}")
        st.info("STLファイルのデータ構造、またはデプロイ環境に問題がある可能性があります。")
