import streamlit as st
import trimesh
import numpy as np
import cv2 # 深度マップの正規化と画像化に必要
from io import BytesIO

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (Ortho-projection)")
st.info("パースを排除した正射影（上面図）による深度マップを生成します。")

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
        # これにより、カメラ設定がよりシンプルになる
        mesh.vertices -= mesh.centroid

        # --- 4. 仮想カメラと正射影の設定 ---
        
        # モデルのX/Y/Zの範囲を取得
        bounds = mesh.bounds # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        min_xyz = bounds[0]
        max_xyz = bounds[1]
        
        # モデルを画面全体に収めるためのビューポートサイズ
        # XとYの最大範囲を基準にする
        view_size_x = max_xyz[0] - min_xyz[0]
        view_size_y = max_xyz[1] - min_xyz[1]
        
        # 深度マップのアスペクト比に合わせてビューポートを調整
        aspect_ratio_mesh = view_size_x / view_size_y
        aspect_ratio_image = W / H

        if aspect_ratio_mesh > aspect_ratio_image:
            # メッシュが横長の場合、ビューポートの幅をメッシュに合わせ、高さを調整
            view_width = view_size_x * 1.2 # 少し余白を持たせる
            view_height = view_width / aspect_ratio_image
        else:
            # メッシュが縦長の場合、ビューポートの高さをメッシュに合わせ、幅を調整
            view_height = view_size_y * 1.2
            view_width = view_height * aspect_ratio_image

        # カメラの位置 (Z軸の非常に遠い位置から正対する)
        camera_origin_z = max_xyz[2] + view_size_y * 2 # モデルのZ軸最大値よりさらに上
        
        # 正射影なので、レイの始点は投影平面上をカバーし、方向はすべてZ軸マイナス方向
        
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
        
        # 深度マップを初期化 (ヒットしなかったピクセルはモデルの最大Z値)
        # 深度値をZ軸の値（高さ）として扱うため、Zmaxで初期化
        depth_map = np.full(W * H, max_xyz[2], dtype=np.float32) 

        # ヒットしたレイの、Z座標（深度）を取得
        # レイトレーシングで得られるlocationsは3D座標なので、そのZ成分が深度
        hit_depths = locations[:, 2] 
        
        # 深度マップにZ座標を書き込む
        depth_map[index_ray] = hit_depths
        
        # 深度マップを2D配列にリシェイプ
        depth_map = depth_map.reshape((H, W))

        # --- 8. 深度値の正規化と画像化 (OpenCV) ---
        
        # 深度値を0-255の範囲に正規化 (8bitグレースケール)
        # NORM_MINMAX を使用し、深度範囲を min_xyz[2] から max_xyz[2] に指定する
        # これにより、モデルの実際の高さ範囲で深度が正規化され、灰色の階調が出る
        
        # Z軸の最大範囲を計算 (深度マップの表現範囲)
        actual_z_range = max_xyz[2] - min_xyz[2]
        
        if actual_z_range <= 1e-6: # ほぼ平面の場合の対策
            depth_normalized = np.full((H, W), 128, dtype=np.uint8) # 中間グレーで埋める
        else:
            # 深度値をZminからZmaxの範囲で正規化
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # 近いものが黒、遠いものが白になるように反転
            depth_normalized = 255 - depth_normalized
            
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
        st.error(f"処理中にエラーが発生しました: {e}")
        st.info("STLファイルのデータ構造に問題がある可能性があります。")
