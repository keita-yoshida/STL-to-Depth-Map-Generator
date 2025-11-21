import streamlit as st
import trimesh
import numpy as np
import cv2 # 深度マップの正規化と画像化に必要
from io import BytesIO

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (Stable Raycasting)")
st.info("Trimeshのバージョン依存性を排除した、純粋なNumPyベースのレイトレーシングです。")

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
            st.stop() # Streamlitの実行を停止

        # モデルの中心を原点に移動し、Z軸方向を上面に固定
        mesh.vertices -= mesh.centroid

        # --- 4. 仮想カメラと投影の設定 (NumPyのみ) ---
        
        # モデルのX/Y方向の最大寸法
        max_xy_extents = np.max(mesh.extents[:2])
        
        # カメラはZ軸方向からメッシュ全体が見えるように配置
        camera_distance = max_xy_extents * 2.5
        
        # カメラの原点 (Z軸上から見下ろす)
        camera_origin = np.array([0.0, 0.0, camera_distance])
        
        # 投影平面 (深度マップのスクリーン) のサイズ
        # 視野の幅と高さをモデルの寸法に合わせて設定
        view_width = max_xy_extents * 1.5 
        view_height = max_xy_extents * 1.5 
        
        # --- 5. レイトレーシングのためのレイを生成 (NumPyのみ) ---
        
        # ピクセルグリッドの座標を生成
        x_indices = np.linspace(-view_width / 2, view_width / 2, W)
        y_indices = np.linspace(-view_height / 2, view_height / 2, H)
        
        # XとYのメッシュグリッドを作成
        X, Y = np.meshgrid(x_indices, y_indices)
        
        # 投影平面上のターゲットポイント（カメラが向かう方向）を計算
        target_points = np.stack((X.flatten(), Y.flatten(), np.zeros(W * H)), axis=1)
        
        # レイの始点は全てカメラの原点
        ray_origins = np.tile(camera_origin, (W * H, 1)).astype(np.float64)
        
        # レイの方向は、カメラの原点からターゲットポイントへ
        ray_directions = target_points - ray_origins
        
        # 方向ベクトルを正規化
        ray_directions /= np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
        
        # --- 6. レイトレーシングを実行 ---
        # locations: 交点の座標, index_ray: どのレイがヒットしたか
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=False
        )
        
        # --- 7. 深度マップの生成 ---
        
        # 全てのピクセルに対応する深度配列を初期化 (遠い点を最大値として設定)
        max_dist = camera_distance * 3
        depth_map = np.full(W * H, max_dist, dtype=np.float32)

        # ヒットしたレイの、カメラ位置から交点までのユークリッド距離
        distances = np.linalg.norm(locations - camera_origin, axis=1)
        
        # 深度マップに距離を書き込む
        # NOTE: Y軸の向きを反転させるため、H - 1 - index_ray // W などの操作が必要な場合があるが、
        #       OpenCVの正規化後にPillowで画像を反転させる方が簡単
        depth_map[index_ray] = distances
        depth_map = depth_map.reshape((H, W))

        # --- 8. 深度値の正規化と画像化 (OpenCV) ---
        
        # 深度値を0-255の範囲に正規化
        depth_normalized = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # PNGファイルとしてメモリに書き出し
        is_success, buffer = cv2.imencode(".png", depth_normalized)
        png_bytes = BytesIO(buffer.tobytes())

        # --- 9. 結果の表示とダウンロード ---
        st.subheader("生成された深度マップ")
        st.image(png_bytes, caption="Depth Map (近: 黒, 遠: 白)")
        
        st.download_button(
            label="深度マップ (.png) をダウンロード",
            data=png_bytes,
            file_name="depth_map_final.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
        st.info("依存関係は満たされていますが、STLファイルのデータ構造に問題がある可能性があります。")
