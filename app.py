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
    # 🔥 修正: オブジェクトを時計回り (+90) に回転させ、カメラが左に回り込んだように見せる
    st.button("左へ 90°", on_click=rotate_yaw, args=(90,), use_container_width=True, key="yaw_left", help="Z軸周りに回転 (カメラが左に回り込む)")
with col_y_right:
    # 🔥 修正: オブジェクトを反時計回り (-90) に回転させ、カメラが右に回り込んだように見せる
    st.button("右へ 90°", on_click=rotate_yaw, args=(-90,), use_container_width=True, key="yaw_right", help="Z軸周りに回転 (カメラが右に回り込む)")

# 3. 上下回転（下ボタン）: 中央に配置
col_p_up_2, col_p_mid_2, col_p_down_2 = st.sidebar.columns([1, 1, 1])
with col_p_mid_2:
    st.button("下へ 90°", on_click=rotate_pitch, args=(90,), use_container_width=True, key="pitch_down", help="X軸周りに回転 (モデルが下へ傾く)")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Z軸角度 (左右): {st.session_state['yaw_angle']}°**")
st.sidebar.markdown(f"**X軸角度 (上下): {st.session_state['pitch_angle']}°**")
st.sidebar.markdown("---")


# --- 3. ファイルアップロード ---
uploaded_file = st.file_uploader("STLファイルをアップロードしてください", type=["stl"])

if uploaded_file is not None:
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    try:
        # STLの読み込みとメッシュの前処理
        mesh = trimesh.load_mesh(file_bytes, file_type='stl')
        
        if not isinstance(mesh, trimesh.Trimesh):
            st.error("アップロードされたファイルは有効なメッシュデータではありません。")
            st.stop() 

        mesh.vertices -= mesh.centroid

        # 回転処理の適用
        yaw_rad = np.radians(st.session_state['yaw_angle'])
        pitch_rad = np.radians(st.session_state['pitch_angle'])

        # Z軸回転行列 (左右)
        yaw_matrix = trimesh.transformations.rotation_matrix(yaw_rad, [0, 0, 1])
        
        # X軸回転行列 (上下)
        pitch_matrix = trimesh.transformations.rotation_matrix(pitch_rad, [1, 0, 0])

        # 変換行列を合成 (先にピッチを適用してからヨーを適用)
        combined_matrix = trimesh.transformations.concatenate_matrices(pitch_matrix, yaw_matrix)
        
        # メッシュに適用
        mesh.apply_transform(combined_matrix)

    except Exception as e:
        st.error(f"STLファイルの読み込みまたは処理中にエラーが発生しました: {e}")
        st.info("ファイルが破損しているか、依存ライブラリの初期化に失敗している可能性があります。")
        st.stop()

    # --- 4. 仮想カメラと正射影の設定 ---
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
    
    # --- 5. レイトレーシングのためのレイを生成 ---
    x_coords = np.linspace(-view_width / 2, view_width / 2, W)
    y_coords = np.linspace(-view_height / 2, view_height / 2, H)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    origins_stack = np.stack((X.flatten(), Y.flatten(), np.full(W * H, camera_origin_z)), axis=1)
    ray_origins = origins_stack.astype(np.float64)
    ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (W * H, 1)).astype(np.float64)
    
    # --- 6. レイトレーシングを実行 ---
    try:
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=False
        )
    except Exception as e:
        st.error(f"レイトレーシング中にエラーが発生しました: {e}")
        st.info("STLモデルの構造が複雑すぎるか、レイトレーシング機能に問題があります。")
        st.stop()
    
    # --- 7. 深度マップの生成と表示 ---
    depth_map = np.full(W * H, min_xyz[2], dtype=np.float32) 
    hit_depths = locations[:, 2] 
    depth_map[index_ray] = hit_depths
    depth_map = depth_map.reshape((H, W))

    actual_z_range = max_xyz[2] - min_xyz[2]
    
    if actual_z_range <= 1e-6:
        depth_normalized = np.full((H, W), 128, dtype=np.uint8) 
    else:
        depth_normalized = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    # PNGファイルとしてメモリに書き出し
    is_success, buffer = cv2.imencode(".png", depth_normalized)
    png_bytes = BytesIO(buffer.tobytes())

    # --- 8. 結果の表示とダウンロード ---
    st.subheader("生成された上面図深度マップ（正射影）")
    
    caption_text = f"Depth Map (Z軸: {st.session_state['yaw_angle']}°, X軸: {st.session_state['pitch_angle']}°) - Z値が低い: 黒, Z値が高い: 白"
    st.image(png_bytes, caption=caption_text)
    
    st.download_button(
        label="深度マップ (.png) をダウンロード",
        data=png_bytes,
        file_name=f"depth_map_ortho_z{st.session_state['yaw_angle']}_x{st.session_state['pitch_angle']}.png",
        mime="image/png"
    )
