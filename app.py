import streamlit as st
import trimesh
import numpy as np
import cv2 
from io import BytesIO

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (Pro Version)")
st.info("正射影深度マップ生成。回転、解像度、余白を自由にカスタマイズ可能です。")

# --- 2. セッションステートの初期化と回転ボタン ---

if 'yaw_angle' not in st.session_state:
    st.session_state['yaw_angle'] = 0
if 'pitch_angle' not in st.session_state:
    st.session_state['pitch_angle'] = 0

def rotate_yaw(degrees):
    st.session_state['yaw_angle'] = (st.session_state['yaw_angle'] + degrees) % 360

def rotate_pitch(degrees):
    st.session_state['pitch_angle'] = (st.session_state['pitch_angle'] + degrees) % 360

st.sidebar.subheader("モデル回転 (十字キー)")
# 十字キーレイアウト
col_p_up, col_p_mid, col_p_down = st.sidebar.columns([1, 1, 1])
with col_p_mid:
    st.button("上へ 90°", on_click=rotate_pitch, args=(-90,), use_container_width=True, key="pitch_up")

col_y_left, col_y_mid, col_y_right = st.sidebar.columns([1, 1, 1])
with col_y_left:
    st.button("左へ 90°", on_click=rotate_yaw, args=(90,), use_container_width=True, key="yaw_left")
with col_y_right:
    st.button("右へ 90°", on_click=rotate_yaw, args=(-90,), use_container_width=True, key="yaw_right")

col_p_up_2, col_p_mid_2, col_p_down_2 = st.sidebar.columns([1, 1, 1])
with col_p_mid_2:
    st.button("下へ 90°", on_click=rotate_pitch, args=(90,), use_container_width=True, key="pitch_down")

# --- 3. 詳細設定 (解像度・余白) ---
st.sidebar.markdown("---")
st.sidebar.subheader("出力設定")

# 解像度設定
W = st.sidebar.number_input("出力幅 (px)", min_value=100, max_value=4096, value=512, step=128)
H = st.sidebar.number_input("出力高さ (px)", min_value=100, max_value=4096, value=512, step=128)

# 🔥 修正点: 余白の設定スライダーを追加
margin_percent = st.sidebar.slider("モデル周囲の余白 (%)", min_value=0, max_value=100, value=10, step=1)
# 1.0 (0%) ~ 2.0 (100%) の係数に変換
padding_factor = 1.0 + (margin_percent / 100.0)

st.sidebar.markdown("---")
st.sidebar.caption(f"Y軸角度: {st.session_state['yaw_angle']}° / X軸角度: {st.session_state['pitch_angle']}°")


# --- 4. ファイルアップロード ---
uploaded_file = st.file_uploader("STLファイルをアップロードしてください", type=["stl"])

if uploaded_file is not None:
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    try:
        # メッシュの読み込み
        mesh = trimesh.load_mesh(file_bytes, file_type='stl')
        if not isinstance(mesh, trimesh.Trimesh):
            st.error("有効なメッシュデータではありません。")
            st.stop() 

        mesh.vertices -= mesh.centroid

        # 回転の適用
        yaw_rad = np.radians(st.session_state['yaw_angle'])
        pitch_rad = np.radians(st.session_state['pitch_angle'])
        yaw_matrix = trimesh.transformations.rotation_matrix(yaw_rad, [0, 1, 0])
        pitch_matrix = trimesh.transformations.rotation_matrix(pitch_rad, [1, 0, 0])
        combined_matrix = trimesh.transformations.concatenate_matrices(pitch_matrix, yaw_matrix)
        mesh.apply_transform(combined_matrix)

    except Exception as e:
        st.error(f"STL処理エラー: {e}")
        st.stop()

    # --- 5. ビューポート計算 (余白設定を適用) ---
    bounds = mesh.bounds 
    min_xyz, max_xyz = bounds[0], bounds[1]
    view_size_x = max_xyz[0] - min_xyz[0]
    view_size_y = max_xyz[1] - min_xyz[1]
    
    aspect_ratio_mesh = view_size_x / view_size_y
    aspect_ratio_image = W / H

    # 🔥 padding_factor を使用してビュー幅を計算
    if aspect_ratio_mesh > aspect_ratio_image:
        view_width = view_size_x * padding_factor
        view_height = view_width / aspect_ratio_image
    else:
        view_height = view_size_y * padding_factor
        view_width = view_height * aspect_ratio_image

    camera_origin_z = max_xyz[2] + view_size_y * 2 
    
    # --- 6. レイ生成と実行 ---
    x_coords = np.linspace(-view_width / 2, view_width / 2, W)
    y_coords = np.linspace(-view_height / 2, view_height / 2, H)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    origins_stack = np.stack((X.flatten(), Y.flatten(), np.full(W * H, camera_origin_z)), axis=1)
    ray_origins = origins_stack.astype(np.float64)
    ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (W * H, 1)).astype(np.float64)
    
    try:
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=False
        )
    except Exception as e:
        st.error(f"計算エラー: {e}")
        st.stop()
    
    # --- 7. 深度マップ生成 ---
    # 背景（ヒットなし）はモデルの底と同じ深さにする
    depth_map = np.full(W * H, min_xyz[2], dtype=np.float32) 
    if len(locations) > 0:
        depth_map[index_ray] = locations[:, 2]
    
    depth_map = depth_map.reshape((H, W))

    actual_z_range = max_xyz[2] - min_xyz[2]
    if actual_z_range <= 1e-6:
        depth_normalized = np.full((H, W), 128, dtype=np.uint8) 
    else:
        depth_normalized = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    is_success, buffer = cv2.imencode(".png", depth_normalized)
    png_bytes = BytesIO(buffer.tobytes())

    # --- 8. 表示とダウンロード ---
    st.subheader("プレビュー")
    st.image(png_bytes, caption=f"解像度: {W}x{H} / 余白: {margin_percent}%")
    
    st.download_button(
        label="深度マップ (.png) をダウンロード",
        data=png_bytes,
        file_name=f"depth_map_{W}x{H}_m{margin_percent}.png",
        mime="image/png"
    )
