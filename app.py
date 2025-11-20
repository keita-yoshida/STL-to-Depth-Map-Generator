import streamlit as st
import trimesh
import pyrender # レンダリングに必要
import numpy as np
import cv2
from io import BytesIO

# --- 1. STLファイルアップロード ---
st.title("STL to Depth Map Generator")
uploaded_file = st.file_uploader("STLファイルをアップロードしてください", type=["stl"])

if uploaded_file is not None:
    # ファイルをメモリ（BytesIO）から読み込む
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    try:
        # --- 2. STLの読み込み (trimesh) ---
        mesh = trimesh.load_mesh(file_bytes, file_type='stl')

        # --- 3. シーンとメッシュの設定 ---
        scene = pyrender.Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]))
        
        # trimeshのメッシュをpyrenderのメッシュに変換してシーンに追加
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(pyrender_mesh)

        # --- 4. レンダラーとカメラのセットアップ ---
        # 深度マップの解像度
        W, H = 512, 512
        r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        
        # モデル全体が映るようにカメラを自動設定 (trimeshの機能を使用)
        camera_pose = trimesh.viewer.trackball.around_bounding_box(mesh)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=W/H)
        scene.add(camera, pose=camera_pose)

        # --- 5. 深度マップのレンダリング ---
        # レンダリングを実行。color_rgbは不要なので無視
        _, depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY) 
        
        r.delete() # リソースの解放

        # --- 6. 深度値の正規化と画像化 (OpenCV) ---
        # 深度値を0-255の範囲に正規化 (8bitグレースケール)
        depth_normalized = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # PNGファイルとしてメモリに書き出し
        is_success, buffer = cv2.imencode(".png", depth_normalized)
        png_bytes = BytesIO(buffer)

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
