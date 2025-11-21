import streamlit as st
import trimesh
import numpy as np
from io import BytesIO
from PIL import Image # 画像生成のためにPillowを使用

# --- 1. アプリケーション設定 ---
st.title("STL to Depth Map Generator (Simple)")
st.info("外部依存を最小限に抑えた、trimeshとnumpyのみによる処理です。")

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
            st.stop() # st.stop()を使用

        # モデルのZ座標を反転し、最小値を0に調整 (上面図の深度データとして扱いやすくする)
        # Z座標が最も高い（上面）部分が、深度マップで最も暗くなるようにする
        
        # Z座標の最大値と最小値を取得
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()
        
        # モデルをZ軸で正規化し、Z=0をベースにする
        normalized_vertices = mesh.vertices.copy()
        normalized_vertices[:, 2] -= z_min
        
        z_range = z_max - z_min
        
        if z_range == 0:
            st.error("モデルが2次元（平面）であるため、深度マップを生成できません。")
            st.stop()
            
        # --- 4. 2D投影の実行 (Z軸方向の投影) ---
        
        # 頂点をX-Y平面に投影し、ピクセルグリッドにマッピングする
        
        # モデルのXとYの範囲を計算
        x_min, y_min = mesh.vertices[:, 0:2].min(axis=0)
        x_max, y_max = mesh.vertices[:, 0:2].max(axis=0)
        
        # ピクセルあたりのスケールを計算 (アスペクト比を維持)
        x_span = x_max - x_min
        y_span = y_max - y_min
        
        # グリッドの初期化 (深度データとして、最も遠い深度で初期化)
        # 深度を反転させるため、最も遠いZ値（z_range）で初期化
        depth_grid = np.full((H, W), z_range, dtype=np.float32) 
        
        # 頂点の座標をピクセル座標に変換
        # x_coords: 0 から W-1, y_coords: 0 から H-1
        x_coords = ((normalized_vertices[:, 0] - x_min) / x_span * (W - 1)).astype(int)
        y_coords = ((normalized_vertices[:, 1] - y_min) / y_span * (H - 1)).astype(int)
        
        # Z値（深度）を取得
        z_values = normalized_vertices[:, 2]

        # グリッドに深度を書き込む
        # ここでは単純に頂点のみをプロットしていますが、これが最もシンプルな上面投影です
        for x, y, z in zip(x_coords, y_coords, z_values):
            # 同じピクセルに複数の頂点がある場合、最も近いもの（Z値が小さいもの）を採用
            # しかし、上面図なので、ここでは単純にZ値を書き込む
            # Z値を255スケールに正規化して格納
            depth_grid[y, x] = min(depth_grid[y, x], z)
        
        # --- 5. 画像化 ---
        
        # 深度を0-255のグレースケールに正規化
        # 深度が浅い（Z値が大きい）ほど明るく（255）なるように反転させる
        normalized_depth = (255 * (depth_grid / z_range)).astype(np.uint8)
        
        # PIL (Pillow) を使用して画像に変換
        img = Image.fromarray(normalized_depth).transpose(Image.FLIP_TOP_BOTTOM) # Y軸を反転して正しい向きに
        
        # PNGファイルとしてメモリに書き出し
        png_bytes = BytesIO()
        img.save(png_bytes, format='PNG')
        png_bytes.seek(0)


        # --- 6. 結果の表示とダウンロード ---
        st.subheader("生成された上面投影深度マップ")
        st.image(png_bytes, caption="Depth Map (Z値が深い: 黒, Z値が浅い: 白)")
        
        st.download_button(
            label="深度マップ (.png) をダウンロード",
            data=png_bytes,
            file_name="depth_map_simple.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"処理中に予期せぬエラーが発生しました: {e}")
