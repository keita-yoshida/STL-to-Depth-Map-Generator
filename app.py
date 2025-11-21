import trimesh

# ====== 設定 ======
INPUT_FILE = "model.stl"        # 読み込みたいSTL/OBJ
OUTPUT_FILE = "top_view.png"    # 出力画像ファイル名
RESOLUTION = (1024, 1024)       # 出力画像サイズ
# ===================

# メッシュを読み込む
mesh = trimesh.load(INPUT_FILE)

# シーンを作成
scene = mesh.scene()

# カメラを「真上」（+Z方向）に向ける
# 正射影（パースなし）カメラ
camera = trimesh.scene.cameras.OrthographicCamera(
    resolution=RESOLUTION,
    zfar=1000,
    znear=0.01
)

# カメラを mesh のバウンディングボックスの上に配置
bbox = mesh.bounds
center = bbox.mean(axis=0)
extent = bbox[1] - bbox[0]

# 上方向 (Z軸) から見下ろす位置
camera_position = [center[0], center[1], center[2] + extent[2] * 2]

# カメラ変換行列を設定（Z軸方向から見下ろす）
camera_transform = trimesh.geometry.look_at(
    points=[camera_position],
    center=center,
    up=[0, 1, 0]
)[0]

# カメラをシーンに設定
scene.camera = camera
scene.camera_transform = camera_transform

# 画像として保存（PNGバイトが返る）
img = scene.save_image(
    resolution=RESOLUTION,
    visible=True
)

# ファイルへ書き込み
with open(OUTPUT_FILE, "wb") as f:
    f.write(img)

print("上面ビューを出力しました:", OUTPUT_FILE)
