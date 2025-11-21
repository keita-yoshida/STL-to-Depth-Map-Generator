# ... (app.pyの続き)

        # --- 3. 深度マップのシミュレーション (TrimeshのRaycastingを使用) ---
        # 深度マップの解像度
W, H = 512, 512
        # 仮想カメラと投影の設定 (簡単な正面投影)
        # Bounding boxの対角線の長さ
max_extents = mesh.extents.max()
        # カメラはZ軸方向からメッシュ全体が見えるように配置
        camera_distance = max_extents * 2.0
        
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
        # camera_rays の代わりに trimesh.util.create_perspective_rays を使用
        ray_origins, ray_directions = trimesh.util.create_perspective_rays(
            resolution=[W, H],
            transform=camera_transform,
            fov=fov
        )
        
        # レイトレーシングを実行
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins, ray_directions, multiple_hits=False
        )
        
        # ... (残りのコードは同じ)
        # 全てのピクセルに対応する深度配列を初期化 (遠い点を最大値として設定)
        max_dist = camera_distance * 2
        depth_map = np.full(W * H, max_dist, dtype=np.float32)

        # 交点までの距離を計算 (カメラ位置から交点まで)
        camera_origin = camera_transform[:3, 3]
        
        distances = np.linalg.norm(locations - camera_origin, axis=1)
        
        # 深度マップに距離を書き込む
        depth_map[index_ray] = distances
        depth_map = depth_map.reshape((H, W))
        # ... (続く)
