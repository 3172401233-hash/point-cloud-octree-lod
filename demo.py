"""端到端演示：非均匀密度点云的八叉树 LOD。

运行方式::

    python demo.py

场景包含一个高密度簇和一个低密度簇，脚本打印：
- 构建规模与密/疏区域的点间距（密度）差异；
- 不同屏幕误差阈值与节点预算下的 LOD 前沿、预算占用；
- 原始点云相对 LOD 代表点的 Hausdorff / RMS / 屏幕空间轮廓误差。
"""

import numpy as np

from pointcloud_lod import (
    Camera,
    Octree,
    budget_usage,
    lod_point_set,
    rms_nearest_distance,
    screen_space_hausdorff,
    select_lod,
    symmetric_hausdorff,
)
from pointcloud_lod.lod import region_breakdown


def main() -> None:
    rng = np.random.default_rng(2026)

    # 密集簇：2000 点挤在 2x2x2 盒子；稀疏簇：64 点占据同样大小的盒子。
    dense = rng.uniform([0.0, -1.0, -1.0], [2.0, 1.0, 1.0], size=(2000, 3))
    sparse = rng.uniform([6.0, -1.0, -1.0], [8.0, 1.0, 1.0], size=(64, 3))
    points = np.vstack([dense, sparse])
    colors = rng.uniform(0.0, 255.0, (points.shape[0], 3))
    normals = rng.normal(size=(points.shape[0], 3))
    intensities = rng.random(points.shape[0])

    tree = Octree(
        points, colors, normals, intensities,
        max_points_per_leaf=8, max_depth=18,
    )
    print(f"构建完成：{points.shape[0]} 点 -> {len(tree)} 个八叉树节点，"
          f"叶子 {sum(1 for n in tree.nodes if n.is_leaf)} 个")

    camera = Camera(
        position=np.array([4.0, -12.0, 0.0]),
        target=np.array([4.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0]),
        fov_y=np.radians(60.0),
        aspect=1.7,
        viewport_height=1080,
    )

    print("\n== 密度差异（阈值 100 px，无预算限制） ==")
    result = select_lod(tree, camera, 100.0, 10**9, enable_occlusion=False)
    dense_stat = region_breakdown(tree, result, lambda p: p[0] < 3.0)
    sparse_stat = region_breakdown(tree, result, lambda p: p[0] > 5.0)
    print(f"密集区：{dense_stat['points']:5d} 点 / {dense_stat['nodes']:4d} 节点，"
          f"平均点间距 {dense_stat['mean_spacing']:.4f}")
    print(f"稀疏区：{sparse_stat['points']:5d} 点 / {sparse_stat['nodes']:4d} 节点，"
          f"平均点间距 {sparse_stat['mean_spacing']:.4f}")
    print(f"每千点节点数：密集 {dense_stat['nodes'] / dense_stat['points'] * 1000:.1f}"
          f" vs 稀疏 {sparse_stat['nodes'] / sparse_stat['points'] * 1000:.1f}")

    print("\n== 屏幕误差阈值对前沿规模的影响 ==")
    for threshold in (512.0, 256.0, 128.0, 64.0):
        r = select_lod(tree, camera, threshold, 10**9, enable_occlusion=False)
        print(f"阈值 {threshold:6.1f}px -> 前沿 {len(r.node_ids):4d} 节点，"
              f"实际最大 SSE {r.max_sse:8.2f}px，均值 {r.mean_sse:8.2f}px")

    print("\n== 节点预算强制收敛（阈值 64 px） ==")
    for budget in (512, 128, 32, 8, 1):
        r = select_lod(tree, camera, 64.0, budget, enable_occlusion=False)
        usage = budget_usage(r)
        print(f"预算 {budget:4d} -> 选中 {usage['selected_nodes']:4d} 节点 "
              f"({usage['budget_ratio'] * 100:5.1f}% 占用)，"
              f"合并 {usage['forced_merges']:3d} 次，"
              f"覆盖 {usage['represented_points']} 原始点")

    print("\n== 轮廓误差度量（预算 128，阈值 64 px） ==")
    r = select_lod(tree, camera, 64.0, 128, enable_occlusion=False)
    lod_points = lod_point_set(tree, r)
    print(f"LOD 代表点数：{lod_points.shape[0]}（原始 {points.shape[0]} 点）")
    print(f"对称 Hausdorff 距离（世界单位，最坏轮廓偏差）："
          f"{symmetric_hausdorff(points, lod_points):.4f}")
    print(f"RMS 最近邻距离（世界单位，平均轮廓偏差）："
          f"{rms_nearest_distance(points, lod_points):.4f}")
    print(f"屏幕空间 Hausdorff 误差（像素）："
          f"{screen_space_hausdorff(camera, points, lod_points):.2f}")

    print("\n== 遮挡与视锥统计 ==")
    r_occ = select_lod(tree, camera, 64.0, 10**9, enable_occlusion=True)
    print(f"启用遮挡：前沿 {len(r_occ.node_ids)} 节点，"
          f"视锥剔除 {r_occ.frustum_culled}，遮挡剔除 {r_occ.occluded_nodes}")


if __name__ == "__main__":
    main()
