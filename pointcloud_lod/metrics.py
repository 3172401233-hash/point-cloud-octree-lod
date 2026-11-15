"""误差度量：轮廓/Hausdorff 误差、LOD 近似集构建与预算统计。

度量口径
========
- **几何误差上界（每节点）**：节点 tight 包围盒对角线。盒内任意点到
  代表点的距离都不超过该值，因此是确定性上界，不依赖采样。
- **RMS 误差（每节点）**：节点内各点到质心距离的均方根，描述统计离散度。
- **有向 Hausdorff 距离**：``max_{p in A} min_{q in B} |p - q|``，
  衡量简化点集 B 相对原始点集 A 的最坏轮廓偏差；对称版本取两个方向的最大。
- **屏幕空间 Hausdorff 误差**：把最近邻世界距离按相机投影换算成像素，
  与 LOD 选择使用的 SSE 口径一致，便于和阈值直接比较。
"""

from __future__ import annotations

import numpy as np

from .camera import Camera
from .lod import SelectionResult
from .octree import Octree


def lod_point_set(tree: Octree, result: SelectionResult) -> np.ndarray:
    """用前沿节点代表点构成简化点集，形状 (K, 3)。"""
    return result.representative_points()


def directed_hausdorff(reference: np.ndarray, simplified: np.ndarray) -> float:
    """有向 Hausdorff 距离：reference 中每个点到 simplified 最近距离的最大值。"""
    reference = np.asarray(reference, dtype=np.float64)
    simplified = np.asarray(simplified, dtype=np.float64)
    if reference.size == 0 or simplified.size == 0:
        raise ValueError("输入点集不能为空")
    block = max(1, min(4096, reference.shape[0]))
    worst = 0.0
    for lo in range(0, reference.shape[0], block):
        chunk = reference[lo:lo + block]
        diff = chunk[:, None, :] - simplified[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        worst = max(worst, float(np.sqrt(dist2.min(axis=1)).max()))
    return worst


def symmetric_hausdorff(reference: np.ndarray, simplified: np.ndarray) -> float:
    """对称 Hausdorff 距离（两个方向的最大值）。"""
    return max(
        directed_hausdorff(reference, simplified),
        directed_hausdorff(simplified, reference),
    )


def rms_nearest_distance(reference: np.ndarray, simplified: np.ndarray) -> float:
    """reference 各点到 simplified 最近距离的均方根（平均轮廓误差）。"""
    reference = np.asarray(reference, dtype=np.float64)
    simplified = np.asarray(simplified, dtype=np.float64)
    block = max(1, min(4096, reference.shape[0]))
    total = 0.0
    for lo in range(0, reference.shape[0], block):
        chunk = reference[lo:lo + block]
        diff = chunk[:, None, :] - simplified[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        total += float(dist2.min(axis=1).sum())
    return float(np.sqrt(total / reference.shape[0]))


def screen_space_hausdorff(
    camera: Camera,
    reference: np.ndarray,
    simplified: np.ndarray,
) -> float:
    """屏幕空间 Hausdorff 误差（像素）。

    对 reference 中每个点取到 simplified 的最近世界距离 d，
    再按该点视空间深度 z 投影为像素::

        pixel = d * (viewport_height / 2) / tan(fov_y / 2) / z

    返回所有点中的最大值（最坏轮廓偏差，像素单位）。
    """
    reference = np.asarray(reference, dtype=np.float64)
    simplified = np.asarray(simplified, dtype=np.float64)
    rel = reference - camera.position
    depths = np.maximum(rel @ camera.forward, camera.near)
    block = max(1, min(4096, reference.shape[0]))
    pixel_scale = (camera.viewport_height * 0.5) / np.tan(camera.fov_y * 0.5)
    worst = 0.0
    for lo in range(0, reference.shape[0], block):
        chunk = reference[lo:lo + block]
        diff = chunk[:, None, :] - simplified[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        nearest = np.sqrt(dist2.min(axis=1))
        pixels = nearest * pixel_scale / depths[lo:lo + block]
        worst = max(worst, float(pixels.max()))
    return worst


def budget_usage(result: SelectionResult) -> dict:
    """预算占用统计：节点数、代表点数、覆盖原始点数、占预算比例。"""
    return {
        "selected_nodes": len(result.node_ids),
        "node_budget": result.node_budget,
        "budget_ratio": len(result.node_ids) / result.node_budget,
        "represented_points": result.total_points,
        "max_sse_px": result.max_sse,
        "mean_sse_px": result.mean_sse,
        "frustum_culled": result.frustum_culled,
        "occluded": result.occluded_nodes,
        "forced_merges": result.forced_merges,
    }
