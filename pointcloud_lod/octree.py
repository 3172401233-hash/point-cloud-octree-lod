"""八叉树构建：Morton 排序细分、节点统计与属性聚合。

构建流程
--------
1. 计算根包围盒并把点量化到 21 bit 整数网格，计算 3D Morton 码；
2. 按 Morton 码全局排序，同一节点的点在排序数组中天然连续；
3. 递归细分：每个节点只保存排序数组中的区间 ``[start, end)``，
   不复制点数据，空八分体直接跳过（不会产生空节点）；
4. 自底向上聚合：保存包围盒、点数、代表属性（颜色/法线/强度均值）、
   局部点间距（密度）以及误差上界。

重复点（坐标完全相同）会落到同一个叶子，正常统计且不会产生空子节点。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import AABB, MAX_DEPTH, morton_encode, octant_at_depth, quantize_points

_EPS = 1e-12


@dataclass
class NodeStats:
    """单个八叉树节点的聚合统计。"""

    node_id: int
    parent_id: Optional[int]
    depth: int
    bbox: AABB
    cell_size: float
    point_count: int
    start: int
    end: int
    representative: np.ndarray
    color: Optional[np.ndarray]
    normal: Optional[np.ndarray]
    intensity: Optional[float]
    mean_spacing: float
    rms_error: float
    error_bound: float
    child_ids: list[int] = field(default_factory=list)
    is_leaf: bool = True

    @property
    def density(self) -> float:
        """点密度：点数 / 包围盒体积，体积退化时按点数计。"""
        volume = float(np.prod(np.maximum(self.bbox.size, _EPS)))
        return self.point_count / max(volume, _EPS)


class Octree:
    """点云八叉树。节点保存在 ``nodes`` 列表中，id 即列表下标。"""

    def __init__(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None,
        max_points_per_leaf: int = 64,
        max_depth: int = MAX_DEPTH,
        bounds: Optional[AABB] = None,
    ) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points 必须是形状 (N, 3) 的数组")
        if points.shape[0] == 0:
            raise ValueError("输入点云为空，无法构建八叉树")
        if max_depth < 1 or max_depth > MAX_DEPTH:
            raise ValueError(f"max_depth 必须在 [1, {MAX_DEPTH}] 之间")
        if max_points_per_leaf < 1:
            raise ValueError("max_points_per_leaf 必须 >= 1")

        self.points = points
        self.colors = self._check_attr(colors, "colors", (points.shape[0], 3))
        self.normals = self._check_attr(normals, "normals", (points.shape[0], 3))
        self.intensities = self._check_attr(
            intensities, "intensities", (points.shape[0],)
        )
        self.max_points_per_leaf = int(max_points_per_leaf)
        self.max_depth = int(max_depth)

        if bounds is None:
            bounds = AABB(points.min(axis=0), points.max(axis=0)).expand(1e-6)
        self.bounds = bounds

        quantized = quantize_points(points, bounds)
        self._morton = morton_encode(quantized)
        self.sorted_indices = np.argsort(self._morton, kind="stable")
        self._sorted_morton = self._morton[self.sorted_indices]

        self.nodes: list[NodeStats] = []
        self.root_id = self._build(0, 0, 0, points.shape[0], bounds)

    @staticmethod
    def _check_attr(attr, name, shape):
        if attr is None:
            return None
        attr = np.asarray(attr, dtype=np.float64)
        if attr.shape != shape:
            raise ValueError(f"{name} 形状应为 {shape}，实际为 {attr.shape}")
        return attr

    def _new_node_id(self) -> int:
        node_id = len(self.nodes)
        self.nodes.append(None)  # type: ignore[arg-type]
        return node_id

    def _build(
        self, depth: int, parent_id: Optional[int], start: int, end: int, bbox: AABB
    ) -> int:
        node_id = self._new_node_id()
        count = end - start

        must_split = count > self.max_points_per_leaf and depth < self.max_depth
        child_ids: list[int] = []
        if must_split:
            codes = self._sorted_morton[start:end]
            keys = octant_at_depth(codes, depth)
            # Morton 有序：相同八分体编号构成连续区间，搜索 8 个分界即可。
            boundaries = np.searchsorted(keys, np.arange(1, 8))
            cuts = np.concatenate(([0], boundaries, [count]))
            for octant in range(8):
                cs, ce = int(cuts[octant]), int(cuts[octant + 1])
                if ce > cs:  # 跳过空八分体，不产生空节点
                    child = self._build(
                        depth + 1, node_id, start + cs, start + ce, bbox.child(octant)
                    )
                    child_ids.append(child)

        stats = self._aggregate(node_id, parent_id, depth, start, end, bbox, child_ids)
        self.nodes[node_id] = stats
        return node_id

    def _aggregate(
        self,
        node_id: int,
        parent_id: Optional[int],
        depth: int,
        start: int,
        end: int,
        bbox: AABB,
        child_ids: list[int],
    ) -> NodeStats:
        idx = self.sorted_indices[start:end]
        pts = self.points[idx]
        count = int(idx.shape[0])

        tight_min = pts.min(axis=0)
        tight_max = pts.max(axis=0)
        tight_box = AABB(tight_min, tight_max)
        representative = pts.mean(axis=0)

        color = self._mean_attr(self.colors, idx, axis=0)
        normal = self._mean_attr(self.normals, idx, axis=0)
        if normal is not None:
            norm = np.linalg.norm(normal)
            if norm > _EPS:
                normal = normal / norm
        intensity = self._mean_attr(self.intensities, idx, axis=None)

        # 局部点间距：用节点量化网格的占用率线性估计，
        # spacing = cell_diag / sqrt(occupied/total)，密集 -> 小、稀疏 -> 大；
        # 重复点不增加占用网格，因此间距随重复程度自然变大（回退口径）。
        mean_spacing = _grid_spacing_estimate(
            self._sorted_morton, start, end, depth, bbox, float(bbox.diagonal)
        )

        rms_error = _rms_distance_to_centroid(pts, representative)
        # 严格误差上界：盒内任意点到代表点的距离不超过 tight 包围盒对角线。
        error_bound = max(float(tight_box.diagonal), _EPS)

        return NodeStats(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            bbox=tight_box,
            cell_size=float(bbox.diagonal),
            point_count=count,
            start=int(start),
            end=int(end),
            representative=representative,
            color=color,
            normal=normal,
            intensity=None if intensity is None else float(intensity),
            mean_spacing=mean_spacing,
            rms_error=rms_error,
            error_bound=error_bound,
            child_ids=child_ids,
            is_leaf=not child_ids,
        )

    @staticmethod
    def _mean_attr(attr, idx, axis):
        if attr is None:
            return None
        values = attr[idx]
        if axis is None:
            return float(values.mean())
        return values.mean(axis=axis)

    def node_points(self, node: NodeStats) -> np.ndarray:
        """返回该节点覆盖的原始点（按 Morton 顺序）。"""
        return self.points[self.sorted_indices[node.start:node.end]]

    def __len__(self) -> int:
        return len(self.nodes)

    def iter_nodes(self):
        return iter(self.nodes)


def _rms_distance_to_centroid(points: np.ndarray, centroid: np.ndarray) -> float:
    """点到质心距离的均方根（RMS），几何离散度的统计度量。"""
    delta = points - centroid
    return float(np.sqrt(np.einsum("ij,ij->i", delta, delta).mean()))


def _grid_spacing_estimate(
    sorted_morton: np.ndarray,
    start: int,
    end: int,
    depth: int,
    bbox: AABB,
    cell_diagonal: float,
) -> float:
    """基于节点细分网格占用率的局部点间距估计（O(k) 时间、O(k) 内存）。

    把节点占据的空间按再下一层八叉树网格（每轴 2 份，共 8 个 cell）
    统计被占用的 cell 数。若点在 cell 间均匀分布，典型点间距与
    ``cell 对角线 / sqrt(占用率)`` 同阶。全部点落在同一 cell
    （含完全重复点）时占用率饱和，回退为当前八叉树 cell 对角线，
    保证结果有限且非零。
    """
    codes = sorted_morton[start:end]
    node_diag = max(float(bbox.diagonal), _EPS)
    cell_diagonal = max(float(cell_diagonal), _EPS)
    if depth < MAX_DEPTH:
        octants = octant_at_depth(codes, depth)
        occupied = int(np.unique(octants).size)
        if occupied == 1:
            # 所有点（含完全重复点）落在同一细分 cell：
            # 无法从占用率区分间距，回退为八叉树 cell 对角线。
            return cell_diagonal
        occupancy = occupied / 8.0
        cell_diag = node_diag * 0.5
    else:
        occupancy = 1.0
        cell_diag = cell_diagonal
    return max(cell_diag / np.sqrt(max(occupancy, _EPS)), _EPS)
