"""LOD 选择：屏幕误差驱动细分，结合密度、视锥、遮挡与节点预算。

选择分两个阶段：

1. **自顶向下细分（前向后顺序）**
   - 视锥外的节点直接剔除；
   - 被深度瓦片完全遮挡的节点跳过（保守判定，宁可多保留）；
   - 密度感知有效误差
     ``e_eff = sqrt(error_bound^2 + (beta * mean_spacing)^2)``，
     其中 ``error_bound`` 是几何误差上界（包围盒对角线），
     ``mean_spacing`` 是局部点间距。稀疏区域点间距大，单个代表点
     无法覆盖轮廓空洞，因此会以更高优先级继续细分；
   - SSE 低于阈值或到达叶子时，节点进入当前 LOD 前沿，并写入深度瓦片。

2. **预算强制收敛**
   若前沿节点数超过 ``node_budget``，反复合并“所有子节点都在前沿中”
   的父节点，每次合并选 SSE 最小（视觉损失最小）的一组，直到满足预算。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .camera import (
    Camera,
    DepthTileBuffer,
    node_distance,
    screen_space_error,
    sphere_in_frustum,
)
from .octree import NodeStats, Octree


@dataclass
class SelectionResult:
    """一次 LOD 选择的结果。"""

    node_ids: list[int]
    tree: Octree
    camera: Camera
    sse_threshold: float
    node_budget: int
    sse_by_node: dict[int, float]
    frustum_culled: int
    occluded_nodes: int
    forced_merges: int
    budget_forced: bool
    culled_node_ids: set[int] = field(default_factory=set)

    @property
    def nodes(self) -> list[NodeStats]:
        return [self.tree.nodes[nid] for nid in self.node_ids]

    @property
    def total_points(self) -> int:
        return sum(self.tree.nodes[nid].point_count for nid in self.node_ids)

    @property
    def max_sse(self) -> float:
        return max((self.sse_by_node[nid] for nid in self.node_ids), default=0.0)

    @property
    def mean_sse(self) -> float:
        values = [self.sse_by_node[nid] for nid in self.node_ids]
        return float(np.mean(values)) if values else 0.0

    def representative_points(self) -> np.ndarray:
        """前沿各节点的代表点，形状 (K, 3)。"""
        return np.array([self.tree.nodes[nid].representative for nid in self.node_ids])

    def representative_colors(self) -> Optional[np.ndarray]:
        colors = [self.tree.nodes[nid].color for nid in self.node_ids]
        if any(c is None for c in colors):
            return None
        return np.array(colors)


def effective_geometric_error(node: NodeStats, density_weight: float) -> float:
    """密度感知的有效几何误差（世界单位，仍是上界口径）。"""
    spacing_term = density_weight * node.mean_spacing
    return float(np.hypot(node.error_bound, spacing_term))


def select_lod(
    tree: Octree,
    camera: Camera,
    sse_threshold: float,
    node_budget: int,
    *,
    density_weight: float = 0.5,
    enable_occlusion: bool = True,
    tiles_x: int = 32,
    tiles_y: int = 32,
) -> SelectionResult:
    """按相机与阈值选择 LOD 前沿。

    参数
    ----
    sse_threshold:
        允许的最大屏幕空间误差（像素）。SSE 高于该值的节点继续细分。
    node_budget:
        前沿节点数硬上限（>= 1）；超限时按最小 SSE 损失合并兄弟子树。
    density_weight:
        局部点间距在有效误差中的权重 beta，越大越偏向细化稀疏区域。
    """
    if sse_threshold <= 0:
        raise ValueError("sse_threshold 必须为正数")
    if node_budget < 1:
        raise ValueError("node_budget 必须 >= 1")

    depth_buffer = DepthTileBuffer(tiles_x, tiles_y)
    heap: list[tuple[float, int]] = [(0.0, tree.root_id)]
    front: list[int] = []
    front_set: set[int] = set()
    sse_cache: dict[int, float] = {}
    frustum_culled = 0
    occluded_nodes = 0
    culled_node_ids: set[int] = set()

    while heap:
        _, node_id = heapq.heappop(heap)
        node = tree.nodes[node_id]
        center = node.bbox.center
        radius = float(np.linalg.norm(node.bbox.half_extent))

        if not sphere_in_frustum(camera, center, radius):
            frustum_culled += 1
            culled_node_ids.add(node_id)
            continue

        distance = node_distance(camera, node.bbox)
        if distance <= camera.near * 1e-3:
            # 相机在节点内部：距离退化，SSE 视为无穷，强制继续细分。
            sse = float("inf")
        else:
            geom_error = effective_geometric_error(node, density_weight)
            sse = screen_space_error(camera, geom_error, distance)
        sse_cache[node_id] = sse

        # 保守遮挡：深度瓦片只由“已选祖先节点”写入。祖先一旦入选，
        # 其整个子树本来就会被它的代表点替代，因此剔除该子树不会丢失
        # 任何当前 LOD 应显示的内容，也不会把点云空洞误判为实心。
        if enable_occlusion and depth_buffer.is_occluded(camera, center, radius):
            occluded_nodes += 1
            culled_node_ids.add(node_id)
            continue

        if node.is_leaf or sse <= sse_threshold:
            if node_id not in front_set:
                front.append(node_id)
                front_set.add(node_id)
            if enable_occlusion:
                depth_buffer.rasterize_node(camera, center, radius)
        else:
            for child_id in node.child_ids:
                child = tree.nodes[child_id]
                heapq.heappush(
                    heap, (node_distance(camera, child.bbox), child_id)
                )

    # 所有后代都被剔除的内部节点本身也不贡献任何可见点，
    # 标记为剔除，使预算合并可以越过这些“空贡献”子树。
    # 自底向上不动点传播，直到剔除集合不再增长。
    changed = True
    while changed:
        changed = False
        for node_id in sse_cache.keys() - front_set - culled_node_ids:
            node = tree.nodes[node_id]
            if node.child_ids and all(
                c in culled_node_ids for c in node.child_ids
            ):
                culled_node_ids.add(node_id)
                changed = True

    forced = 0
    if len(front) > node_budget:
        forced = _enforce_budget(
            tree, front, front_set, sse_cache, node_budget, culled_node_ids
        )

    return SelectionResult(
        node_ids=front,
        tree=tree,
        camera=camera,
        sse_threshold=sse_threshold,
        node_budget=node_budget,
        sse_by_node=sse_cache,
        frustum_culled=frustum_culled,
        occluded_nodes=occluded_nodes,
        forced_merges=forced,
        budget_forced=forced > 0,
        culled_node_ids=culled_node_ids,
    )


def _enforce_budget(
    tree: Octree,
    front: list[int],
    front_set: set[int],
    sse_cache: dict[int, float],
    budget: int,
    culled_node_ids: set[int],
) -> int:
    """不断合并兄弟子树，直到节点数不超过预算。

    视锥外（或被遮挡）的子树不在前沿中，也不携带任何已选点，因此
    合并条件只看“已访问的可见子节点”：父节点的每个子节点要么本身
    在前沿，要么从未进入遍历（被剔除）。正在细分中的子节点不允许提前合并。
    """

    def mergeable(parent_id: int) -> bool:
        parent = tree.nodes[parent_id]
        if not parent.child_ids:
            return False
        for child_id in parent.child_ids:
            if child_id in front_set:
                continue
            if child_id in culled_node_ids:
                continue  # 被视锥/遮挡剔除的子树，不占预算
            return False
        return any(c in front_set for c in parent.child_ids)

    merge_heap: list[tuple[float, int]] = []

    def push_if_mergeable(parent_id: Optional[int]) -> None:
        if parent_id is not None and mergeable(parent_id):
            heapq.heappush(merge_heap, (sse_cache[parent_id], parent_id))

    for node_id in list(front_set):
        push_if_mergeable(tree.nodes[node_id].parent_id)

    merges = 0
    while len(front_set) > budget:
        while merge_heap and not mergeable(merge_heap[0][1]):
            heapq.heappop(merge_heap)
        if not merge_heap:
            break

        _, parent_id = heapq.heappop(merge_heap)
        if not mergeable(parent_id):
            continue
        parent = tree.nodes[parent_id]
        for child_id in parent.child_ids:
            front_set.discard(child_id)
        front_set.add(parent_id)
        merges += 1
        push_if_mergeable(parent.parent_id)

    # 以集合为准重建前沿，按深度再按节点 id 排序，保证结果确定。
    front[:] = sorted(front_set, key=lambda nid: (tree.nodes[nid].depth, nid))
    return merges


def region_breakdown(
    tree: Octree,
    result: SelectionResult,
    region: Callable[[np.ndarray], bool],
) -> dict:
    """按空间谓词统计某区域在当前 LOD 前沿中的占用情况。

    ``region`` 接收节点代表点，返回该节点是否属于目标区域。
    """
    chosen = [
        tree.nodes[nid]
        for nid in result.node_ids
        if region(tree.nodes[nid].representative)
    ]
    if not chosen:
        return {"nodes": 0, "points": 0, "mean_density": 0.0, "mean_spacing": 0.0}
    return {
        "nodes": len(chosen),
        "points": sum(n.point_count for n in chosen),
        "mean_density": float(np.mean([n.density for n in chosen])),
        "mean_spacing": float(np.mean([n.mean_spacing for n in chosen])),
    }
