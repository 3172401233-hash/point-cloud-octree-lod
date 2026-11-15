"""点云八叉树构建与 LOD 选择库。

公开接口：
- :class:`~pointcloud_lod.geometry.AABB`：轴对齐包围盒与 Morton 编码工具；
- :class:`~pointcloud_lod.octree.Octree` /
  :class:`~pointcloud_lod.octree.NodeStats`：八叉树与节点聚合统计；
- :class:`~pointcloud_lod.camera.Camera` /
  :class:`~pointcloud_lod.camera.DepthTileBuffer`：相机、视锥与遮挡；
- :func:`~pointcloud_lod.lod.select_lod` /
  :class:`~pointcloud_lod.lod.SelectionResult`：LOD 选择；
- :mod:`~pointcloud_lod.metrics`：Hausdorff / 屏幕空间误差与预算统计。
"""

from .camera import Camera, DepthTileBuffer, screen_space_error
from .geometry import AABB
from .lod import SelectionResult, effective_geometric_error, select_lod
from .metrics import (
    budget_usage,
    directed_hausdorff,
    lod_point_set,
    rms_nearest_distance,
    screen_space_hausdorff,
    symmetric_hausdorff,
)
from .octree import NodeStats, Octree

__all__ = [
    "AABB",
    "Camera",
    "DepthTileBuffer",
    "NodeStats",
    "Octree",
    "SelectionResult",
    "budget_usage",
    "directed_hausdorff",
    "effective_geometric_error",
    "lod_point_set",
    "rms_nearest_distance",
    "screen_space_error",
    "screen_space_hausdorff",
    "select_lod",
    "symmetric_hausdorff",
]
