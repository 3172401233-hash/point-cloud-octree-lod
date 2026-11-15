"""针孔相机模型、视锥剔除、屏幕误差与深度瓦片遮挡。

屏幕空间误差 (screen-space error, SSE)
--------------------------------------
对几何误差上界为 ``e`` 的节点，在距离相机 ``d`` 处，其投影到屏幕上的
像素尺寸为（透视投影标准公式）::

    SSE = e * (viewport_height / 2) / tan(fov_y / 2) / max(d, near)

距离取相机到节点包围盒的最近距离（而非中心距离），这样相机进入点云
内部时距离为 0，节点必然被强制细分。

遮挡判定采用保守口径：深度瓦片只记录已提交节点的**最近**深度，
测试节点只有在其包围球**最远**深度仍大于覆盖瓦片的已存深度时才算被
遮挡，从而不会错误剔除可见内容。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import AABB


@dataclass
class Camera:
    """简单针孔相机：eye/look-at/up + 垂直视场（弧度）。"""

    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y: float
    aspect: float
    near: float = 0.05
    far: float = 1.0e6
    viewport_height: int = 1080

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.target = np.asarray(self.target, dtype=np.float64)
        self.up = np.asarray(self.up, dtype=np.float64)
        if self.fov_y <= 0 or self.fov_y >= np.pi:
            raise ValueError("fov_y 必须在 (0, pi) 之间")
        if self.aspect <= 0 or self.near <= 0:
            raise ValueError("aspect 与 near 必须为正数")
        forward = self.target - self.position
        if np.linalg.norm(forward) < 1e-12:
            raise ValueError("相机 position 与 target 不能重合")
        self._basis()
        self.planes = self._frustum_planes()

    def _basis(self) -> None:
        self.forward = self.target - self.position
        self.forward /= np.linalg.norm(self.forward)
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        tan_half = np.tan(self.fov_y * 0.5)
        self.tan_half_y = float(tan_half)
        self.tan_half_x = float(tan_half * self.aspect)

    def _frustum_planes(self):
        """返回 (normal, point) 列表，法线指向视锥外侧，共 4 个侧平面。

        近/远平面不参与剔除：点云常跨越巨大深度范围，且遮挡模块单独处理
        深度，因此这里只做侧向视锥剔除。
        """
        f, r, u = self.forward, self.right, self.up
        ty, tx = self.tan_half_y, self.tan_half_x
        # 方向向量指向视锥外侧：视锥内点与外侧法线的点积 <= 0。
        plane_dirs = [
            u - f * ty,   # 下
            -u - f * ty,  # 上
            r - f * tx,   # 左
            -r - f * tx,  # 右
        ]
        return [(d / np.linalg.norm(d), self.position) for d in plane_dirs]

    def view_direction(self) -> np.ndarray:
        return self.forward.copy()


def sphere_in_frustum(camera: Camera, center: np.ndarray, radius: float) -> bool:
    """包围球是否与视锥相交（保守：边界情况视为可见）。"""
    center = np.asarray(center, dtype=np.float64)
    for normal, point in camera.planes:
        if float(normal @ (center - point)) > radius:
            return False
    return True


def node_distance(camera: Camera, bbox: AABB) -> float:
    """相机到节点包围盒的最近距离。"""
    return bbox.distance_to_point(camera.position)


def screen_space_error(
    camera: Camera, geometric_error: float, distance: float
) -> float:
    """几何误差上界投影到视口高度方向的像素数。"""
    d = max(float(distance), camera.near)
    return float(
        geometric_error
        * (camera.viewport_height * 0.5)
        / np.tan(camera.fov_y * 0.5)
        / d
    )


def project_points(camera: Camera, points: np.ndarray):
    """把世界坐标投影为归一化设备坐标 (NDC, [-1,1]) 与视空间深度。

    返回 ``(ndc_x, ndc_y, view_depth)``，视空间深度 <= 0 的点位于相机后方。
    """
    points = np.asarray(points, dtype=np.float64)
    rel = points - camera.position
    z = rel @ camera.forward
    x = rel @ camera.right
    y = rel @ camera.up
    safe_z = np.where(np.abs(z) < 1e-12, 1e-12, z)
    ndc_x = x / (safe_z * camera.tan_half_x)
    ndc_y = y / (safe_z * camera.tan_half_y)
    return ndc_x, ndc_y, z


class DepthTileBuffer:
    """低分辨率深度瓦片缓冲，用于保守的前向后遮挡剔除。

    屏幕被划分为 ``tiles_x * tiles_y`` 个瓦片。每个瓦片记录已提交节点
    包围球的**最近**深度（越小越靠近相机）。测试节点时，只有它覆盖的
    全部瓦片都已被更近的内容占满，且节点包围球的最远深度仍在这些深度
    之后，才判定为被遮挡——保证不会漏掉实际可见的几何。
    """

    def __init__(self, tiles_x: int = 32, tiles_y: int = 32) -> None:
        if tiles_x < 1 or tiles_y < 1:
            raise ValueError("瓦片数量必须为正整数")
        self.tiles_x = int(tiles_x)
        self.tiles_y = int(tiles_y)
        self.depth = np.full((self.tiles_y, self.tiles_x), np.inf, dtype=np.float64)

    def reset(self) -> None:
        self.depth.fill(np.inf)

    def _tile_span(self, ndc_center, ndc_radius):
        x0 = int(np.floor((ndc_center[0] - ndc_radius + 1.0) * 0.5 * self.tiles_x))
        x1 = int(np.ceil((ndc_center[0] + ndc_radius + 1.0) * 0.5 * self.tiles_x))
        y0 = int(np.floor((1.0 - (ndc_center[1] + ndc_radius)) * 0.5 * self.tiles_y))
        y1 = int(np.ceil((1.0 - (ndc_center[1] - ndc_radius)) * 0.5 * self.tiles_y))
        return (
            max(0, x0), min(self.tiles_x, x1),
            max(0, y0), min(self.tiles_y, y1),
        )

    def _projection(self, camera: Camera, center, radius):
        ndc_x, ndc_y, z = project_points(camera, center[None, :])
        if z[0] <= camera.near:
            return None
        center_depth = float(z[0])
        nearest = max(center_depth - radius, camera.near)
        farthest = center_depth + radius
        ndc_radius = radius / (nearest * camera.tan_half_y)
        x0, x1, y0, y1 = self._tile_span(
            (float(ndc_x[0]), float(ndc_y[0])), ndc_radius
        )
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1, nearest, farthest

    def rasterize_node(
        self, camera: Camera, center: np.ndarray, radius: float
    ) -> None:
        """把一个已选节点作为遮挡体写入其最近深度（相机后方则忽略）。"""
        proj = self._projection(camera, np.asarray(center, dtype=np.float64), radius)
        if proj is None:
            return
        x0, x1, y0, y1, nearest, _ = proj
        current = self.depth[y0:y1, x0:x1]
        np.minimum(current, nearest, out=current)

    def is_occluded(
        self, camera: Camera, center: np.ndarray, radius: float
    ) -> bool:
        """节点包围球是否完全落在已写入遮挡体之后（保守判定）。"""
        proj = self._projection(camera, np.asarray(center, dtype=np.float64), radius)
        if proj is None:
            return False  # 相机后方/内部或投影退化：保守保留
        x0, x1, y0, y1, _, farthest = proj
        tile_depths = self.depth[y0:y1, x0:x1]
        return bool(np.all(tile_depths < farthest))
