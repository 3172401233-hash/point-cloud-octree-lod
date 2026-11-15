"""轴对齐包围盒与 3D Morton 编码等几何基础工具。

所有向量化运算基于 ``numpy``；Morton 码使用 64 位无符号整数，
每轴保留 21 bit，因此八叉树最大深度为 20（每消耗 3 bit 下降一层）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_DEPTH = 20
QUANT_BITS = 21


@dataclass
class AABB:
    """轴对齐包围盒（axis-aligned bounding box）。"""

    min: np.ndarray
    max: np.ndarray

    def __post_init__(self) -> None:
        self.min = np.asarray(self.min, dtype=np.float64)
        self.max = np.asarray(self.max, dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        return (self.min + self.max) * 0.5

    @property
    def size(self) -> np.ndarray:
        return self.max - self.min

    @property
    def half_extent(self) -> np.ndarray:
        return (self.max - self.min) * 0.5

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.max - self.min))

    def expand(self, margin: float = 1e-9) -> "AABB":
        """返回外扩后的包围盒，保证 min < max（退化点云也可递归细分）。"""
        return AABB(self.min - margin, self.max + margin)

    def child(self, octant: int) -> "AABB":
        """按八分体编号 (0..7) 返回子包围盒，位 0/1/2 对应 x/y/z。"""
        center = self.center
        lo = np.empty(3, dtype=np.float64)
        hi = np.empty(3, dtype=np.float64)
        for axis in range(3):
            high_side = (octant >> axis) & 1
            lo[axis] = center[axis] if high_side else self.min[axis]
            hi[axis] = self.max[axis] if high_side else center[axis]
        return AABB(lo, hi)

    def distance_to_point(self, point: np.ndarray) -> float:
        """点到包围盒的最短距离（点在盒内时为 0）。"""
        point = np.asarray(point, dtype=np.float64)
        delta = np.maximum(self.min - point, 0.0)
        delta = np.maximum(delta, point - self.max)
        return float(np.linalg.norm(delta))

    def closest_point(self, point: np.ndarray) -> np.ndarray:
        """包围盒上离给定点最近的点。"""
        point = np.asarray(point, dtype=np.float64)
        return np.minimum(np.maximum(point, self.min), self.max)

    def corners(self) -> np.ndarray:
        """返回 8 个角点，形状 (8, 3)。"""
        m, M = self.min, self.max
        return np.array(
            [
                [m[0], m[1], m[2]],
                [M[0], m[1], m[2]],
                [m[0], M[1], m[2]],
                [M[0], M[1], m[2]],
                [m[0], m[1], M[2]],
                [M[0], m[1], M[2]],
                [m[0], M[1], M[2]],
                [M[0], M[1], M[2]],
            ],
            dtype=np.float64,
        )


def _split_by_bits(values: np.ndarray) -> np.ndarray:
    """把每个整数的低 21 bit 以 0b...0x0x0x 的形式分散开。"""
    x = values.astype(np.uint64)
    x &= np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton_encode(xyz: np.ndarray) -> np.ndarray:
    """对量化后的整数坐标 (..., 3) 编码 3D Morton 码，返回 uint64 数组。"""
    xyz = np.asarray(xyz, dtype=np.uint64)
    return (
        _split_by_bits(xyz[..., 0])
        | (_split_by_bits(xyz[..., 1]) << np.uint64(1))
        | (_split_by_bits(xyz[..., 2]) << np.uint64(2))
    )


def octant_at_depth(morton: np.ndarray, depth: int) -> np.ndarray:
    """取出 Morton 码在第 ``depth`` 层（根为 0）使用的八分体编号 0..7。"""
    shift = np.uint64(3 * (MAX_DEPTH - depth))
    return (morton >> shift) & np.uint64(7)


def quantize_points(points: np.ndarray, bounds: AABB) -> np.ndarray:
    """把世界坐标量化到 [0, 2**21-1] 的整数网格，形状 (N, 3)。"""
    points = np.asarray(points, dtype=np.float64)
    extent = np.maximum(bounds.max - bounds.min, 1e-12)
    scale = (2**QUANT_BITS - 1) / extent
    q = np.floor((points - bounds.min) * scale).astype(np.int64)
    return np.clip(q, 0, 2**QUANT_BITS - 1).astype(np.uint64)
