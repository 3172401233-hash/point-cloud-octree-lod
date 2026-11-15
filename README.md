# point-cloud-octree-lod

Python 点云八叉树构建与 LOD（level of detail）选择库。面向数千万点规模的
渲染场景：远处不再按原始密度提交全部顶点，而是按相机屏幕误差选择聚合节点，
在节点预算内控制顶点处理开销，同时避免朴素等间隔抽样造成的颜色/轮廓抖动。

## 依赖

- Python 3.10+
- numpy（唯一第三方依赖）

## 模块结构

```
pointcloud_lod/
  geometry.py  轴对齐包围盒 AABB、3D Morton 编码/解码、坐标量化
  octree.py    八叉树构建、节点统计与属性聚合（Octree / NodeStats）
  camera.py    针孔相机、视锥剔除、屏幕误差 SSE、深度瓦片遮挡
  lod.py       LOD 选择、密度感知误差、预算强制合并、区域统计
  metrics.py   Hausdorff / RMS / 屏幕空间轮廓误差、预算占用报告
tests/         38 个 unittest 用例
demo.py        密集/稀疏对照的端到端演示
```

## 快速开始

```python
import numpy as np
from pointcloud_lod import Camera, Octree, select_lod, lod_point_set, symmetric_hausdorff

points = np.random.default_rng(0).uniform(0, 10, (1_000_000, 3))
colors = np.random.default_rng(1).uniform(0, 255, (1_000_000, 3))

tree = Octree(points, colors=colors, max_points_per_leaf=64)

camera = Camera(
    position=np.array([20.0, 5.0, 5.0]),
    target=np.array([5.0, 5.0, 5.0]),
    up=np.array([0.0, 0.0, 1.0]),
    fov_y=np.radians(60.0),
    aspect=1.7,
    viewport_height=1080,
)

# 屏幕误差阈值 2 像素，前沿节点数硬上限 2000
result = select_lod(tree, camera, sse_threshold=2.0, node_budget=2000)
lod_points = lod_point_set(tree, result)
contour_error = symmetric_hausdorff(points, lod_points)
```

运行演示与测试：

```bash
python demo.py
python -m unittest discover -s tests -v
```

## 八叉树构建

1. 根包围盒取点云 AABB 并外扩一个极小量，退化点云（所有点重合）也能递归。
2. 点坐标量化到每轴 21 bit 整数网格，编码 3D Morton 码（uint64）。
3. 按 Morton 码全局排序：同一节点的点在排序数组中连续，节点只保存
   区间 `[start, end)`，不复制点数据。
4. 递归八分体细分，空八分体直接跳过，**不会产生空节点**；达到叶子点数
   上限或最大深度（20 层）停止。
5. 每个节点聚合保存：
   - tight 包围盒、八叉树 cell 尺寸、点数；
   - 代表点（质心）、颜色/法线/强度均值（法线归一化，缺失属性为 `None`）；
   - 局部点间距 `mean_spacing`（密度度量）；
   - RMS 误差与误差上界 `error_bound`。

### 重复点与非均匀密度

- 重复点量化到同一网格，落在同一叶子，不产生空子节点。
- 局部点间距用节点细分网格（8 个次层 cell）的占用率线性估计：
  `间距 ≈ cell 对角线 / sqrt(占用率)`，O(k) 时间、O(k) 内存，
  不会出现暴力最近邻的 O(k²) 内存爆炸。所有点重合时回退为 cell 对角线，
  保证结果有限且为正。

## LOD 选择

### 密度感知有效误差

每个节点的有效几何误差（世界单位）：

```
e_eff = sqrt(error_bound² + (beta · mean_spacing)²)
```

- `error_bound` 是几何误差上界（节点 tight 包围盒对角线）；
- `mean_spacing` 是局部点间距，`beta` 由 `density_weight` 控制（默认 0.5）。

稀疏区域点间距大，单个代表点无法覆盖轮廓空洞，因此会以更高优先级继续
细分；密集区域则更激进地聚合。这避免了“每隔几个点抽一个”在非均匀密度下
的轮廓/颜色抖动。

### 屏幕空间误差（SSE）

透视投影标准公式，距离取相机到节点包围盒的**最近距离**：

```
SSE = e_eff · (viewport_height / 2) / tan(fov_y / 2) / max(d, near)
```

SSE 高于阈值则继续细分，否则节点进入 LOD 前沿。相机进入节点内部时
最近距离为 0，SSE 视为无穷大，包含相机的节点被强制细分到叶子。

### 视锥与遮挡

- 视锥剔除：节点包围球对 4 个侧平面做保守测试，完全在外则跳过。
- 遮挡剔除：前向后遍历，深度瓦片只记录**已选祖先节点**包围球的最近深度；
  测试节点只有在其最远深度仍落在所覆盖瓦片的已存深度之后才剔除。
  因为瓦片只由已选祖先写入，剔除的子树本就会被祖先代表点替代，
  不会把点云空洞误判为实心几何。可用 `enable_occlusion=False` 关闭。

### 节点预算

前沿节点数超过 `node_budget` 时，反复合并“可见子节点全部在前沿中”的
父节点（被视锥/遮挡剔除的子树不参与），每次选 SSE 最小的一组以最小化
视觉损失，直到满足硬上限。受八叉树兄弟粒度限制，最终节点数可能低于
预算，但绝不会超过；合并过程中被选节点覆盖的原始点集合保持完整。

## 误差如何度量

| 度量 | 定义 | 口径 |
| --- | --- | --- |
| 节点误差上界 `error_bound` | 节点 tight 包围盒对角线 | 确定性上界：盒内任意点到代表点距离不超过它 |
| 节点 RMS 误差 `rms_error` | 节点内各点到质心距离的均方根 | 统计离散度 |
| 有向 Hausdorff 距离 | `max_{p∈A} min_{q∈B} |p−q|` | 简化集 B 相对原始集 A 的最坏轮廓偏差 |
| 对称 Hausdorff 距离 | 两个方向有向 Hausdorff 的最大值 | 双向最坏轮廓偏差（世界单位） |
| RMS 最近邻距离 | 原始点到最近 LOD 代表点距离的均方根 | 平均轮廓偏差（世界单位） |
| 屏幕空间 Hausdorff 误差 | 最近邻世界距离按各点视深度投影为像素的最大值 | 与 SSE 阈值同口径（像素） |

屏幕空间换算与 LOD 选择使用同一投影公式：

```
pixel = d · (viewport_height / 2) / tan(fov_y / 2) / z
```

因此 `screen_space_hausdorff` 的输出（像素）可以直接与 `sse_threshold`
比较，判断当前 LOD 是否满足屏幕误差预算。

## 边界情况覆盖

- 重复点：同一叶子，间距/误差有限，无空节点。
- 空输入：构造 `Octree` 抛出 `ValueError`。
- 属性缺失：对应字段为 `None`，代表颜色接口返回 `None`。
- 非均匀密度：点间距估计区分密/疏区域，驱动差异化细化。
- 相机在点云内部：最近距离为 0，强制细分；预算优先级更高时仍可强制合并。
- 退化包围盒：根 AABB 外扩，单点点云正常构建。
