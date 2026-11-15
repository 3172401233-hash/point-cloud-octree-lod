"""八叉树构建与节点聚合的单元测试。"""

import unittest

import numpy as np

from pointcloud_lod.geometry import AABB, morton_encode, octant_at_depth, quantize_points
from pointcloud_lod.octree import Octree


def grid_points(n: int = 8, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    axis = np.linspace(lo, hi, n)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


class MortonTest(unittest.TestCase):
    def test_known_morton_codes(self):
        cases = [
            ((1, 0, 0), 0b001),
            ((0, 1, 0), 0b010),
            ((0, 0, 1), 0b100),
            ((2, 0, 0), 0b001000),
            ((3, 3, 3), 0b111111),
        ]
        for xyz, expected in cases:
            code = morton_encode(np.array([xyz], dtype=np.uint64))[0]
            self.assertEqual(int(code), expected)

    def test_octant_prefix_ordering(self):
        # 同一父节点下的 8 个八分体，其 Morton 码按八分体编号有序。
        # depth=19 右移 3 bit，取的是每轴 bit1（Morton bit3=x 的 bit1）：
        # x 的 bit1 为 0/1/0/1/0/1/0/1，映射到八分体即 0,0,1,1 周期重复。
        coords = np.array([[i, 0, 0] for i in range(8)], dtype=np.uint64)
        codes = morton_encode(coords)
        self.assertTrue(np.all(np.diff(codes) > 0))
        octants = octant_at_depth(codes, 19)
        np.testing.assert_array_equal(octants, np.array([0, 0, 1, 1, 0, 0, 1, 1]))
        # depth=20（超出最大深度的边界）等价于不右移，x=1 落在八分体 1。
        edge = octant_at_depth(morton_encode(np.array([[1, 0, 0]], dtype=np.uint64)), 20)
        self.assertEqual(int(edge[0]), 1)

    def test_quantize_clamps_to_grid(self):
        bounds = AABB(np.zeros(3), np.ones(3))
        q = quantize_points(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), bounds)
        self.assertEqual(q.min(), 0)
        self.assertEqual(q.max(), 2**21 - 1)


class OctreeBuildTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.points = rng.uniform(0.0, 10.0, (3000, 3))
        self.colors = rng.uniform(0.0, 255.0, (3000, 3))
        self.normals = rng.normal(size=(3000, 3))
        self.intensities = rng.random(3000)

    def test_leaf_partition_covers_all_points_disjointly(self):
        tree = Octree(self.points, self.colors, self.normals, self.intensities,
                      max_points_per_leaf=32)
        leaves = [n for n in tree.nodes if n.is_leaf]
        self.assertGreater(len(leaves), 1)
        ranges = sorted((n.start, n.end) for n in leaves)
        for (a, b), (c, d) in zip(ranges, ranges[1:]):
            self.assertLessEqual(b, c)
        self.assertEqual(sum(n.point_count for n in leaves), self.points.shape[0])

    def test_points_stay_inside_node_bbox(self):
        tree = Octree(self.points, max_points_per_leaf=16)
        for node in tree.nodes:
            pts = tree.node_points(node)
            self.assertTrue(np.all(pts <= node.bbox.max + 1e-9))
            self.assertTrue(np.all(pts >= node.bbox.min - 1e-9))

    def test_leaf_capacity_and_depth_limit(self):
        tree = Octree(self.points, max_points_per_leaf=32, max_depth=6)
        for node in tree.nodes:
            if node.is_leaf and node.depth < 6:
                self.assertLessEqual(node.point_count, 32)
            self.assertLessEqual(node.depth, 6)

    def test_no_empty_nodes(self):
        tree = Octree(self.points, max_points_per_leaf=32)
        for node in tree.nodes:
            self.assertGreater(node.point_count, 0)

    def test_duplicate_points_share_a_leaf(self):
        points = np.zeros((100, 3))
        points[:] = np.array([1.0, 2.0, 3.0])
        tree = Octree(points, max_points_per_leaf=8)
        leaves = [n for n in tree.nodes if n.is_leaf]
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].point_count, 100)
        # 重复点无法从空间区分间距：估计值必须有限且为正，不能出现 NaN/0。
        self.assertTrue(np.isfinite(leaves[0].mean_spacing))
        self.assertGreater(leaves[0].mean_spacing, 0.0)
        # 误差上界同样必须有限。
        self.assertTrue(np.isfinite(leaves[0].error_bound))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            Octree(np.zeros((0, 3)))

    def test_bad_attribute_shape_raises(self):
        with self.assertRaises(ValueError):
            Octree(self.points, colors=np.zeros((5, 3)))

    def test_missing_attributes_are_none(self):
        tree = Octree(self.points, max_points_per_leaf=32)
        for node in tree.nodes:
            self.assertIsNone(node.color)
            self.assertIsNone(node.normal)
            self.assertIsNone(node.intensity)

    def test_representative_attributes_aggregated(self):
        tree = Octree(self.points, self.colors, self.normals, self.intensities,
                      max_points_per_leaf=32)
        root = tree.nodes[tree.root_id]
        np.testing.assert_allclose(root.representative, self.points.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(root.color, self.colors.mean(axis=0), atol=1e-10)
        self.assertAlmostEqual(root.intensity, float(self.intensities.mean()), places=10)
        # 聚合法线被归一化。
        self.assertAlmostEqual(np.linalg.norm(root.normal), 1.0, places=10)

    def test_error_bound_dominates_rms_and_covers_every_point(self):
        tree = Octree(self.points, max_points_per_leaf=32)
        for node in tree.nodes:
            self.assertGreaterEqual(node.error_bound, node.rms_error - 1e-9)
            pts = tree.node_points(node)
            dist = np.linalg.norm(pts - node.representative, axis=1)
            self.assertTrue(np.all(dist <= node.error_bound + 1e-9))

    def test_nonuniform_density_spacing_differs(self):
        # 密集簇 + 稀疏簇：平均点间距应明显不同。
        dense = np.random.default_rng(0).normal(0.0, 0.05, (2000, 3))
        sparse = np.random.default_rng(1).uniform(4.0, 10.0, (200, 3))
        tree = Octree(np.vstack([dense, sparse]), max_points_per_leaf=64)
        dense_nodes = [n for n in tree.nodes if n.representative[0] < 1.0 and n.is_leaf]
        sparse_nodes = [n for n in tree.nodes if n.representative[0] > 3.0 and n.is_leaf]
        dense_spacing = np.mean([n.mean_spacing for n in dense_nodes])
        sparse_spacing = np.mean([n.mean_spacing for n in sparse_nodes])
        self.assertLess(dense_spacing, sparse_spacing)

    def test_zero_normals_do_not_produce_nan(self):
        points = np.random.default_rng(0).uniform(0.0, 1.0, (500, 3))
        normals = np.zeros((500, 3))
        normals[:250] = np.array([1.0, 0.0, 0.0])
        tree = Octree(points, normals=normals, max_points_per_leaf=16)
        for node in tree.nodes:
            if node.normal is not None:
                self.assertTrue(np.isfinite(node.normal).all())

    def test_all_duplicate_points_with_attributes(self):
        points = np.tile(np.array([[1.0, 2.0, 3.0]]), (30, 1))
        colors = np.ones((30, 3)) * 128.0
        normals = np.tile(np.array([0.0, 0.0, 1.0]), (30, 1))
        intensities = np.full(30, 0.5)
        tree = Octree(
            points, colors, normals, intensities,
            max_points_per_leaf=4, max_depth=5,
        )
        leaves = [n for n in tree.nodes if n.is_leaf]
        self.assertEqual(len(leaves), 1)
        leaf = leaves[0]
        np.testing.assert_allclose(leaf.color, 128.0)
        self.assertAlmostEqual(leaf.intensity, 0.5)
        self.assertLessEqual(max(n.depth for n in tree.nodes), 5)

    def test_single_point_cloud(self):
        tree = Octree(np.array([[0.5, 0.5, 0.5]]), max_points_per_leaf=1)
        self.assertEqual(len(tree.nodes), 1)
        self.assertTrue(tree.nodes[0].is_leaf)
        self.assertTrue(np.isfinite(tree.nodes[0].error_bound))


if __name__ == "__main__":
    unittest.main()
