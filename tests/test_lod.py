"""LOD 选择（阈值、预算、遮挡、密度差异、相机在内部）的单元测试。"""

import unittest

import numpy as np

from pointcloud_lod import (
    Camera,
    Octree,
    budget_usage,
    select_lod,
)
from pointcloud_lod.camera import screen_space_error
from pointcloud_lod.lod import region_breakdown


def make_camera(position, target, fov=60.0, aspect=1.7):
    return Camera(
        position=np.asarray(position, dtype=float),
        target=np.asarray(target, dtype=float),
        up=np.array([0.0, 0.0, 1.0]),
        fov_y=np.radians(fov),
        aspect=aspect,
    )


class LODThresholdTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.points = rng.uniform(0.0, 10.0, (6000, 3))
        self.tree = Octree(self.points, max_points_per_leaf=64)
        self.camera = make_camera([20.0, 5.0, 5.0], [5.0, 5.0, 5.0])

    def test_tighter_threshold_selects_more_nodes(self):
        coarse = select_lod(self.tree, self.camera, 1024.0, 100000, enable_occlusion=False)
        medium = select_lod(self.tree, self.camera, 512.0, 100000, enable_occlusion=False)
        fine = select_lod(self.tree, self.camera, 256.0, 100000, enable_occlusion=False)
        self.assertGreater(len(medium.node_ids), len(coarse.node_ids))
        self.assertGreater(len(fine.node_ids), len(medium.node_ids))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            select_lod(self.tree, self.camera, 0.0, 10)
        with self.assertRaises(ValueError):
            select_lod(self.tree, self.camera, 1.0, 0)

    def test_sse_scales_with_distance(self):
        near = screen_space_error(self.camera, 1.0, 5.0)
        far = screen_space_error(self.camera, 1.0, 10.0)
        self.assertAlmostEqual(near / far, 2.0, places=8)

    def test_front_disjoint_and_covers_visible_points(self):
        result = select_lod(self.tree, self.camera, 1.0, 100000, enable_occlusion=False)
        ranges = sorted(
            (self.tree.nodes[i].start, self.tree.nodes[i].end)
            for i in result.node_ids
        )
        for (a, b), (c, d) in zip(ranges, ranges[1:]):
            self.assertLessEqual(b, c)
        # 每个被选叶子点恰好被计数一次。
        self.assertEqual(
            result.total_points,
            sum(self.tree.nodes[i].point_count for i in result.node_ids),
        )


class BudgetTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        points = rng.uniform(0.0, 10.0, (4000, 3))
        self.tree = Octree(points, max_points_per_leaf=24)
        self.camera = make_camera([20.0, 5.0, 5.0], [5.0, 5.0, 5.0])

    def test_budget_is_hard_limit(self):
        for budget in (50, 20, 10, 1):
            result = select_lod(
                self.tree, self.camera, 0.5, budget, enable_occlusion=False
            )
            self.assertLessEqual(len(result.node_ids), budget)
            self.assertTrue(result.budget_forced or len(result.node_ids) <= budget)

    def test_budget_one_returns_root(self):
        result = select_lod(self.tree, self.camera, 0.5, 1, enable_occlusion=False)
        self.assertEqual(result.node_ids, [self.tree.root_id])
        self.assertEqual(result.total_points, self.tree.nodes[self.tree.root_id].point_count)

    def test_larger_budget_never_increases_sse_worst_case_monotonic(self):
        small = select_lod(self.tree, self.camera, 0.5, 5, enable_occlusion=False)
        large = select_lod(self.tree, self.camera, 0.5, 500, enable_occlusion=False)
        self.assertLessEqual(len(small.node_ids), len(large.node_ids))

    def test_budget_usage_report(self):
        result = select_lod(self.tree, self.camera, 1.0, 8, enable_occlusion=False)
        report = budget_usage(result)
        self.assertLessEqual(report["selected_nodes"], 8)
        self.assertLessEqual(report["budget_ratio"], 1.0 + 1e-12)
        self.assertGreaterEqual(report["forced_merges"], 0)


class FrustumAndOcclusionTest(unittest.TestCase):
    def test_frustum_culling_removes_back_nodes(self):
        points = np.random.default_rng(0).uniform(-10.0, 10.0, (3000, 3))
        tree = Octree(points, max_points_per_leaf=32)
        camera = make_camera([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], fov=50.0)
        result = select_lod(tree, camera, 1.0, 100000, enable_occlusion=False)
        self.assertGreater(result.frustum_culled, 0)
        for node in result.nodes:
            self.assertGreaterEqual(node.representative[0], -1.0)

    def test_occlusion_keeps_near_wall_drops_far_wall(self):
        axis = np.linspace(-2.0, 2.0, 32)
        xx, zz = np.meshgrid(axis, axis)
        wall = lambda x: np.column_stack(
            [np.full(xx.size, x), xx.ravel(), zz.ravel()]
        )
        points = np.vstack([wall(0.0), wall(5.0)])
        tree = Octree(points, max_points_per_leaf=32, max_depth=12)
        camera = make_camera([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], fov=90.0, aspect=1.0)
        visible = select_lod(tree, camera, 1.0, 100000, enable_occlusion=False)
        occluded = select_lod(tree, camera, 1.0, 100000, enable_occlusion=True)
        far_visible = [n for n in visible.nodes if n.representative[0] > 2.5]
        far_occluded = [n for n in occluded.nodes if n.representative[0] > 2.5]
        near_occluded = [n for n in occluded.nodes if n.representative[0] < 2.5]
        self.assertGreater(len(far_visible), 0)
        self.assertEqual(len(far_occluded), 0)
        self.assertGreater(len(near_occluded), 0)
        self.assertGreater(occluded.occluded_nodes, 0)


class CameraInsideCloudTest(unittest.TestCase):
    def test_inside_camera_refines_containing_nodes(self):
        points = np.random.default_rng(3).uniform(-10.0, 10.0, (5000, 3))
        tree = Octree(points, max_points_per_leaf=32)
        camera = make_camera([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], fov=90.0)
        result = select_lod(tree, camera, 2.0, 100000)
        for node in result.nodes:
            if node.bbox.distance_to_point(camera.position) < 1e-9 and not node.is_leaf:
                self.fail("包含相机的非叶子节点不应停在 LOD 前沿")

    def test_budget_overrides_inside_refinement(self):
        points = np.random.default_rng(4).uniform(-10.0, 10.0, (3000, 3))
        tree = Octree(points, max_points_per_leaf=32)
        camera = make_camera([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], fov=90.0)
        result = select_lod(
            tree, camera, 1.0, 3, enable_occlusion=False
        )
        self.assertLessEqual(len(result.node_ids), 3)


class DensityAwareLODTest(unittest.TestCase):
    """等尺寸、等距的两个盒子，仅点密度不同，用于隔离密度项的作用。"""

    def _two_cluster_cloud(self):
        rng = np.random.default_rng(0)
        dense = rng.uniform(low=[0.0, -1.0, -1.0], high=[2.0, 1.0, 1.0], size=(2000, 3))
        sparse = rng.uniform(low=[6.0, -1.0, -1.0], high=[8.0, 1.0, 1.0], size=(64, 3))
        return np.vstack([dense, sparse])

    def _camera(self):
        return make_camera([4.0, -12.0, 0.0], [4.0, 0.0, 0.0], fov=60.0)

    def test_sparse_region_has_larger_mean_spacing(self):
        tree = Octree(self._two_cluster_cloud(), max_points_per_leaf=4)
        result = select_lod(
            tree, self._camera(), 100.0, 100000, enable_occlusion=False
        )
        dense = region_breakdown(tree, result, lambda p: p[0] < 3.0)
        sparse = region_breakdown(tree, result, lambda p: p[0] > 5.0)
        self.assertGreater(sparse["mean_spacing"], dense["mean_spacing"])
        self.assertGreater(dense["points"], sparse["points"])

    def test_density_weight_refines_sparse_region_more(self):
        tree = Octree(self._two_cluster_cloud(), max_points_per_leaf=4)
        camera = self._camera()
        unweighted = select_lod(
            tree, camera, 150.0, 100000, density_weight=0.0, enable_occlusion=False
        )
        weighted = select_lod(
            tree, camera, 150.0, 100000, density_weight=2.0, enable_occlusion=False
        )
        sparse_u = region_breakdown(tree, unweighted, lambda p: p[0] > 5.0)
        sparse_w = region_breakdown(tree, weighted, lambda p: p[0] > 5.0)
        dense_w = region_breakdown(tree, weighted, lambda p: p[0] < 3.0)
        self.assertGreater(sparse_w["nodes"], sparse_u["nodes"])
        # 密集区域不应被密度项推动而增加节点。
        dense_u = region_breakdown(tree, unweighted, lambda p: p[0] < 3.0)
        self.assertLessEqual(dense_w["nodes"], dense_u["nodes"] + 1)

    def test_sparse_uses_more_nodes_per_point(self):
        tree = Octree(self._two_cluster_cloud(), max_points_per_leaf=4)
        result = select_lod(
            tree, self._camera(), 100.0, 100000, enable_occlusion=False
        )
        dense = region_breakdown(tree, result, lambda p: p[0] < 3.0)
        sparse = region_breakdown(tree, result, lambda p: p[0] > 5.0)
        sparse_per_point = sparse["nodes"] / sparse["points"]
        dense_per_point = dense["nodes"] / dense["points"]
        self.assertGreater(sparse_per_point, dense_per_point)


if __name__ == "__main__":
    unittest.main()
