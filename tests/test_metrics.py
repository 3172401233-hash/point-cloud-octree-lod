"""误差度量（Hausdorff、屏幕空间误差、预算统计、端到端）的单元测试。"""

import unittest

import numpy as np

from pointcloud_lod import (
    Camera,
    Octree,
    budget_usage,
    directed_hausdorff,
    lod_point_set,
    rms_nearest_distance,
    screen_space_error,
    screen_space_hausdorff,
    symmetric_hausdorff,
)


def make_camera(position, target, fov=60.0, aspect=1.7):
    return Camera(
        position=np.asarray(position, dtype=float),
        target=np.asarray(target, dtype=float),
        up=np.array([0.0, 0.0, 1.0]),
        fov_y=np.radians(fov),
        aspect=aspect,
    )


class HausdorffTest(unittest.TestCase):
    def test_identical_sets_zero_error(self):
        points = np.random.default_rng(0).uniform(0, 1, (100, 3))
        self.assertEqual(directed_hausdorff(points, points), 0.0)
        self.assertEqual(symmetric_hausdorff(points, points), 0.0)
        self.assertEqual(rms_nearest_distance(points, points), 0.0)

    def test_known_offset_hausdorff(self):
        reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        simplified = np.array([[0.0, 0.0, 0.0]])
        # reference 中离 simplified 最远的点距离为 1。
        self.assertAlmostEqual(directed_hausdorff(reference, simplified), 1.0)
        self.assertAlmostEqual(symmetric_hausdorff(reference, simplified), 1.0)
        self.assertAlmostEqual(
            rms_nearest_distance(reference, simplified), np.sqrt(0.5)
        )

    def test_empty_sets_raise(self):
        with self.assertRaises(ValueError):
            directed_hausdorff(np.zeros((0, 3)), np.zeros((1, 3)))

    def test_more_lod_nodes_reduce_hausdorff_error(self):
        rng = np.random.default_rng(5)
        points = rng.uniform(0.0, 10.0, (6000, 3))
        tree = Octree(points, max_points_per_leaf=64)
        camera = make_camera([20.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        coarse = select_lod_local(tree, camera, 1024.0)
        fine = select_lod_local(tree, camera, 256.0)
        self.assertGreater(len(fine.node_ids), len(coarse.node_ids))
        # 代表点越多，原始点到最近代表点的平均（RMS）轮廓误差越小。
        rms_coarse = rms_nearest_distance(points, lod_point_set(tree, coarse))
        rms_fine = rms_nearest_distance(points, lod_point_set(tree, fine))
        self.assertLess(rms_fine, rms_coarse)


class ScreenSpaceMetricTest(unittest.TestCase):
    def test_screen_space_hausdorff_matches_sse_formula(self):
        camera = make_camera([10.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        reference = np.array([[0.0, 1.0, 0.0]])
        simplified = np.array([[0.0, 0.0, 0.0]])
        expected = screen_space_error(camera, 1.0, 10.0)
        self.assertAlmostEqual(
            screen_space_hausdorff(camera, reference, simplified), expected, places=6
        )

    def test_screen_error_grows_when_camera_closer(self):
        camera_near = make_camera([5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        camera_far = make_camera([20.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        reference = np.array([[0.0, 1.0, 0.0]])
        simplified = np.array([[0.0, 0.0, 0.0]])
        near_err = screen_space_hausdorff(camera_near, reference, simplified)
        far_err = screen_space_hausdorff(camera_far, reference, simplified)
        self.assertGreater(near_err, far_err)


class EndToEndTest(unittest.TestCase):
    def test_pipeline_with_all_attributes(self):
        rng = np.random.default_rng(9)
        points = rng.uniform(0.0, 20.0, (5000, 3))
        colors = rng.uniform(0.0, 255.0, (5000, 3))
        normals = rng.normal(size=(5000, 3))
        intensities = rng.random(5000)
        tree = Octree(points, colors, normals, intensities, max_points_per_leaf=48)
        camera = make_camera([35.0, 10.0, 10.0], [10.0, 10.0, 10.0])
        result = tree_and_select(tree, camera, budget=64)

        self.assertLessEqual(len(result.node_ids), 64)
        report = budget_usage(result)
        self.assertLessEqual(report["budget_ratio"], 1.0 + 1e-12)
        lod_points = lod_point_set(tree, result)
        self.assertEqual(lod_points.shape, (len(result.node_ids), 3))
        # 代表颜色被保留下来。
        self.assertEqual(result.representative_colors().shape, (len(result.node_ids), 3))
        # 误差有限且为正。
        hausdorff = symmetric_hausdorff(points, lod_points)
        screen = screen_space_hausdorff(camera, points, lod_points)
        self.assertTrue(np.isfinite(hausdorff) and hausdorff > 0.0)
        self.assertTrue(np.isfinite(screen) and screen > 0.0)

    def test_pipeline_with_missing_attributes(self):
        rng = np.random.default_rng(10)
        points = rng.uniform(0.0, 10.0, (1500, 3))
        tree = Octree(points, max_points_per_leaf=32)
        camera = make_camera([18.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        result = tree_and_select(tree, camera, budget=32)
        self.assertIsNone(result.representative_colors())
        self.assertLessEqual(len(result.node_ids), 32)


def select_lod_local(tree, camera, threshold):
    from pointcloud_lod import select_lod
    return select_lod(tree, camera, threshold, 100000, enable_occlusion=False)


def tree_and_select(tree, camera, budget):
    from pointcloud_lod import select_lod
    return select_lod(tree, camera, 1.0, budget, enable_occlusion=False)


if __name__ == "__main__":
    unittest.main()
