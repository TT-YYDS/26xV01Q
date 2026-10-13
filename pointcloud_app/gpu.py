from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Literal, Sequence

import numpy as np


GPU_IMPLEMENTATION_VERSION = 2


ComputeBackend = Literal["auto", "gpu", "cpu"]


class GpuUnavailableError(RuntimeError):
    """GPU 后端不可用时抛出。"""


@dataclass(frozen=True)
class GpuCapabilities:
    available: bool
    device_count: int
    device_name: str | None
    compute_capability: str | None
    cuda_runtime_version: str | None
    cupy_version: str | None
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=1)
def get_gpu_capabilities() -> GpuCapabilities:
    """检测 CuPy/CUDA 是否能真正执行 GPU 计算。"""
    try:
        import cupy as cp
    except Exception as error:
        return GpuCapabilities(False, 0, None, None, None, None, f"未安装或无法导入 CuPy：{error}")

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            return GpuCapabilities(False, 0, None, None, None, cp.__version__, "未检测到可用 CUDA GPU。")
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"].decode("utf-8", errors="replace")
        capability = f"{int(properties['major'])}.{int(properties['minor'])}"
        runtime_version = int(cp.cuda.runtime.runtimeGetVersion())
        cuda_runtime_version = f"{runtime_version // 1000}.{(runtime_version % 1000) // 10}"
        cp.asarray([1.0], dtype=cp.float32).sum().item()
        return GpuCapabilities(True, device_count, name, capability, cuda_runtime_version, cp.__version__, None)
    except Exception as error:
        return GpuCapabilities(False, 0, None, None, None, cp.__version__, f"CUDA 初始化失败：{error}")


def resolve_compute_backend(requested_backend: str) -> Literal["gpu", "cpu"]:
    """解析用户选择的后端；自动模式会在 CUDA 不可用时安全回退到 CPU。"""
    if requested_backend not in {"auto", "gpu", "cpu"}:
        raise ValueError("计算后端必须是 auto、gpu 或 cpu。")
    if requested_backend == "cpu":
        return "cpu"

    capabilities = get_gpu_capabilities()
    if capabilities.available:
        return "gpu"
    if requested_backend == "gpu":
        raise GpuUnavailableError(f"已选择强制 GPU，但 CUDA GPU 不可用：{capabilities.reason}")
    return "cpu"


def gpu_runtime_summary(active_backend: Literal["gpu", "cpu"], requested_backend: str) -> dict[str, object]:
    capabilities = get_gpu_capabilities()
    return {
        "requested_backend": requested_backend,
        "active_backend": active_backend,
        "gpu_capabilities": capabilities.as_dict(),
        "gpu_accelerated_stages": [
            "体素采样",
            "统计与半径离群点滤波",
            "ROI 裁剪",
            "ICP 精配准",
            "位姿应用与融合",
            "竖直切片",
        ] if active_backend == "gpu" else [],
        "cpu_stages": [
            "PLY 文件读取与写入",
            "FPFH + RANSAC 全局粗配准",
            "PNG 截面图与顶视图导出",
        ],
    }


def gpu_voxel_down_sample(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """在 CUDA 上按体素聚合点和颜色。"""
    if voxel_size <= 0:
        raise ValueError("体素大小必须大于 0。")
    if len(points) == 0:
        return _empty_points(), _empty_colors_or_none(colors)

    cp = _require_cupy()
    with cp.cuda.Device(0):
        device_points = _to_device_matrix(points, cp)
        device_colors = _to_device_colors(colors, len(device_points), cp)
        sampled_points, sampled_colors = _device_voxel_down_sample(device_points, device_colors, voxel_size, cp)
        return _to_host_matrix(sampled_points, cp), _to_host_matrix(sampled_colors, cp) if sampled_colors is not None else None


def gpu_statistical_inlier_mask(points: np.ndarray, neighbor_count: int, std_ratio: float) -> np.ndarray:
    """在 CUDA KDTree 上计算统计离群点的保留掩码。"""
    if neighbor_count < 1:
        raise ValueError("统计邻居数必须大于 0。")
    if std_ratio <= 0:
        raise ValueError("统计标准差倍率必须大于 0。")
    if len(points) <= neighbor_count:
        return np.ones(len(points), dtype=bool)

    cp = _require_cupy()
    with cp.cuda.Device(0):
        device_points = _to_device_matrix(points, cp)
        distances = _query_knn_distances(device_points, device_points, min(neighbor_count + 1, len(points)), cp)
        neighbor_distances = distances[:, 1:]
        mean_distances = cp.mean(neighbor_distances, axis=1)
        threshold = cp.mean(mean_distances) + float(std_ratio) * cp.std(mean_distances)
        return cp.asnumpy(mean_distances <= threshold)


def gpu_radius_inlier_mask(points: np.ndarray, radius: float, minimum_points: int) -> np.ndarray:
    """以第 N 近邻距离实现 CUDA 半径离群点过滤。"""
    if radius <= 0:
        raise ValueError("半径必须大于 0。")
    if minimum_points < 1:
        raise ValueError("半径内最少点数必须大于 0。")
    if len(points) <= minimum_points:
        return np.ones(len(points), dtype=bool)

    cp = _require_cupy()
    with cp.cuda.Device(0):
        device_points = _to_device_matrix(points, cp)
        distances = _query_knn_distances(device_points, device_points, minimum_points, cp)
        kth_distance = distances if distances.ndim == 1 else distances[:, -1]
        return cp.asnumpy(kth_distance <= float(radius))


def gpu_bounds_mask(points: np.ndarray, minimum: Sequence[float], maximum: Sequence[float]) -> np.ndarray:
    """在 CUDA 上计算轴对齐 ROI 掩码。"""
    cp = _require_cupy()
    with cp.cuda.Device(0):
        device_points = _to_device_matrix(points, cp)
        lower_bound = cp.asarray(minimum, dtype=cp.float32)
        upper_bound = cp.asarray(maximum, dtype=cp.float32)
        mask = cp.all((device_points >= lower_bound) & (device_points <= upper_bound), axis=1)
        return cp.asnumpy(mask)


def gpu_vertical_section_mask(
    points: np.ndarray,
    normal: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    plane_offset: float,
    thickness: float,
    horizontal_min: float,
    horizontal_max: float,
    vertical_min: float,
    vertical_max: float,
) -> np.ndarray:
    """在 CUDA 上计算任意倾斜截面的保留掩码。"""
    cp = _require_cupy()
    with cp.cuda.Device(0):
        device_points = _to_device_matrix(points, cp)
        device_normal = cp.asarray(normal, dtype=cp.float32)
        device_horizontal = cp.asarray(horizontal, dtype=cp.float32)
        device_vertical = cp.asarray(vertical, dtype=cp.float32)
        signed_distance = device_points @ device_normal - float(plane_offset)
        horizontal_coordinate = device_points @ device_horizontal
        vertical_coordinate = device_points @ device_vertical
        mask = (
            (cp.abs(signed_distance) <= float(thickness) / 2.0)
            & (horizontal_coordinate >= float(horizontal_min))
            & (horizontal_coordinate <= float(horizontal_max))
            & (vertical_coordinate >= float(vertical_min))
            & (vertical_coordinate <= float(vertical_max))
        )
        return cp.asnumpy(mask)


def gpu_transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """在 CUDA 上应用 4×4 刚体变换。"""
    if len(points) == 0:
        return _empty_points()

    cp = _require_cupy()
    with cp.cuda.Device(0):
        device_points = _to_device_matrix(points, cp)
        device_transform = cp.asarray(transformation, dtype=cp.float32)
        transformed = _apply_transform(device_points, device_transform)
        return _to_host_matrix(transformed, cp)
def gpu_fuse_clouds(
    point_sets: Sequence[np.ndarray],
    color_sets: Sequence[np.ndarray | None],
    transformations: Sequence[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """在 CUDA 上应用位姿、拼接并按体素融合多个点云。"""
    if not point_sets:
        return _empty_points(), None
    if len(point_sets) != len(transformations) or len(color_sets) != len(point_sets):
        raise ValueError("融合时点云、颜色和位姿数量必须一致。")

    cp = _require_cupy()
    with cp.cuda.Device(0):
        transformed_sets = []
        device_color_sets = []
        has_complete_colors = all(colors is not None and len(colors) == len(points) for points, colors in zip(point_sets, color_sets))
        for points, colors, transformation in zip(point_sets, color_sets, transformations):
            device_points = _to_device_matrix(points, cp)
            device_transform = cp.asarray(transformation, dtype=cp.float32)
            transformed_sets.append(_apply_transform(device_points, device_transform))
            if has_complete_colors:
                device_color_sets.append(_to_device_colors(colors, len(device_points), cp))

        merged_points = cp.concatenate(transformed_sets, axis=0)
        merged_colors = cp.concatenate(device_color_sets, axis=0) if has_complete_colors else None
        sampled_points, sampled_colors = _device_voxel_down_sample(merged_points, merged_colors, voxel_size, cp)
        return _to_host_matrix(sampled_points, cp), _to_host_matrix(sampled_colors, cp) if sampled_colors is not None else None


def gpu_refine_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray | None,
    max_correspondence_distance: float,
    max_iterations: int,
    use_point_to_plane: bool,
    initial_transformation: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """在 CUDA KDTree 上执行点到点或点到平面 ICP。

    ``initial_transformation`` is intentionally optional for backward
    compatibility.  The calibration-block geometry stage can provide a small
    plane-constrained seed before the iterative solver starts.
    """
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("GPU ICP 至少需要 3 个源点和目标点。")
    if max_correspondence_distance <= 0 or max_iterations < 1:
        raise ValueError("GPU ICP 参数无效。")

    cp = _require_cupy()
    with cp.cuda.Device(0):
        source = _to_device_matrix(source_points, cp)
        target = _to_device_matrix(target_points, cp)
        normals = _to_device_colors(target_normals, len(target), cp) if use_point_to_plane else None
        if use_point_to_plane and normals is None:
            raise ValueError("点到平面 GPU ICP 需要目标点法线。")

        from cupyx.scipy.spatial import KDTree

        target_tree = KDTree(target)
        if initial_transformation is None:
            transformation = cp.eye(4, dtype=cp.float32)
        else:
            initial = np.asarray(initial_transformation, dtype=np.float32)
            if initial.shape != (4, 4) or not np.isfinite(initial).all():
                raise ValueError("GPU ICP 初始变换必须是有限的 4×4 矩阵。")
            transformation = cp.asarray(initial, dtype=cp.float32)
        previous_rmse: float | None = None
        for _ in range(max_iterations):
            transformed = _apply_transform(source, transformation)
            distances, indices = target_tree.query(transformed, k=1)
            correspondence_mask = distances <= float(max_correspondence_distance)
            correspondence_count = int(cp.count_nonzero(correspondence_mask).item())
            if correspondence_count < (6 if use_point_to_plane else 3):
                break

            matched_source = transformed[correspondence_mask]
            matched_target = target[indices[correspondence_mask]]
            if use_point_to_plane:
                matched_normals = normals[indices[correspondence_mask]]
                delta = _point_to_plane_delta(matched_source, matched_target, matched_normals, cp)
            else:
                delta = _point_to_point_delta(matched_source, matched_target, cp)
            transformation = delta @ transformation

            rmse = float(cp.sqrt(cp.mean(distances[correspondence_mask] ** 2)).item())
            delta_motion = float(cp.linalg.norm(delta - cp.eye(4, dtype=cp.float32)).item())
            if delta_motion < 1e-6 or (previous_rmse is not None and abs(previous_rmse - rmse) < 1e-7):
                break
            previous_rmse = rmse

        transformed = _apply_transform(source, transformation)
        distances, _ = target_tree.query(transformed, k=1)
        correspondence_mask = distances <= float(max_correspondence_distance)
        correspondence_count = int(cp.count_nonzero(correspondence_mask).item())
        fitness = correspondence_count / len(source_points)
        rmse = float(cp.sqrt(cp.mean(distances[correspondence_mask] ** 2)).item()) if correspondence_count else float("inf")
        return cp.asnumpy(transformation).astype(np.float64, copy=False), float(fitness), rmse


def _require_cupy():
    capabilities = get_gpu_capabilities()
    if not capabilities.available:
        raise GpuUnavailableError(capabilities.reason or "CUDA GPU 不可用。")
    import cupy as cp

    return cp


def _to_device_matrix(values: np.ndarray, cp):
    matrix = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("点数据必须是形状为 (N, 3) 的数组。")
    return cp.asarray(matrix)


def _to_device_colors(colors: np.ndarray | None, point_count: int, cp):
    if colors is None:
        return None
    matrix = np.ascontiguousarray(np.asarray(colors, dtype=np.float32))
    if matrix.ndim != 2 or matrix.shape != (point_count, 3):
        raise ValueError("颜色数据必须与点数据一一对应，且形状为 (N, 3)。")
    return cp.asarray(matrix)


def _to_host_matrix(values, cp) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(values), dtype=np.float64)


def _device_voxel_down_sample(device_points, device_colors, voxel_size: float, cp):
    grid = cp.floor(device_points / float(voxel_size)).astype(cp.int64)
    shifted_grid = grid - cp.min(grid, axis=0)
    spans = tuple(int(value) for value in cp.asnumpy(cp.max(shifted_grid, axis=0) + 1))
    if int(spans[0]) * int(spans[1]) * int(spans[2]) > np.iinfo(np.int64).max:
        raise ValueError("点云坐标范围过大，无法安全编码 GPU 体素索引。")
    voxel_keys = (shifted_grid[:, 0] * spans[1] + shifted_grid[:, 1]) * spans[2] + shifted_grid[:, 2]
    _, inverse = cp.unique(voxel_keys, return_inverse=True)
    voxel_count = int(cp.max(inverse).item()) + 1
    counts = cp.bincount(inverse, minlength=voxel_count).astype(cp.float32)
    sampled_points = cp.stack(
        [cp.bincount(inverse, weights=device_points[:, axis], minlength=voxel_count) for axis in range(3)],
        axis=1,
    ) / counts[:, None]
    sampled_colors = None
    if device_colors is not None:
        sampled_colors = cp.stack(
            [cp.bincount(inverse, weights=device_colors[:, axis], minlength=voxel_count) for axis in range(3)],
            axis=1,
        ) / counts[:, None]
    return sampled_points, sampled_colors


def _query_knn_distances(reference_points, query_points, neighbor_count: int, cp):
    from cupyx.scipy.spatial import KDTree

    tree = KDTree(reference_points)
    distances, _ = tree.query(query_points, k=neighbor_count)
    return distances


def _apply_transform(points, transformation):
    return points @ transformation[:3, :3].T + transformation[:3, 3]


def _point_to_plane_delta(source_points, target_points, target_normals, cp):
    matrix = cp.concatenate((cp.cross(source_points, target_normals), target_normals), axis=1)
    residual = cp.sum(target_normals * (source_points - target_points), axis=1)
    solution, _, _, _ = cp.linalg.lstsq(matrix, -residual, rcond=None)
    rotation = _rodrigues_rotation(solution[:3], cp)
    delta = cp.eye(4, dtype=cp.float32)
    delta[:3, :3] = rotation
    delta[:3, 3] = solution[3:]
    return delta


def _point_to_point_delta(source_points, target_points, cp):
    source_center = cp.mean(source_points, axis=0)
    target_center = cp.mean(target_points, axis=0)
    covariance = (source_points - source_center).T @ (target_points - target_center)
    left, _, right_transposed = cp.linalg.svd(covariance)
    rotation = right_transposed.T @ left.T
    if float(cp.linalg.det(rotation).item()) < 0:
        right_transposed[-1, :] *= -1
        rotation = right_transposed.T @ left.T
    translation = target_center - rotation @ source_center
    delta = cp.eye(4, dtype=cp.float32)
    delta[:3, :3] = rotation
    delta[:3, 3] = translation
    return delta


def _rodrigues_rotation(rotation_vector, cp):
    theta = float(cp.linalg.norm(rotation_vector).item())
    skew = cp.zeros((3, 3), dtype=cp.float32)
    skew[0, 1] = -rotation_vector[2]
    skew[0, 2] = rotation_vector[1]
    skew[1, 0] = rotation_vector[2]
    skew[1, 2] = -rotation_vector[0]
    skew[2, 0] = -rotation_vector[1]
    skew[2, 1] = rotation_vector[0]
    if theta < 1e-8:
        return cp.eye(3, dtype=cp.float32) + skew
    return cp.eye(3, dtype=cp.float32) + (cp.sin(theta) / theta) * skew + ((1.0 - cp.cos(theta)) / (theta * theta)) * (skew @ skew)


def _empty_points() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def _empty_colors_or_none(colors: np.ndarray | None) -> np.ndarray | None:
    return np.empty((0, 3), dtype=np.float64) if colors is not None else None
