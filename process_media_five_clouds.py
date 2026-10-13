import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import matplotlib.pyplot as plt
from matplotlib import font_manager
import open3d as o3d

from media_gpu_acceleration import try_create_cuda_backend
from stitch_pointclouds import compute_feature, denoise_cloud, evaluate, extract_calibration_target, refine_icp, remove_target_region

INPUT = ROOT / "media2"
OUTPUT = INPUT / "processed_output2"
DENOISED = OUTPUT / "denoised_inputs2"
SECTION = OUTPUT / "vertical_section2"
for directory in (OUTPUT, DENOISED, SECTION):
    directory.mkdir(parents=True, exist_ok=True)

CLEANING_VOXEL = 0.025
STATISTICAL_NEIGHBORS = 24
STATISTICAL_STD_RATIO = 2.2
RADIUS_NEIGHBORS = 5
RADIUS = 0.10
REGISTRATION_VOXEL = 0.10
OUTPUT_VOXEL = 0.035
TARGET_CROP_X = -2.5
REGISTRATION_SCORE_THRESHOLD = 6.2
REGISTRATION_RMSE_THRESHOLD = 0.05


def ensure_output_directories():
    for directory in (OUTPUT, DENOISED, SECTION):
        directory.mkdir(parents=True, exist_ok=True)


def registration_score(metrics):
    if not metrics["fitness"]:
        return 0.0
    return metrics["fitness"] / (0.03 + metrics["inlier_rmse"])


def is_registration_accepted(metrics):
    return (
        metrics["inlier_rmse"] <= REGISTRATION_RMSE_THRESHOLD
        and registration_score(metrics) >= REGISTRATION_SCORE_THRESHOLD
    )


def refine_registration_candidate(source, target, initial):
    transformation = refine_icp(source, target, initial, REGISTRATION_VOXEL)
    metrics = evaluate(source, target, transformation, REGISTRATION_VOXEL)
    return {
        "transformation": transformation,
        "metrics": metrics,
        "score": registration_score(metrics),
    }


def register_feature(source, target, source_feature, target_feature, trials=12):
    started = time.perf_counter()
    candidates = []
    fgr_metrics = None
    fgr_error = None
    try:
        options = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=REGISTRATION_VOXEL * 2.2,
            iteration_number=64,
            maximum_tuple_count=1000,
        )
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source,
            target,
            source_feature,
            target_feature,
            options,
        )
        candidate = refine_registration_candidate(source, target, result.transformation)
        candidate["method"] = "fast_global_registration"
        candidates.append(candidate)
        fgr_metrics = candidate["metrics"]
        if is_registration_accepted(candidate["metrics"]):
            return candidate["transformation"], {
                "method": candidate["method"],
                "fitness": candidate["metrics"]["fitness"],
                "inlier_rmse": candidate["metrics"]["inlier_rmse"],
                "score": candidate["score"],
                "accepted": True,
                "fgr_metrics": fgr_metrics,
                "ransac_attempts": 0,
                "ransac_attempt_limit": trials,
                "seconds": time.perf_counter() - started,
            }
    except Exception as exc:
        fgr_error = f"{type(exc).__name__}: {exc}"

    for attempt in range(1, trials + 1):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_feature,
            target_feature,
            True,
            REGISTRATION_VOXEL * 2.2,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.85),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(REGISTRATION_VOXEL * 2.2),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(120000, 0.999),
        )
        candidate = refine_registration_candidate(source, target, result.transformation)
        candidate["method"] = "ransac_icp"
        candidate["attempt"] = attempt
        candidates.append(candidate)
        if is_registration_accepted(candidate["metrics"]):
            return candidate["transformation"], {
                "method": candidate["method"],
                "fitness": candidate["metrics"]["fitness"],
                "inlier_rmse": candidate["metrics"]["inlier_rmse"],
                "score": candidate["score"],
                "accepted": True,
                "fgr_metrics": fgr_metrics,
                "fgr_error": fgr_error,
                "ransac_attempts": attempt,
                "ransac_attempt_limit": trials,
                "seconds": time.perf_counter() - started,
            }

    if not candidates:
        raise RuntimeError("标定块全局配准失败，未生成任何候选姿态")
    best = max(candidates, key=lambda item: item["score"])
    return best["transformation"], {
        "method": best["method"],
        "fitness": best["metrics"]["fitness"],
        "inlier_rmse": best["metrics"]["inlier_rmse"],
        "score": best["score"],
        "accepted": is_registration_accepted(best["metrics"]),
        "fgr_metrics": fgr_metrics,
        "fgr_error": fgr_error,
        "ransac_attempts": trials,
        "ransac_attempt_limit": trials,
        "seconds": time.perf_counter() - started,
    }

def height_colors(values):
    low, high = np.percentile(values, [1.0, 99.0])
    normalized = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    return plt.get_cmap("turbo")(normalized)


def save_preview(cloud, path):
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    rng = np.random.default_rng(20260819)
    indices = rng.choice(len(points), min(350000, len(points)), replace=False)
    sampled = points[indices]
    sampled_colors = colors[indices] if len(colors) == len(points) else np.full((len(indices), 3), 0.3)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=170)
    for axis, (first, second), title in zip(axes, ((0, 1), (0, 2), (1, 2)), ("XY", "XZ", "YZ")):
        axis.scatter(sampled[:, first], sampled[:, second], c=np.clip(sampled_colors, 0, 1), s=0.09, linewidths=0, rasterized=True)
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("XYZ"[first])
        axis.set_ylabel("XYZ"[second])
        axis.grid(alpha=0.15)
    figure.suptitle("Media: five-cloud denoised registration")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def vertical_section(cloud):
    points = np.asarray(cloud.points)
    roi_mask = (
        (points[:, 0] > -4.0) & (points[:, 0] < 22.0)
        & (points[:, 1] > -18.0) & (points[:, 1] < -4.0)
        & (points[:, 2] > -1.0) & (points[:, 2] < 8.5)
    )
    roi_xy = points[roi_mask, :2]
    if len(roi_xy) < 1000:
        raise RuntimeError("无法定位长构件区域")
    center_xy = roi_xy.mean(axis=0)
    covariance = np.cov((roi_xy - center_xy).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_xy = eigenvectors[:, np.argmax(eigenvalues)]
    if axis_xy[0] < 0:
        axis_xy = -axis_xy
    lateral_xy = np.array([-axis_xy[1], axis_xy[0]])
    roi_station = (roi_xy - center_xy) @ axis_xy
    station = float(np.quantile(roi_station, 0.45))
    plane_center_xy = center_xy + station * axis_xy
    half_thickness = 0.15
    all_station = (points[:, :2] - center_xy) @ axis_xy
    section_mask = np.abs(all_station - station) <= half_thickness
    section_points = points[section_mask]
    section_cloud = cloud.select_by_index(np.flatnonzero(section_mask))
    section_cloud_path = SECTION / "vertical_section_points.ply"
    o3d.io.write_point_cloud(str(section_cloud_path), section_cloud, write_ascii=False)
    lateral = (section_points[:, :2] - plane_center_xy) @ lateral_xy
    vertical = section_points[:, 2]

    fonts = {font.name for font in font_manager.fontManager.ttflist}
    if "Microsoft YaHei" in fonts:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    figure = plt.figure(figsize=(16, 9), dpi=170, facecolor="#151a21")
    header = figure.add_axes([0.0, 0.92, 1.0, 0.08])
    header.set_facecolor("#222831")
    header.set_xticks([]); header.set_yticks([])
    for spine in header.spines.values(): spine.set_visible(False)
    header.text(0.025, 0.55, "Media五帧点云拼接与垂直切割", color="white", fontsize=20, va="center", weight="bold")
    header.text(0.76, 0.55, "切割平面垂直于构件主轴", color="#66d9ef", fontsize=13, va="center")

    overview = figure.add_axes([0.035, 0.10, 0.57, 0.76], projection="3d")
    overview.set_facecolor("#0d1117")
    rng = np.random.default_rng(19)
    indices = rng.choice(len(points), min(180000, len(points)), replace=False)
    sampled = points[indices]
    overview.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], c=height_colors(sampled[:, 2]), s=0.18, alpha=0.82, linewidths=0)
    y_min, y_max = np.percentile(points[:, 1], [0.5, 99.5])
    z_min, z_max = np.percentile(points[:, 2], [0.5, 99.5])
    yy, zz = np.meshgrid(np.linspace(y_min, y_max, 2), np.linspace(z_min, z_max, 2))
    xx = plane_center_xy[0] - axis_xy[1] * (yy - plane_center_xy[1]) / axis_xy[0]
    overview.plot_surface(xx, yy, zz, color="#00d9ff", alpha=0.30, shade=False)
    overview.set_title("去噪拼接点云与垂直切割平面", color="white", fontsize=15, pad=14)
    overview.set_xlabel("X", color="white"); overview.set_ylabel("Y", color="white"); overview.set_zlabel("Z", color="white")
    overview.tick_params(colors="#b9c0c9", labelsize=8)
    overview.view_init(elev=24, azim=-62); overview.set_box_aspect((2.4, 1.2, 0.7)); overview.grid(False)

    section_axis = figure.add_axes([0.64, 0.28, 0.32, 0.58], facecolor="#f5f7fa")
    section_axis.scatter(lateral, vertical, c=height_colors(vertical), s=1.1, alpha=0.9, linewidths=0, rasterized=True)
    section_axis.set_title("垂直截面（沿主轴观察）", fontsize=15, pad=12)
    section_axis.set_xlabel("横向距离"); section_axis.set_ylabel("Z 高度")
    section_axis.set_aspect("equal", adjustable="datalim"); section_axis.grid(color="#c9d1d9", alpha=0.45)

    angle = float(np.degrees(np.arctan2(axis_xy[1], axis_xy[0])))
    info = figure.add_axes([0.64, 0.10, 0.32, 0.13])
    info.set_facecolor("#222831"); info.set_xticks([]); info.set_yticks([])
    for spine in info.spines.values(): spine.set_color("#39424e")
    info.text(0.04, 0.78, f"主轴方向：{angle:.2f}°", color="white", fontsize=11)
    info.text(0.04, 0.52, f"切割中心：X={plane_center_xy[0]:.3f}, Y={plane_center_xy[1]:.3f}", color="white", fontsize=11)
    info.text(0.04, 0.27, f"切片厚度：{half_thickness * 2:.2f}    截面点数：{len(section_points):,}", color="white", fontsize=11)
    info.text(0.76, 0.27, "已保存", color="#7ee787", fontsize=12, weight="bold")
    interface_path = SECTION / "vertical_cut_interface.png"
    figure.savefig(interface_path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)

    clean_figure, clean_axis = plt.subplots(figsize=(8, 8), dpi=180)
    clean_axis.scatter(lateral, vertical, c=height_colors(vertical), s=1.2, linewidths=0, rasterized=True)
    clean_axis.set_title("垂直截面"); clean_axis.set_xlabel("横向距离"); clean_axis.set_ylabel("Z 高度")
    clean_axis.set_aspect("equal", adjustable="datalim"); clean_axis.grid(alpha=0.3)
    clean_figure.tight_layout(); clean_figure.savefig(SECTION / "vertical_section_only.png", bbox_inches="tight"); plt.close(clean_figure)
    return {"axis_xy": axis_xy.tolist(), "axis_angle_deg": angle, "plane_center_xy": plane_center_xy.tolist(), "slice_thickness": half_thickness * 2, "section_points": int(len(section_points)), "interface_image": str(interface_path), "section_cloud": str(section_cloud_path)}


def merge_clouds_with_acceleration(clouds, targets, transformations):
    target_union = o3d.geometry.PointCloud()
    for target, transformation in zip(targets, transformations):
        transformed_target = copy.deepcopy(target)
        transformed_target.transform(transformation)
        target_union += transformed_target
    target_box = target_union.get_axis_aligned_bounding_box()
    margin = OUTPUT_VOXEL * 3.0
    target_bounds = (
        target_box.get_min_bound() - margin,
        target_box.get_max_bound() + margin,
    )

    backend, backend_error = try_create_cuda_backend()
    acceleration = {
        "requested": "cuda_auto",
        "gpu_available": backend is not None,
        "gpu_initialization_error": backend_error,
        "backend": "open3d_cpu",
        "used": False,
        "task": "merge_voxel_downsample_statistical_filter",
    }
    if backend is not None:
        acceleration["gpu"] = backend.describe()
        try:
            merged, object_only, gpu_timing = backend.merge_clouds(
                clouds,
                transformations,
                OUTPUT_VOXEL,
                statistical_neighbors=20,
                statistical_std_ratio=3.0,
                target_bounds=target_bounds,
            )
            acceleration.update(
                {
                    "backend": backend.info.name,
                    "used": True,
                    "fusion_seconds": gpu_timing["total_seconds"],
                }
            )
            return merged, object_only, acceleration
        except Exception as exc:
            acceleration["gpu_execution_error"] = f"{type(exc).__name__}: {exc}"
            try:
                backend.clear_cache()
            except Exception:
                pass

    started = time.perf_counter()
    merged = o3d.geometry.PointCloud()
    for cloud, transformation in zip(clouds, transformations):
        transformed_cloud = copy.deepcopy(cloud)
        transformed_cloud.transform(transformation)
        merged += transformed_cloud
    merged = merged.voxel_down_sample(OUTPUT_VOXEL)
    merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
    object_only = remove_target_region(merged, target_box, margin)
    acceleration["fusion_seconds"] = time.perf_counter() - started
    return merged, object_only, acceleration


def main():
    ensure_output_directories()
    total_started = time.perf_counter()
    paths = sorted(INPUT.glob("pointcloud_*.ply"))
    if len(paths) != 5:
        raise RuntimeError(f"需要 5 个点云，实际找到 {len(paths)} 个")
    o3d.utility.random.seed(20260819)

    read_started = time.perf_counter()
    raw_clouds = [o3d.io.read_point_cloud(str(path)) for path in paths]
    read_seconds = time.perf_counter() - read_started

    cleaned_clouds = []
    cleaning = []
    cleaning_started = time.perf_counter()
    for path, raw in zip(paths, raw_clouds):
        cloud_started = time.perf_counter()
        cleaned = denoise_cloud(
            raw,
            CLEANING_VOXEL,
            STATISTICAL_NEIGHBORS,
            STATISTICAL_STD_RATIO,
            RADIUS_NEIGHBORS,
            RADIUS,
        )
        cleaned_clouds.append(cleaned)
        output_path = DENOISED / f"{path.stem}_denoised.ply"
        o3d.io.write_point_cloud(str(output_path), cleaned, write_ascii=False)
        cleaning.append(
            {
                "file": path.name,
                "raw_points": len(raw.points),
                "cleaned_points": len(cleaned.points),
                "removed_points": len(raw.points) - len(cleaned.points),
                "removed_ratio": (len(raw.points) - len(cleaned.points)) / len(raw.points),
                "output": str(output_path),
                "seconds": time.perf_counter() - cloud_started,
            }
        )
    cleaning_seconds = time.perf_counter() - cleaning_started
    print(f"[Media] 去噪完成：{cleaning_seconds:.2f} 秒", flush=True)

    target_started = time.perf_counter()
    targets = [
        extract_calibration_target(cloud, TARGET_CROP_X, REGISTRATION_VOXEL)
        for cloud in cleaned_clouds
    ]
    target_seconds = time.perf_counter() - target_started

    feature_started = time.perf_counter()
    features = [compute_feature(target, REGISTRATION_VOXEL) for target in targets]
    feature_seconds = time.perf_counter() - feature_started
    print(f"[Media] 标定块与特征完成：{target_seconds + feature_seconds:.2f} 秒", flush=True)

    registration_started = time.perf_counter()
    edge_0_to_1, edge_0_to_1_info = register_feature(targets[0], targets[1], features[0], features[1], 14)
    edge_0_to_3, edge_0_to_3_info = register_feature(targets[0], targets[3], features[0], features[3], 24)
    edge_2_to_3, edge_2_to_3_info = register_feature(targets[2], targets[3], features[2], features[3], 14)
    edge_0_to_4, edge_0_to_4_info = register_feature(targets[0], targets[4], features[0], features[4], 14)
    registration_seconds = time.perf_counter() - registration_started
    print(f"[Media] 自动配准完成：{registration_seconds:.2f} 秒", flush=True)

    transformations = [np.eye(4) for _ in paths]
    transformations[1] = np.linalg.inv(edge_0_to_1)
    transformations[3] = np.linalg.inv(edge_0_to_3)
    transformations[2] = transformations[3] @ edge_2_to_3
    transformations[4] = np.linalg.inv(edge_0_to_4)

    fusion_started = time.perf_counter()
    merged, object_only, acceleration = merge_clouds_with_acceleration(
        cleaned_clouds,
        targets,
        transformations,
    )
    fusion_seconds = time.perf_counter() - fusion_started
    print(
        f"[Media] 融合完成：{fusion_seconds:.2f} 秒，后端：{acceleration['backend']}",
        flush=True,
    )

    stitched_path = OUTPUT / "media_five_clouds_stitched_with_calibration.ply"
    object_path = OUTPUT / "media_five_clouds_stitched_object_only.ply"
    o3d.io.write_point_cloud(str(stitched_path), merged, write_ascii=False)
    o3d.io.write_point_cloud(str(object_path), object_only, write_ascii=False)

    preview_started = time.perf_counter()
    preview_path = OUTPUT / "media_five_clouds_stitched_preview.png"
    save_preview(merged, preview_path)
    preview_seconds = time.perf_counter() - preview_started

    section_started = time.perf_counter()
    section = vertical_section(object_only)
    section_seconds = time.perf_counter() - section_started

    registration = []
    for path, target, transformation in zip(paths, targets, transformations):
        registration.append(
            {
                "file": path.name,
                "target_points": len(target.points),
                "transformation_to_reference": transformation.tolist(),
                "registration_to_reference": evaluate(
                    target,
                    targets[0],
                    transformation,
                    REGISTRATION_VOXEL,
                ),
            }
        )
    edge_registration = [
        {"source": paths[0].name, "target": paths[1].name, **edge_0_to_1_info},
        {"source": paths[0].name, "target": paths[3].name, **edge_0_to_3_info},
        {"source": paths[2].name, "target": paths[3].name, **edge_2_to_3_info},
        {"source": paths[0].name, "target": paths[4].name, **edge_0_to_4_info},
    ]
    report = {
        "inputs": cleaning,
        "reference_file": paths[0].name,
        "registration": registration,
        "edge_registration": edge_registration,
        "outputs": {
            "stitched_with_calibration": str(stitched_path),
            "stitched_object_only": str(object_path),
            "preview": str(preview_path),
            "stitched_points": len(merged.points),
            "object_only_points": len(object_only.points),
        },
        "vertical_section": section,
        "processing": {
            "accelerator": acceleration,
            "timing_seconds": {
                "input_read": read_seconds,
                "denoise": cleaning_seconds,
                "calibration_target": target_seconds,
                "feature": feature_seconds,
                "registration": registration_seconds,
                "fusion": fusion_seconds,
                "preview": preview_seconds,
                "vertical_section": section_seconds,
                "total": time.perf_counter() - total_started,
            },
        },
    }
    (OUTPUT / "processing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

if __name__=="__main__": main()



