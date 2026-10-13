import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
if (ROOT/'.deps').exists(): sys.path.insert(0, str(ROOT/'.deps'))
import ezdxf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import open3d as o3d
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

def sample_entity(e, step=.025):
    if e.dxftype()=='LINE':
        a=np.array([e.dxf.start.x,e.dxf.start.y]); b=np.array([e.dxf.end.x,e.dxf.end.y]); return np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/step)+1))
    if e.dxftype()=='ARC':
        c=np.array([e.dxf.center.x,e.dxf.center.y]); span=(e.dxf.end_angle-e.dxf.start_angle)%360; n=max(3,int(np.radians(span)*e.dxf.radius/step)+1); t=np.radians(np.linspace(e.dxf.start_angle,e.dxf.start_angle+span,n)); return c+e.dxf.radius*np.c_[np.cos(t),np.sin(t)]
    return np.empty((0,2))


def _stack_sampled_entities(sampled_entities, indices):
    parts = [
        sampled_entities[index]
        for index in (indices if indices is not None else [])
        if 0 <= int(index) < len(sampled_entities) and len(sampled_entities[int(index)])
    ]
    return np.vstack(parts) if parts else None


def _normalize_entity_path_collection(paths):
    if not isinstance(paths, (list, tuple, set)):
        return []
    values = list(paths)
    if not values:
        return []
    if all(not isinstance(value, (list, tuple, set)) for value in values):
        normalized = _unique_entity_indices(values)
        return [normalized] if normalized else []
    result = []
    for value in values:
        normalized = _unique_entity_indices(value)
        if normalized and normalized not in result and normalized[::-1] not in result:
            result.append(normalized)
    return result


def _stack_sampled_path(sampled_entities, indices):
    path_indices = _unique_entity_indices(indices)
    if not path_indices:
        return None
    geometries = []
    for index in path_indices:
        if not 0 <= index < len(sampled_entities):
            continue
        geometry = np.asarray(sampled_entities[index], dtype=float)
        if geometry.ndim != 2 or geometry.shape[1] != 2:
            continue
        geometry = geometry[np.isfinite(geometry).all(axis=1)]
        if len(geometry):
            geometries.append(geometry)
    if not geometries:
        return None

    orientation_costs = np.zeros((len(geometries), 2), dtype=float)
    backpointers = np.full((len(geometries), 2), -1, dtype=np.int64)
    for position in range(1, len(geometries)):
        current = geometries[position]
        current_start = current[0]
        current_end = current[-1]
        for current_orientation, current_point in enumerate((current_start, current_end)):
            best_previous = -1
            best_cost = float("inf")
            for previous_orientation in (0, 1):
                previous = geometries[position - 1]
                previous_point = previous[-1] if previous_orientation == 0 else previous[0]
                candidate_cost = orientation_costs[position - 1, previous_orientation] + float(
                    np.linalg.norm(previous_point - current_point)
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_previous = previous_orientation
            orientation_costs[position, current_orientation] = best_cost
            backpointers[position, current_orientation] = best_previous

    orientations = [0] * len(geometries)
    orientations[-1] = int(np.argmin(orientation_costs[-1]))
    for position in range(len(geometries) - 1, 0, -1):
        orientations[position - 1] = int(backpointers[position, orientations[position]])

    parts = []
    for geometry, orientation in zip(geometries, orientations):
        oriented = geometry if orientation == 0 else geometry[::-1]
        if parts and np.linalg.norm(parts[-1][-1] - oriented[0]) <= 1e-6:
            oriented = oriented[1:]
        if len(oriented):
            parts.append(oriented)
    return np.vstack(parts) if parts else None


def _blue_fit_anchor(sampled_entities, fit_indices):
    anchor = _stack_sampled_entities(sampled_entities, fit_indices)
    if anchor is None:
        raise RuntimeError('选中的 CAD 配准基准线没有可用几何')
    return anchor


def _normalize_cad_transform(cad_transform):
    if cad_transform is None:
        return None
    if isinstance(cad_transform, dict):
        center = cad_transform.get('center', cad_transform.get('cad_center'))
        mirror = cad_transform.get('mirror', cad_transform.get('mirror_x'))
        angle = cad_transform.get('angle_rad')
        if angle is None:
            angle_degrees = cad_transform.get('angle_deg')
            angle = None if angle_degrees is None else np.radians(float(angle_degrees))
        translation = cad_transform.get('translation', cad_transform.get('offset'))
        cad_transform = (center, mirror, angle, translation)
    if not isinstance(cad_transform, (tuple, list)) or len(cad_transform) != 4:
        raise RuntimeError('CAD 配准变换参数无效')
    center, mirror, angle, translation = cad_transform
    center = np.asarray(center, dtype=float).reshape(-1)
    translation = np.asarray(translation, dtype=float).reshape(-1)
    if center.shape != (2,) or translation.shape != (2,):
        raise RuntimeError('CAD 配准变换的中心或平移参数无效')
    mirror = float(mirror)
    angle = float(angle)
    if mirror not in (-1.0, 1.0) or not np.isfinite(center).all() or not np.isfinite(translation).all() or not np.isfinite(angle):
        raise RuntimeError('CAD 配准变换参数包含无效数值')
    return center, mirror, angle, translation

def transform(p,c,m,a,t):
    r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]); q=np.asarray(p)-c; q=q.copy(); q[:,0]*=m; return q@r.T+t

def inverse_transform(p, c, m, a, t):
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q = (np.asarray(p) - t) @ r
    q = q.copy()
    q[:, 0] *= m
    return q + c


def _angle_degrees(direction):
    direction = np.asarray(direction, dtype=float).reshape(2)
    return float(np.degrees(np.arctan2(direction[1], direction[0])))


def _weighted_circular_mean_degrees(values, weights=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        return None
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != values.shape or not np.isfinite(weights).all() or np.sum(weights) <= 0:
        return None
    radians = np.radians(values)
    angle = np.degrees(np.arctan2(
        np.sum(weights * np.sin(radians)),
        np.sum(weights * np.cos(radians)),
    ))
    return float((angle + 180.0) % 360.0 - 180.0)


def cad_registration_frame(cad_transform, registration_path):
    center, mirror, angle, translation = _normalize_cad_transform(cad_transform)
    basis_points = transform(np.asarray([[0., 0.], [1., 0.], [0., 1.]]),
                             center, mirror, angle, translation)
    matrix = np.eye(3)
    matrix[:2, :2] = (basis_points[1:] - basis_points[0]).T
    matrix[:2, 2] = basis_points[0]
    starts, ends, lengths, tangents, _ = _polyline_segments(registration_path)
    midpoints = (starts + ends) / 2
    origin = np.average(midpoints, axis=0, weights=lengths)
    radius = max(float(np.max(np.linalg.norm(midpoints - origin, axis=1))), 1e-12)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    relative = (midpoints - origin) / radius
    jacobian = np.column_stack((normals, -relative[:, 1] * normals[:, 0]
                               + relative[:, 0] * normals[:, 1]))
    information = (jacobian.T * (lengths / lengths.sum())) @ jacobian
    eigenvalues = np.maximum(np.linalg.eigvalsh(information), 0)
    direction_cross_products = [
        abs(float(left[0] * right[1] - left[1] * right[0]))
        for position, left in enumerate(tangents)
        for right in tangents[position + 1:]
    ]
    has_nonparallel_geometry = bool(
        direction_cross_products
        and max(direction_cross_products) > 1e-6
    )
    rank = 3 if has_nonparallel_geometry else int(
        np.count_nonzero(eigenvalues > max(eigenvalues[-1], 1e-12) * 1e-6)
    )
    return {
        'id': 'cad_original_xy',
        'axes': 'DXF 原图 +X/+Y，配准后映射到冻结截面',
        'origin_cad': [0.0, 0.0],
        'origin_in_section': basis_points[0].tolist(),
        'x_axis_in_section': (basis_points[1] - basis_points[0]).tolist(),
        'y_axis_in_section': (basis_points[2] - basis_points[0]).tolist(),
        'cad_to_section_matrix': matrix.tolist(),
        'section_to_cad_matrix': np.linalg.inv(matrix).tolist(),
        'mirror_applied': mirror < 0,
        'source_pointcloud_modified': False,
        'constraint_rank': rank,
        'constraint_eigenvalues': eigenvalues.tolist(),
        'fully_constrained': rank == 3,
        'constraint_method': 'selected_path_local_point_to_normal_jacobian',
        'constraint_scope': '局部约束检查，不证明全局对称姿态唯一',
        'warning': None if rank == 3 else (
            '配准路径不能独立约束全部二维位姿；沿线滑移或旋转可能未定，'
            '应增加非平行基准几何或独立定位特征，当前 X/Y 仅供参考'
        ),
    }


def section_profile_from_points(points, processing_report=None):
    """Project the final section with the coordinate contract stored in its report."""
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        return None
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if not len(coordinates) or not isinstance(processing_report, dict):
        return None

    section = processing_report.get('vertical_section', {})
    if not isinstance(section, dict):
        return None
    if section.get('coordinate_system') == '26xV01Q_section_local':
        return np.column_stack((coordinates[:, 0], coordinates[:, 2]))

    horizontal = np.asarray(section.get('horizontal_xyz', []), dtype=float)
    vertical = np.asarray(section.get('vertical_xyz', []), dtype=float)
    if horizontal.shape == (3,) and vertical.shape == (3,):
        if np.isfinite(horizontal).all() and np.isfinite(vertical).all():
            return np.column_stack((coordinates @ horizontal, coordinates @ vertical))

    axis = np.asarray(section.get('axis_xy', []), dtype=float)
    center = np.asarray(section.get('plane_center_xy', []), dtype=float)
    if axis.shape == (2,) and center.shape == (2,):
        if np.isfinite(axis).all() and np.isfinite(center).all() and np.linalg.norm(axis) > 1e-12:
            axis = axis / np.linalg.norm(axis)
            lateral = np.array([-axis[1], axis[0]])
            return np.column_stack(((coordinates[:, :2] - center) @ lateral, coordinates[:, 2]))
    return None


def validate_section_input_contract(pointcloud_path, processing_report=None):
    if not isinstance(processing_report, dict):
        return None
    processing = processing_report.get('processing', {})
    section = processing_report.get('vertical_section', {})
    if not isinstance(processing, dict) or processing.get('mode') != 'unified_final_section':
        return None
    if not isinstance(section, dict):
        raise RuntimeError('CAD 截面交接记录无效')
    expected_hash = str(section.get('analysis_section_sha256', '')).strip().lower()
    if not expected_hash:
        raise RuntimeError('CAD 截面交接记录缺少输入文件校验值')
    actual_hash = None
    if expected_hash:
        digest = hashlib.sha256()
        with Path(pointcloud_path).open('rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError('CAD 输入截面与最终融合截面交接记录不一致，请重新准备 CAD 分析截面')
    section_source = section.get('section_source')
    if section_source != 'merged_fused_cloud':
        raise RuntimeError('CAD 输入截面不是最终融合点云截面，已停止分析')
    coordinate_system = section.get('coordinate_system')
    if coordinate_system == '26xV01Q_final_section_xyz':
        if section.get('pointcloud_transform_applied') is not False:
            raise RuntimeError('CAD 输入截面未声明为原始最终点云，已停止分析')
        if section.get('coordinates_preserved_from_source') is not True:
            raise RuntimeError('CAD 输入截面坐标未声明为源截面坐标，已停止分析')
        if section.get('point_order_preserved') is not True:
            raise RuntimeError('CAD 输入截面点顺序未声明为源点顺序，已停止分析')
    return {
        'mode': processing.get('mode'),
        'section_source': section_source,
        'coordinate_system': coordinate_system,
        'pointcloud_transform_applied': section.get('pointcloud_transform_applied'),
        'coordinates_preserved_from_source': section.get('coordinates_preserved_from_source'),
        'point_order_preserved': section.get('point_order_preserved'),
        'analysis_section_sha256': actual_hash or expected_hash or None,
        'section_profile_sha256': section.get('section_profile_sha256'),
    }


def save_cad_alignment_image(
    path,
    pointcloud_profile,
    entities,
    fit_indices=None,
    x_indices=None,
    y_indices=None,
    mirror=1.0,
    angle=0.0,
    residual=0.0,
    alignment=None,
    cad_transform=None,
    registration_indices=None,
    measurement_indices=None,
    display_in_cad_coordinates=True,
):
    pointcloud_profile = np.asarray(pointcloud_profile, dtype=float)
    if pointcloud_profile.ndim != 2 or pointcloud_profile.shape[1] != 2 or pointcloud_profile.shape[0] == 0:
        raise RuntimeError('无法生成固定点云与 CAD 拟合对照图')
    normalized_transform = _normalize_cad_transform(cad_transform)
    display_pointcloud = pointcloud_profile
    display_coordinate_label = '固定点云截面坐标'
    if normalized_transform is not None and display_in_cad_coordinates:
        display_pointcloud = inverse_transform(pointcloud_profile, *normalized_transform)
        display_coordinate_label = 'CAD 原图坐标（仅转换坐标表达）'
    cad_parts = []
    for index, entity in enumerate(entities):
        geometry = sample_entity(entity)
        if len(geometry):
            if normalized_transform is not None:
                fit_center, fit_mirror, fit_angle, fit_translation = normalized_transform
                if not display_in_cad_coordinates:
                    geometry = transform(
                        geometry,
                        fit_center,
                        fit_mirror,
                        fit_angle,
                        fit_translation,
                    )
            cad_parts.append((index, geometry))
    if not cad_parts:
        raise RuntimeError('CAD 中没有可显示的线或圆弧')
    cad_points = np.vstack([geometry for _, geometry in cad_parts])
    limits = np.vstack([display_pointcloud, cad_points])
    minimum = limits.min(axis=0)
    maximum = limits.max(axis=0)
    padding = np.maximum((maximum - minimum) * 0.06, 0.12)
    figure, axis = plt.subplots(figsize=(12, 9), dpi=170, facecolor='white')
    axis.set_facecolor('white')
    axis.scatter(
        display_pointcloud[:, 0],
        display_pointcloud[:, 1],
        s=2.2,
        c='#f97316',
        alpha=0.88,
        linewidths=0,
        rasterized=True,
        label=f'最终点云（{display_coordinate_label}）',
    )
    labels_used = set()
    registration_set = {
        int(index)
        for index in (
            registration_indices if registration_indices is not None else (fit_indices or [])
        )
    }
    measurement_set = {
        int(index)
        for index in (
            measurement_indices
            if measurement_indices is not None
            else [*(x_indices or []), *(y_indices or [])]
        )
    }
    for index, geometry in cad_parts:
        if index in registration_set:
            color, width, alpha, label = '#7da2d8', 2.0, 0.82, 'CAD 配准基准线'
        elif index in measurement_set:
            color, width, alpha, label = '#69ad8e', 2.0, 0.82, 'CAD 测量线'
        else:
            color, width, alpha, label = '#cbd5e1', 0.45, 0.08, 'CAD 其他线段（淡化参考）'
        axis.plot(geometry[:, 0], geometry[:, 1], color=color, linewidth=width, alpha=alpha, label=label if label not in labels_used else None)
        labels_used.add(label)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
    axis.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])
    axis.set_xlabel('CAD X' if display_in_cad_coordinates else '截面横向坐标')
    axis.set_ylabel('CAD Y' if display_in_cad_coordinates else '截面竖向坐标')
    axis.set_title('最终点云 / CAD 叠加对照', fontsize=15, weight='bold')
    axis.grid(alpha=0.25)
    mirror_text = '是（CAD 左右镜像）' if mirror < 0 else '否'
    info_lines = [
        f'CAD\u2192\u70b9\u4e91\u65cb\u8f6c\u53c2\u6570\uff1a{np.degrees(angle):+.4f}\N{DEGREE SIGN}',
        f'\u663e\u793a\u5750\u6807\u7cfb\uff1a{display_coordinate_label}',
        '源截面文件和点序未修改',
        f'CAD \u955c\u50cf\uff1a{mirror_text}',
        f'配准基准线平均残差：{residual:.4f}',
    ]
    if alignment:
        info_lines.append(f'\u5168\u56fe CAD\u2192\u70b9\u4e91\u5e73\u5747\u8ddd\u79bb\uff1a{alignment["global_cad_to_pointcloud_mean"]:.4f}')
        info_lines.append(f'\u5168\u56fe\u53cc\u5411\u8ddd\u79bb\uff08\u70b9\u4e91\u2192CAD\uff09\uff1a{alignment["global_pointcloud_to_cad_mean"]:.4f}')
    info = '\n'.join(info_lines)
    axis.text(0.015, 0.985, info, transform=axis.transAxes, va='top', ha='left', fontsize=10.5, color='#17202a', bbox=dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor='#9aa8b2', alpha=0.92))
    axis.legend(loc='best', fontsize=9, framealpha=0.92)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)
    return path

def upright_display_rotation_degrees(points):
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        return 0.0
    direction = coordinates[-1] - coordinates[0]
    if np.linalg.norm(direction) <= 1e-12:
        centered = coordinates - coordinates.mean(axis=0)
        _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
        direction = right_singular_vectors[0]
    angle_degrees = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return float(((-angle_degrees + 90.0) % 180.0) - 90.0)


def _rotate_display_points(points, rotation_degrees, center):
    coordinates = np.asarray(points, dtype=float)
    radians = np.radians(rotation_degrees)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]],
        dtype=float,
    )
    return (coordinates - center) @ rotation.T + center


def _green_local_to_profile(coordinates, origin, x_axis, y_axis):
    local_coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 2))
    return (
        np.asarray(origin, dtype=float)[None, :]
        + local_coordinates[:, 0, None] * np.asarray(x_axis, dtype=float)[None, :]
        + local_coordinates[:, 1, None] * np.asarray(y_axis, dtype=float)[None, :]
    )


def save_height_colored_cad_overlay_image(
    path,
    pointcloud_cad,
    heights,
    entities,
    fit_indices=None,
    x_indices=None,
    y_indices=None,
    cad_transform=None,
    registration_indices=None,
    measurement_indices=None,
    display_in_cad_coordinates=True,
):
    if 'Microsoft YaHei' in {font.name for font in font_manager.fontManager.ttflist}:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    pointcloud_cad = np.asarray(pointcloud_cad, dtype=float)
    heights = np.asarray(heights, dtype=float).reshape(-1)
    if pointcloud_cad.ndim != 2 or pointcloud_cad.shape[1] != 2 or not len(pointcloud_cad):
        raise RuntimeError('无法生成高度彩色点云 / CAD 对照图')
    if len(heights) != len(pointcloud_cad):
        raise RuntimeError('高度颜色数量与 CAD 坐标点数量不一致')
    finite = np.isfinite(pointcloud_cad).all(axis=1) & np.isfinite(heights)
    pointcloud_cad = pointcloud_cad[finite]
    heights = heights[finite]
    if not len(pointcloud_cad):
        raise RuntimeError('高度彩色对照图没有有效点')

    normalized_transform = _normalize_cad_transform(cad_transform)
    display_pointcloud = pointcloud_cad
    display_coordinate_label = '配准坐标'
    if normalized_transform is not None and display_in_cad_coordinates:
        fit_center, fit_mirror, fit_angle, fit_translation = normalized_transform
        display_pointcloud = inverse_transform(
            pointcloud_cad,
            fit_center,
            fit_mirror,
            fit_angle,
            fit_translation,
        )
        display_coordinate_label = 'CAD 原图坐标'
    cad_parts = []
    for index, entity in enumerate(entities):
        geometry = sample_entity(entity)
        if len(geometry):
            if normalized_transform is not None:
                fit_center, mirror, angle, translation = normalized_transform
                if not display_in_cad_coordinates:
                    geometry = transform(geometry, fit_center, mirror, angle, translation)
            cad_parts.append((index, geometry))
    if not cad_parts:
        raise RuntimeError('CAD 中没有可显示的线或圆弧')
    selected_registration_set = {
        int(index)
        for index in (
            registration_indices if registration_indices is not None else (fit_indices or [])
        )
    }
    selected_measurement_set = {
        int(index)
        for index in (
            measurement_indices
            if measurement_indices is not None
            else [*(x_indices or []), *(y_indices or [])]
        )
    }
    selected_sets = selected_registration_set | selected_measurement_set
    selected_parts = [geometry for index, geometry in cad_parts if index in selected_sets]
    display_parts = [display_pointcloud]
    if selected_parts:
        display_parts.extend(selected_parts)
    else:
        display_parts.extend(geometry for _, geometry in cad_parts)
    display_coordinates = np.vstack(display_parts)
    display_minimum = display_coordinates.min(axis=0)
    display_maximum = display_coordinates.max(axis=0)
    display_span = np.maximum(display_maximum - display_minimum, 1e-6)
    display_padding = np.maximum(display_span * 0.08, 0.10)
    display_limits = (
        (float(display_minimum[0] - display_padding[0]), float(display_maximum[0] + display_padding[0])),
        (float(display_minimum[1] - display_padding[1]), float(display_maximum[1] + display_padding[1])),
    )
    height_low, height_high = np.percentile(heights, [1.0, 99.0])
    if not np.isfinite(height_low) or not np.isfinite(height_high):
        raise RuntimeError('点云高度数据无效，无法生成高度彩色渲染图')
    if height_high - height_low < 1e-12:
        height_high = height_low + 1.0
    height_normalizer = matplotlib.colors.Normalize(
        vmin=float(height_low),
        vmax=float(height_high),
        clip=True,
    )
    height_colormap = plt.get_cmap('turbo')
    height_colors = height_colormap(height_normalizer(heights))

    figure = plt.figure(figsize=(14.0, 7.4), dpi=180, facecolor='white')
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.18, 1.0),
        left=0.025,
        right=0.925,
        bottom=0.035,
        top=0.91,
        wspace=0.045,
    )
    full_axis = figure.add_subplot(grid[0, 0])
    focus_axis = figure.add_subplot(grid[0, 1])
    figure.suptitle(
        f'最终截面 / CAD 选中轮廓对照（{display_coordinate_label}）',
        fontsize=16,
        fontweight='bold',
        color='#17202a',
    )

    def draw_cad(axis, points, colors, limits, title, point_size, point_alpha):
        axis.set_facecolor('white')
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=point_size,
            c=colors,
            alpha=point_alpha,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        for index, geometry in cad_parts:
            if index in selected_registration_set:
                color, width, alpha, zorder = '#7da2d8', 1.8, 0.82, 6
            elif index in selected_measurement_set:
                color, width, alpha, zorder = '#69ad8e', 1.8, 0.82, 6
            else:
                color, width, alpha, zorder = '#cbd5e1', 0.45, 0.08, 1
            axis.plot(
                geometry[:, 0],
                geometry[:, 1],
                color=color,
                linewidth=width,
                alpha=alpha,
                solid_capstyle='round',
                zorder=zorder,
            )
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_aspect('equal', adjustable='box')
        axis.set_axis_off()
        axis.set_title(title, fontsize=10.5, fontweight='bold', pad=8, color='#334155')

    draw_cad(
        full_axis,
        display_pointcloud,
        height_colors,
        display_limits,
        f'全截面：{display_coordinate_label}',
        1.15,
        0.92,
    )
    if selected_parts:
        focus_coordinates = np.vstack(selected_parts)
        focus_minimum = focus_coordinates.min(axis=0)
        focus_maximum = focus_coordinates.max(axis=0)
        focus_span = np.maximum(focus_maximum - focus_minimum, 1e-6)
        focus_padding = np.maximum(focus_span * 0.16, 0.12)
        focus_limits = (
            (float(focus_minimum[0] - focus_padding[0]), float(focus_maximum[0] + focus_padding[0])),
            (float(focus_minimum[1] - focus_padding[1]), float(focus_maximum[1] + focus_padding[1])),
        )
        focus_mask = np.all(
            (display_pointcloud >= np.asarray([focus_limits[0][0], focus_limits[1][0]]))
            & (display_pointcloud <= np.asarray([focus_limits[0][1], focus_limits[1][1]])),
            axis=1,
        )
        if np.count_nonzero(focus_mask) >= 30:
            focus_points = display_pointcloud[focus_mask]
            focus_colors = height_colors[focus_mask]
        else:
            focus_points = display_pointcloud
            focus_colors = height_colors
            focus_limits = display_limits
    else:
        focus_points = display_pointcloud
        focus_colors = height_colors
        focus_limits = display_limits
    draw_cad(
        focus_axis,
        focus_points,
        focus_colors,
        focus_limits,
        '局部放大：选中轮廓与同一批点云',
        1.45,
        0.96,
    )
    legend_handles = [
        Line2D([0], [0], color='#f97316', linewidth=0.0, marker='o', markersize=5.5, label='最终点云（高度着色）'),
        Line2D([0], [0], color='#7da2d8', linewidth=1.8, alpha=0.82, label='CAD 配准基准线'),
        Line2D([0], [0], color='#69ad8e', linewidth=1.8, alpha=0.82, label='偏差与角度测量线'),
        Line2D([0], [0], color='#cbd5e1', linewidth=1.0, alpha=0.45, label='其他 CAD（淡化参考）'),
    ]
    full_axis.legend(handles=legend_handles, loc='lower left', fontsize=7.5, framealpha=0.92)
    color_axis = figure.add_axes([0.945, 0.18, 0.015, 0.29])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=height_normalizer, cmap=height_colormap),
        cax=color_axis,
    )
    colorbar.set_label('截面竖向坐标', fontsize=8, labelpad=4)
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.outline.set_linewidth(0.6)
    color_axis.set_facecolor((1.0, 1.0, 1.0, 0.78))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor='white', pad_inches=0)
    plt.close(figure)
    return path


def create_height_colored_cad_overlay_image(
    pointcloud_path,
    dxf_path,
    image_path,
    fit_entities=None,
    x_entities=None,
    y_entities=None,
    cad_transform=None,
    registration_entities=None,
    measurement_entities=None,
):
    cloud = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(cloud.points)
    if len(points) < 30:
        raise RuntimeError('点云点数太少，无法生成 CAD 高度彩色对照图')

    processing_report_path = Path(pointcloud_path).parent.parent / 'processing_report.json'
    processing_report = None
    if processing_report_path.exists():
        processing_report = json.loads(processing_report_path.read_text(encoding='utf-8'))
    validate_section_input_contract(pointcloud_path, processing_report)
    profile_candidates = []
    report_profile = section_profile_from_points(points, processing_report)
    if report_profile is not None:
        profile_candidates.append(report_profile)
    if not profile_candidates:
        center = np.median(points[:, :2], axis=0)
        _, directions = np.linalg.eigh(np.cov((points[:, :2] - center).T))
        for lateral in (directions[:, 0], directions[:, 1]):
            profile_candidates.append(np.column_stack(((points[:, :2] - center) @ lateral, points[:, 2])))

    document = ezdxf.readfile(str(dxf_path))
    entities = list(document.modelspace())
    sampled_entities = [sample_entity(entity) for entity in entities]
    cad_parts = [geometry for geometry in sampled_entities if len(geometry)]
    if not cad_parts:
        raise RuntimeError('DXF 中没有可用于配准的线或圆弧')
    cad_outline = np.vstack(cad_parts)
    selected_registration_indices = list(
        registration_entities if registration_entities is not None else (fit_entities or [])
    )
    selected_measurement_indices = list(
        measurement_entities
        if measurement_entities is not None
        else [*(x_entities or []), *(y_entities or [])]
    )
    selected_anchor = _stack_sampled_entities(sampled_entities, selected_registration_indices)
    saved_transform = _normalize_cad_transform(cad_transform)
    if saved_transform is not None and report_profile is not None:
        profile = report_profile
        fit_center, mirror, fit_angle, fit_translation = saved_transform
        fit_parameters = np.asarray([fit_angle, *fit_translation], dtype=float)
    else:
        candidates = []
        for profile in profile_candidates:
            fit_center, mirror, fit_parameters, distances, diagnostics = fit_profile_global(
                profile,
                cad_outline,
                selected_anchor,
            )
            candidates.append((
                diagnostics['global_score'],
                profile,
                fit_center,
                mirror,
                fit_parameters,
                distances,
            ))
        _, profile, fit_center, mirror, fit_parameters, _ = min(candidates, key=lambda item: item[0])
    return save_height_colored_cad_overlay_image(
        image_path,
        profile,
        profile[:, 1],
        entities,
        registration_indices=selected_registration_indices,
        measurement_indices=selected_measurement_indices,
        cad_transform=(fit_center, mirror, fit_parameters[0], fit_parameters[1:]),
    )


def fit_profile(profile, blue):
    grid=np.floor(profile/.025).astype(int); _,idx=np.unique(grid,axis=0,return_index=True); reduced=profile[idx]; tree=cKDTree(reduced); center=blue.mean(0); centered=blue-center
    def obj(v,m):
        d=tree.query(transform(centered,np.zeros(2),m,v[0],v[1:]))[0]; return np.mean(np.minimum(d,.6)**2)+.35*np.quantile(d,.8)**2+.15*np.quantile(d,.95)**2
    bounds=[(-np.pi,np.pi),(profile[:,0].min(),profile[:,0].max()),(profile[:,1].min(),profile[:,1].max())]; candidates=[]
    for m in (1.,-1.):
        r=differential_evolution(lambda v:obj(v,m),bounds,seed=20260824+int(m),popsize=8,maxiter=80,tol=1e-6,polish=True); d=tree.query(transform(centered,np.zeros(2),m,r.x[0],r.x[1:]))[0]; candidates.append((r.fun,m,r.x,d))
    score,m,v,d=min(candidates,key=lambda x:x[0]); return center,m,v,d


def _grid_reduce(points, cell_size):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise RuntimeError('\u8f6e\u5ed3\u70b9\u6570\u636e\u65e0\u6548')
    grid = np.floor(points / cell_size).astype(np.int64)
    _, indices = np.unique(grid, axis=0, return_index=True)
    return points[indices]


def _trimmed_distance_mean(distances, keep_ratio):
    distances = np.asarray(distances, dtype=float)
    if not len(distances):
        return float('inf')
    cutoff = float(np.quantile(distances, keep_ratio))
    return float(np.mean(np.minimum(distances, cutoff)))


def _select_profile_pose(candidates, centered_anchor, profile_tree):
    """Use outline support only among poses with comparable datum residuals."""
    if centered_anchor is None:
        return min(candidates, key=lambda candidate: candidate[0]), []
    metrics = []
    for score, mirror, parameters, source in candidates:
        distances = profile_tree.query(transform(
            centered_anchor, np.zeros(2), mirror, parameters[0], parameters[1:]
        ))[0]
        metrics.append(dict(source=source, mirror=float(mirror),
                            angle_deg=float(np.degrees(parameters[0])),
                            global_score=float(score),
                            anchor_mean=float(np.mean(distances)),
                            anchor_p95=float(np.quantile(distances, .95))))
    best = min(metrics, key=lambda item: (item["anchor_mean"], item["anchor_p95"]))
    # One optimizer grid cell of mean error (two at P95) is the
    # discretization allowance, not a relaxation of measurement quality gates.
    for item in metrics:
        item["eligible"] = bool(
            item["anchor_mean"] <= best["anchor_mean"] + .035
            and item["anchor_p95"] <= best["anchor_p95"] + .070
        )
    eligible = [candidate for candidate, item in zip(candidates, metrics) if item["eligible"]]
    return min(eligible, key=lambda candidate: candidate[0]), metrics


def fit_profile_global(profile, cad_outline, selected_anchor=None):
    """Fit the whole CAD outline into the fixed point-cloud profile frame."""
    profile = np.asarray(profile, dtype=float)
    cad_outline = np.asarray(cad_outline, dtype=float)
    if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 30:
        raise RuntimeError('\u7528\u4e8e\u8f6e\u5ed3\u914d\u51c6\u7684\u70b9\u4e91\u4e0d\u8db3')
    if cad_outline.ndim != 2 or cad_outline.shape[1] != 2 or len(cad_outline) < 3:
        raise RuntimeError('CAD \u8f6e\u5ed3\u4e0d\u8db3\uff0c\u65e0\u6cd5\u8fdb\u884c\u5168\u56fe\u914d\u51c6')

    optimizer_profile = _grid_reduce(profile, .035)
    optimizer_cad = _grid_reduce(cad_outline, .035)
    anchor = None
    if selected_anchor is not None and len(selected_anchor):
        anchor = _grid_reduce(selected_anchor, .035)

    center = cad_outline.mean(axis=0)
    centered_cad = optimizer_cad - center
    centered_anchor = anchor - center if anchor is not None else None
    profile_tree = cKDTree(optimizer_profile)
    profile_span = np.ptp(optimizer_profile, axis=0)
    cad_span = np.ptp(optimizer_cad, axis=0)
    margin = max(float(profile_span.max()), float(cad_span.max()), .5) * .20
    bounds = [
        (-np.pi, np.pi),
        (float(optimizer_profile[:, 0].min() - margin), float(optimizer_profile[:, 0].max() + margin)),
        (float(optimizer_profile[:, 1].min() - margin), float(optimizer_profile[:, 1].max() + margin)),
    ]

    def transformed(points, parameters, mirror):
        return transform(points, np.zeros(2), mirror, parameters[0], parameters[1:])

    def objective(parameters, mirror):
        transformed_cad = transformed(centered_cad, parameters, mirror)
        cad_to_profile = profile_tree.query(transformed_cad)[0]
        pointcloud_to_cad = cKDTree(transformed_cad).query(optimizer_profile)[0]
        score = .44 * _trimmed_distance_mean(cad_to_profile, .65)
        score += .44 * _trimmed_distance_mean(pointcloud_to_cad, .65)
        if centered_anchor is not None:
            anchor_to_profile = profile_tree.query(transformed(centered_anchor, parameters, mirror))[0]
            score += .12 * _trimmed_distance_mean(anchor_to_profile, .85)
        return score

    candidates = []
    global_search_seeds = []
    for mirror in (1., -1.):
        # The trimmed bidirectional objective is intentionally robust to
        # outliers, but that also makes it mildly multimodal for sparse,
        # nearly symmetric sections.  A single differential-evolution run can
        # therefore settle on a visibly wrong pose even when the same profile
        # has a much better basin.  Keep the original seed and add one fixed
        # restart per mirror so verification and production runs remain
        # deterministic while covering the common local-minimum split.
        for seed in (20260824 + int(mirror), 20260827 + int(mirror)):
            result = differential_evolution(
                lambda parameters: objective(parameters, mirror),
                bounds,
                seed=seed,
                popsize=10,
                maxiter=70,
                tol=2e-5,
                polish=True,
            )
            candidates.append((float(result.fun), mirror, result.x, 'global_search'))
            global_search_seeds.append(int(seed))

    global_search_score = min(candidate[0] for candidate in candidates)
    selected_path_seed_score = None
    if anchor is not None and len(anchor):
        anchor_center, anchor_mirror, anchor_parameters, _ = fit_profile(profile, anchor)
        anchor_translation = transform(
            np.asarray([center], dtype=float),
            anchor_center,
            anchor_mirror,
            anchor_parameters[0],
            anchor_parameters[1:],
        )[0]
        anchor_seed = np.asarray(
            [anchor_parameters[0], anchor_translation[0], anchor_translation[1]],
            dtype=float,
        )
        selected_path_seed_score = float(objective(anchor_seed, anchor_mirror))
        candidates.append(
            (
                selected_path_seed_score,
                anchor_mirror,
                anchor_seed,
                'selected_registration_path',
            )
        )

    selected, candidate_diagnostics = _select_profile_pose(
        candidates, centered_anchor, profile_tree,
    )
    score, mirror, parameters, initialization = selected
    selected_refinement_applied = False
    selected_refinement_before = None
    selected_refinement_after = None
    if anchor is not None and len(anchor):
        initial_anchor = transformed(centered_anchor, parameters, mirror)
        local_padding = max(float(np.ptp(initial_anchor, axis=0).max()) * 0.08, 0.25)
        local_minimum = initial_anchor.min(axis=0) - local_padding
        local_maximum = initial_anchor.max(axis=0) + local_padding
        local_mask = np.all((optimizer_profile >= local_minimum) & (optimizer_profile <= local_maximum), axis=1)
        local_profile = optimizer_profile[local_mask]
        if len(local_profile) >= 30:
            local_profile_tree = cKDTree(local_profile)
            local_anchor_tree = cKDTree(initial_anchor)

            def selected_objective(local_parameters):
                transformed_anchor = transformed(centered_anchor, local_parameters, mirror)
                anchor_to_profile = local_profile_tree.query(transformed_anchor)[0]
                profile_to_anchor = cKDTree(transformed_anchor).query(local_profile)[0]
                return (
                    0.65 * _trimmed_distance_mean(anchor_to_profile, 0.85)
                    + 0.35 * _trimmed_distance_mean(profile_to_anchor, 0.85)
                )

            local_bounds = [
                (float(parameters[0] - np.radians(3.0)), float(parameters[0] + np.radians(3.0))),
                (float(parameters[1] - 0.25), float(parameters[1] + 0.25)),
                (float(parameters[2] - 0.25), float(parameters[2] + 0.25)),
            ]
            selected_refinement_before = float(selected_objective(parameters))
            refinement = minimize(
                selected_objective,
                parameters,
                method='Powell',
                bounds=local_bounds,
                options={'maxiter': 240, 'xtol': 1e-7, 'ftol': 1e-7},
            )
            selected_refinement_after = float(selected_objective(refinement.x))
            if refinement.success and selected_refinement_after <= selected_refinement_before:
                parameters = refinement.x
                selected_refinement_applied = True
    score = float(objective(parameters, mirror))
    full_profile = _grid_reduce(profile, .025)
    full_cad = _grid_reduce(cad_outline, .025)
    transformed_full_cad = transformed(full_cad - center, parameters, mirror)
    cad_to_profile = cKDTree(full_profile).query(transformed_full_cad)[0]
    pointcloud_to_cad = cKDTree(transformed_full_cad).query(full_profile)[0]
    if selected_anchor is not None and len(selected_anchor):
        anchor_distances = cKDTree(full_profile).query(
            transform(np.asarray(selected_anchor) - center, np.zeros(2), mirror, parameters[0], parameters[1:])
        )[0]
    else:
        anchor_distances = cad_to_profile
    diagnostics = {
        'method': 'global_bidirectional_trimmed_chamfer',
        'anchor_used': bool(selected_anchor is not None and len(selected_anchor)),
        'global_score': float(score),
        'initialization': initialization,
        'global_search_score': float(global_search_score),
        'global_search_seeds': global_search_seeds,
        'selected_path_seed_score': selected_path_seed_score,
        'pose_selection': 'datum_residual_gate_then_outline_support',
        'pose_candidates': candidate_diagnostics,
        'global_cad_to_pointcloud_mean': float(np.mean(cad_to_profile)),
        'global_cad_to_pointcloud_p95': float(np.quantile(cad_to_profile, .95)),
        'global_pointcloud_to_cad_mean': float(np.mean(pointcloud_to_cad)),
        'global_pointcloud_to_cad_p95': float(np.quantile(pointcloud_to_cad, .95)),
        'optimizer_pointcloud_count': int(len(optimizer_profile)),
        'optimizer_cad_count': int(len(optimizer_cad)),
        'selected_refinement_applied': selected_refinement_applied,
        'selected_refinement_before': selected_refinement_before,
        'selected_refinement_after': selected_refinement_after,
    }
    return center, mirror, parameters, anchor_distances, diagnostics

def line_angle(p,a,b):
    d=b-a; d=d/np.linalg.norm(d); n=np.array([-d[1],d[0]]); q=p-a; s=q@d; selected=p[(s>=-.08)&(s<=np.linalg.norm(b-a)+.08)&(np.abs(q@n)<=.15)]
    if len(selected)<20: raise RuntimeError('CAD 测量线附近点云不足，无法拟合角度')
    c=selected.mean(0); _,v=np.linalg.eigh(np.cov((selected-c).T)); f=v[:,-1]; f=f if f@d>=0 else -f; res=(selected-c)@np.array([-f[1],f[0]]); ang=np.degrees(np.arctan2(f[1],f[0])-np.arctan2(d[1],d[0])); return (ang+180)%360-180, np.sqrt(np.mean(res**2)), len(selected)

def measure(p,a,b,step):
    d=b-a; length=np.linalg.norm(d); d=d/length; is_x=abs(b[0]-a[0])>=abs(b[1]-a[1]); sa,da=(0,1) if is_x else (1,0); axis='Y' if is_x else 'X'; sign=1 if (b-a)[sa]>=0 else -1; stations=np.arange(step/2,length,step); sc=sign*(p[:,sa]-a[sa]); rows=[]
    for s in stations:
        cad=a+s*d; mask=(np.abs(sc-s)<=step/2)&(np.abs(p[:,da]-cad[da])<=.15); ok=np.count_nonzero(mask)>=3; dev=float(np.median(p[mask,da]-cad[da])) if ok else None; pc=cad.copy(); pc[da]+=dev if ok else 0; rows.append({'station':float(s),'cad_x':float(cad[0]),'cad_y':float(cad[1]),'pointcloud_x':float(pc[0]),'pointcloud_y':float(pc[1]),'measurement_axis':axis,'signed_axis_deviation':dev,'absolute_axis_deviation':abs(dev) if ok else None,'valid':ok})
    vals=np.array([r['signed_axis_deviation'] for r in rows if r['valid']]); valid=[r for r in rows if r['valid']]
    if not len(vals): raise RuntimeError('绿色边附近没有有效点云')
    lo=min(valid,key=lambda r:r['signed_axis_deviation']); hi=max(valid,key=lambda r:r['signed_axis_deviation']); return rows,{'length':float(length),'measurement_axis':axis,'valid_stations':len(vals),'total_stations':len(rows),'coverage_ratio':len(vals)/len(rows),'mean_signed':float(vals.mean()),'median_signed':float(np.median(vals)),'mean_absolute':float(np.abs(vals).mean()),'rms':float(np.sqrt(np.mean(vals**2))),'p95_absolute':float(np.quantile(np.abs(vals),.95)),'max_absolute':float(np.abs(vals).max()),'min_signed':float(lo['signed_axis_deviation']),'min_station':float(lo['station']),'max_signed':float(hi['signed_axis_deviation']),'max_station':float(hi['station'])}

def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def _unique_entity_indices(values):
    result = []
    for value in values or []:
        if isinstance(value, bool):
            continue
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index not in result:
            result.append(index)
    return result


def _empty_deviation_statistics(length, total_stations, measurement_axis, reference, **extra):
    return {
        'length': float(length),
        'measurement_axis': measurement_axis,
        'deviation_reference': reference,
        'valid_stations': 0,
        'total_stations': int(total_stations),
        'coverage_ratio': 0.0,
        'mean_signed': None,
        'median_signed': None,
        'mean_absolute': None,
        'rms': None,
        'p95_absolute': None,
        'max_absolute': None,
        'min_signed': None,
        'min_station': None,
        'max_signed': None,
        'max_station': None,
        **extra,
    }


def _with_cartesian_deviation_statistics(rows, statistics):
    valid_rows = []
    for row in rows:
        cad = np.asarray([row.get('cad_x'), row.get('cad_y')], dtype=float)
        pointcloud = np.asarray([row.get('pointcloud_x'), row.get('pointcloud_y')], dtype=float)
        valid = bool(row.get('valid')) and np.isfinite(cad).all() and np.isfinite(pointcloud).all()
        if valid:
            delta = pointcloud - cad
            row['signed_x_deviation'] = float(delta[0])
            row['signed_y_deviation'] = float(delta[1])
            row['absolute_x_deviation'] = float(abs(delta[0]))
            row['absolute_y_deviation'] = float(abs(delta[1]))
            valid_rows.append(row)
        else:
            row['signed_x_deviation'] = None
            row['signed_y_deviation'] = None
            row['absolute_x_deviation'] = None
            row['absolute_y_deviation'] = None

    if not valid_rows:
        statistics.update({
            'mean_signed_x': None,
            'mean_signed_y': None,
            'mean_absolute_x': None,
            'mean_absolute_y': None,
            'rms_x': None,
            'rms_y': None,
            'p95_absolute_x': None,
            'p95_absolute_y': None,
            'max_absolute_x': None,
            'max_absolute_y': None,
            'deviation_coordinate_system': 'CAD 配准坐标系 X/Y',
            'deviation_sign_convention_xy': '正值表示点云坐标分量减 CAD 坐标分量',
            'deviation_vector_definition': '测量点 = CAD 站位点 + 局部左法线有符号偏差；X/Y 为该偏差向量在 CAD 原图轴上的分量',
        })
        return statistics

    delta = np.asarray([
        [row['signed_x_deviation'], row['signed_y_deviation']]
        for row in valid_rows
    ], dtype=float)
    statistics.update({
        'mean_signed_x': float(delta[:, 0].mean()),
        'mean_signed_y': float(delta[:, 1].mean()),
        'mean_absolute_x': float(np.abs(delta[:, 0]).mean()),
        'mean_absolute_y': float(np.abs(delta[:, 1]).mean()),
        'rms_x': float(np.sqrt(np.mean(delta[:, 0] ** 2))),
        'rms_y': float(np.sqrt(np.mean(delta[:, 1] ** 2))),
        'p95_absolute_x': float(np.quantile(np.abs(delta[:, 0]), .95)),
        'p95_absolute_y': float(np.quantile(np.abs(delta[:, 1]), .95)),
        'max_absolute_x': float(np.abs(delta[:, 0]).max()),
        'max_absolute_y': float(np.abs(delta[:, 1]).max()),
        'deviation_coordinate_system': 'CAD 配准坐标系 X/Y',
        'deviation_sign_convention_xy': '正值表示点云坐标分量减 CAD 坐标分量',
        'deviation_vector_definition': '测量点 = CAD 站位点 + 局部左法线有符号偏差；X/Y 为该偏差向量在 CAD 原图轴上的分量',
    })
    return statistics


def measure_line(points, start, end, step, search_half_width=.15, allow_empty=False):
    points = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float).reshape(2)
    end = np.asarray(end, dtype=float).reshape(2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise RuntimeError('测量线点云坐标无效')
    if not np.isfinite(points).all() or not np.isfinite(start).all() or not np.isfinite(end).all():
        raise RuntimeError('测量线包含无效坐标')
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise RuntimeError('CAD 测量线长度为零')
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        raise RuntimeError('偏差采样间隔必须大于零')

    tangent = direction / length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    relative = points - start
    longitudinal_coordinates = relative @ tangent
    normal_coordinates = relative @ normal
    station_count = max(1, int(np.ceil(length / step)))
    station_spacing = length / station_count
    stations = (np.arange(station_count, dtype=float) + .5) * station_spacing
    rows = []
    for station in stations:
        cad_point = start + station * tangent
        base_half_width = station_spacing / 2.0
        selected_half_width = base_half_width
        mask = np.zeros(len(points), dtype=bool)
        for factor in (1.0, 2.0, 3.0):
            selected_half_width = base_half_width * factor
            mask = (
                (np.abs(longitudinal_coordinates - station) <= selected_half_width + 1e-12)
                & (np.abs(normal_coordinates) <= float(search_half_width))
            )
            if np.count_nonzero(mask) >= 3:
                break
        sample_point_count = int(np.count_nonzero(mask))
        valid = sample_point_count >= 3
        deviation = float(np.median(normal_coordinates[mask])) if valid else None
        pointcloud_point = cad_point + (deviation if valid else 0.0) * normal
        rows.append(
            {
                'station': float(station),
                'cad_x': float(cad_point[0]),
                'cad_y': float(cad_point[1]),
                'pointcloud_x': float(pointcloud_point[0]),
                'pointcloud_y': float(pointcloud_point[1]),
                'measurement_axis': 'CAD 左法线',
                'signed_normal_deviation': deviation,
                'absolute_normal_deviation': abs(deviation) if valid else None,
                'signed_axis_deviation': deviation,
                'absolute_axis_deviation': abs(deviation) if valid else None,
                'sample_point_count': sample_point_count,
                'longitudinal_half_width': float(selected_half_width),
                'search_half_width': float(search_half_width),
                'valid': bool(valid),
            }
        )

    valid_rows = [row for row in rows if row['valid']]
    values = np.asarray([row['signed_normal_deviation'] for row in valid_rows], dtype=float)
    if not len(values):
        if not allow_empty:
            raise RuntimeError('CAD 测量线附近没有有效点云')
        empty_statistics = _empty_deviation_statistics(
            length,
            len(rows),
            'CAD 左法线',
            '点云相对于 CAD 线段起点→终点左法线方向',
            cad_tangent=tangent.tolist(),
            cad_left_normal=normal.tolist(),
            station_minimum_points=3,
            adaptive_station_count=0,
            maximum_longitudinal_half_width=float(station_spacing * 1.5),
        )
        return rows, _with_cartesian_deviation_statistics(rows, empty_statistics)
    minimum_row = min(valid_rows, key=lambda row: row['signed_normal_deviation'])
    maximum_row = max(valid_rows, key=lambda row: row['signed_normal_deviation'])
    return rows, _with_cartesian_deviation_statistics(rows, {
        'length': length,
        'measurement_axis': 'CAD 左法线',
        'deviation_reference': '点云相对于 CAD 线段起点→终点左法线方向',
        'cad_tangent': tangent.tolist(),
        'cad_left_normal': normal.tolist(),
        'valid_stations': len(values),
        'total_stations': len(rows),
        'coverage_ratio': len(values) / len(rows),
        'mean_signed': float(values.mean()),
        'median_signed': float(np.median(values)),
        'mean_absolute': float(np.abs(values).mean()),
        'rms': float(np.sqrt(np.mean(values ** 2))),
        'p95_absolute': float(np.quantile(np.abs(values), .95)),
        'max_absolute': float(np.abs(values).max()),
        'min_signed': float(minimum_row['signed_normal_deviation']),
        'min_station': float(minimum_row['station']),
        'max_signed': float(maximum_row['signed_normal_deviation']),
        'max_station': float(maximum_row['station']),
        'station_minimum_points': 3,
        'adaptive_station_count': int(sum(
            row['valid'] and row['longitudinal_half_width'] > station_spacing / 2.0 + 1e-12
            for row in rows
        )),
        'maximum_longitudinal_half_width': float(max(
            row['longitudinal_half_width'] for row in rows if row['valid']
        )),
    })


def _polyline_segments(polyline):
    coordinates = np.asarray(polyline, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise RuntimeError('CAD 连续路径几何无效')
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if len(coordinates) < 2:
        raise RuntimeError('CAD 连续路径至少需要两个几何点')
    kept = [coordinates[0]]
    for point in coordinates[1:]:
        if np.linalg.norm(point - kept[-1]) > 1e-10:
            kept.append(point)
    coordinates = np.asarray(kept, dtype=float)
    if len(coordinates) < 2:
        raise RuntimeError('CAD 连续路径长度为零')
    segment_vectors = np.diff(coordinates, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid = segment_lengths > 1e-10
    if not np.any(valid):
        raise RuntimeError('CAD 连续路径长度为零')
    starts = coordinates[:-1][valid]
    ends = coordinates[1:][valid]
    segment_vectors = segment_vectors[valid]
    segment_lengths = segment_lengths[valid]
    tangents = segment_vectors / segment_lengths[:, None]
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return starts, ends, segment_lengths, tangents, cumulative


def measure_polyline(points, polyline, step, search_half_width=.15, allow_empty=False):
    starts, ends, segment_lengths, tangents, cumulative = _polyline_segments(polyline)
    if len(starts) == 1:
        return measure_line(
            points,
            starts[0],
            ends[0],
            step,
            search_half_width,
            allow_empty=allow_empty,
        )

    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise RuntimeError('测量路径点云坐标无效')
    if not np.isfinite(coordinates).all():
        raise RuntimeError('测量路径点云包含无效坐标')
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        raise RuntimeError('偏差采样间隔必须大于零')
    total_length = float(cumulative[-1])
    station_count = max(1, int(np.ceil(total_length / step)))
    station_spacing = total_length / station_count
    stations = (np.arange(station_count, dtype=float) + 0.5) * station_spacing
    path_vertices = np.vstack((starts[0], ends))
    nearest_path_distances, nearest_path_vertices = cKDTree(path_vertices).query(coordinates)
    point_path_stations = cumulative[nearest_path_vertices]
    rows = []
    for station in stations:
        segment_index = int(np.searchsorted(cumulative, station, side='right') - 1)
        segment_index = min(max(segment_index, 0), len(starts) - 1)
        local_station = station - cumulative[segment_index]
        tangent = tangents[segment_index]
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        cad_point = starts[segment_index] + local_station * tangent
        relative = coordinates - cad_point
        normal_coordinates = relative @ normal
        base_half_width = station_spacing / 2.0
        selected_half_width = base_half_width
        mask = np.zeros(len(coordinates), dtype=bool)
        for factor in (1.0, 2.0, 3.0):
            selected_half_width = base_half_width * factor
            mask = (
                (np.abs(point_path_stations - station) <= selected_half_width + 1e-12)
                & (nearest_path_distances <= float(search_half_width))
            )
            if np.count_nonzero(mask) >= 3:
                break
        sample_point_count = int(np.count_nonzero(mask))
        valid = sample_point_count >= 3
        deviation = float(np.median(normal_coordinates[mask])) if valid else None
        pointcloud_point = cad_point + (deviation if valid else 0.0) * normal
        rows.append(
            {
                'station': float(station),
                'segment_index': segment_index,
                'cad_x': float(cad_point[0]),
                'cad_y': float(cad_point[1]),
                'pointcloud_x': float(pointcloud_point[0]),
                'pointcloud_y': float(pointcloud_point[1]),
                'measurement_axis': 'CAD 左法线',
                'signed_normal_deviation': deviation,
                'absolute_normal_deviation': abs(deviation) if valid else None,
                'signed_axis_deviation': deviation,
                'absolute_axis_deviation': abs(deviation) if valid else None,
                'sample_point_count': sample_point_count,
                'longitudinal_half_width': float(selected_half_width),
                'search_half_width': float(search_half_width),
                'valid': bool(valid),
            }
        )

    valid_rows = [row for row in rows if row['valid']]
    values = np.asarray([row['signed_normal_deviation'] for row in valid_rows], dtype=float)
    if not len(values):
        if not allow_empty:
            raise RuntimeError('CAD 连续测量路径附近没有有效点云')
        empty_statistics = _empty_deviation_statistics(
            total_length,
            len(rows),
            'CAD 左法线（连续路径）',
            '点云相对于 CAD 连续路径各局部线段起点→终点左法线方向',
            path_length=total_length,
            segment_count=int(len(starts)),
            station_minimum_points=3,
            adaptive_station_count=0,
            maximum_longitudinal_half_width=float(station_spacing * 1.5),
        )
        return rows, _with_cartesian_deviation_statistics(rows, empty_statistics)
    minimum_row = min(valid_rows, key=lambda row: row['signed_normal_deviation'])
    maximum_row = max(valid_rows, key=lambda row: row['signed_normal_deviation'])
    return rows, _with_cartesian_deviation_statistics(rows, {
        'length': total_length,
        'path_length': total_length,
        'segment_count': int(len(starts)),
        'measurement_axis': 'CAD 左法线（连续路径）',
        'deviation_reference': '点云相对于 CAD 连续路径各局部线段起点→终点左法线方向',
        'valid_stations': len(values),
        'total_stations': len(rows),
        'coverage_ratio': len(values) / len(rows),
        'mean_signed': float(values.mean()),
        'median_signed': float(np.median(values)),
        'mean_absolute': float(np.abs(values).mean()),
        'rms': float(np.sqrt(np.mean(values ** 2))),
        'p95_absolute': float(np.quantile(np.abs(values), .95)),
        'max_absolute': float(np.abs(values).max()),
        'min_signed': float(minimum_row['signed_normal_deviation']),
        'min_station': float(minimum_row['station']),
        'max_signed': float(maximum_row['signed_normal_deviation']),
        'max_station': float(maximum_row['station']),
        'station_minimum_points': 3,
        'adaptive_station_count': int(sum(
            row['valid'] and row['longitudinal_half_width'] > station_spacing / 2.0 + 1e-12
            for row in rows
        )),
        'maximum_longitudinal_half_width': float(max(
            row['longitudinal_half_width'] for row in rows if row['valid']
        )),
    })


def _principal_line_fit(points):
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3:
        return None
    center = coordinates.mean(axis=0)
    centered = coordinates - center
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    span = float(np.ptp(centered @ direction))
    if not np.isfinite(span) or span <= 1e-10:
        return None
    normal = np.asarray([-direction[1], direction[0]], dtype=float)
    residuals = centered @ normal
    denominator = max(float(eigenvalues[-2]), 1e-12)
    return {
        'center': center,
        'direction': direction,
        'span': span,
        'rms': float(np.sqrt(np.mean(residuals ** 2))),
        'linearity': float(max(0.0, eigenvalues[-1]) / denominator),
    }


def fit_polyline_angle(
    points,
    polyline,
    search_half_width=.15,
    minimum_points=8,
    minimum_span=.10,
    maximum_window_span=.60,
):
    """Fit local point-cloud/CAD tangent differences inside a bounded corridor.

    DXF entities are sampled into many short chords. Treating every chord as an
    independent line makes the required point count depend on sampling density,
    while joining the first and last point invents one global angle for a curved
    path. This routine assigns each measured point to the sampled path and fits
    non-overlapping local windows instead.
    """
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise RuntimeError('测量路径点云坐标无效')
    if not np.isfinite(coordinates).all():
        raise RuntimeError('测量路径点云包含无效坐标')
    search_half_width = float(search_half_width)
    minimum_points = int(minimum_points)
    minimum_span = float(minimum_span)
    maximum_window_span = float(maximum_window_span)
    if not np.isfinite(search_half_width) or search_half_width <= 0:
        raise RuntimeError('角度拟合邻域必须大于零')
    if minimum_points < 3:
        raise RuntimeError('角度拟合最少点数不能小于 3')
    if (
        not np.isfinite(minimum_span)
        or not np.isfinite(maximum_window_span)
        or minimum_span <= 0
        or maximum_window_span < minimum_span
    ):
        raise RuntimeError('角度拟合窗口范围无效')

    starts, ends, segment_lengths, tangents, cumulative = _polyline_segments(polyline)
    vertices = np.vstack((starts[0], ends))
    path_length = float(cumulative[-1])
    nearest_distances, nearest_vertices = cKDTree(vertices).query(coordinates)
    support_mask = nearest_distances <= search_half_width
    support_indices = np.flatnonzero(support_mask)
    support_stations = cumulative[nearest_vertices[support_indices]]
    result = {
        'status': 'INSUFFICIENT_DATA',
        'method': 'bounded_local_polyline_pca',
        'angle_difference_deg': None,
        'cad_angle_deg': None,
        'pointcloud_angle_deg': None,
        'rms': None,
        'point_count': 0,
        'candidate_point_count': int(len(support_indices)),
        'path_length': path_length,
        'fitted_path_span': 0.0,
        'search_half_width': search_half_width,
        'minimum_points': minimum_points,
        'minimum_span': minimum_span,
        'maximum_window_span': maximum_window_span,
        'support_gap_limit': max(.20, search_half_width * 2.0),
        'segments': [],
        'reason': None,
    }
    if len(support_indices) < minimum_points:
        result['reason'] = (
            f'CAD 测量路径有界邻域内仅 {len(support_indices)} 个点，'
            f'少于局部角度拟合所需的 {minimum_points} 个点'
        )
        return result

    order = np.argsort(support_stations, kind='stable')
    support_indices = support_indices[order]
    support_stations = support_stations[order]
    gaps = np.diff(support_stations)
    split_offsets = np.flatnonzero(gaps > result['support_gap_limit']) + 1
    support_groups = np.split(np.arange(len(support_indices)), split_offsets)
    measurements = []
    for group in support_groups:
        start_position = 0
        while start_position < len(group):
            first_station = support_stations[group[start_position]]
            end_position = int(
                np.searchsorted(
                    support_stations[group],
                    first_station + maximum_window_span,
                    side='right',
                )
            )
            if end_position <= start_position:
                end_position = start_position + 1
            local_positions = group[start_position:end_position]
            local_stations = support_stations[local_positions]
            local_span = float(np.ptp(local_stations)) if len(local_stations) else 0.0
            if len(local_positions) < minimum_points or local_span < minimum_span:
                start_position += 1
                continue

            point_indices = support_indices[local_positions]
            cad_points = vertices[nearest_vertices[point_indices]]
            cloud_fit = _principal_line_fit(coordinates[point_indices])
            cad_fit = _principal_line_fit(cad_points)
            if cloud_fit is None or cad_fit is None:
                start_position = end_position
                continue

            reference_direction = cad_points[-1] - cad_points[0]
            if np.linalg.norm(reference_direction) <= 1e-10:
                middle_vertex = int(np.median(nearest_vertices[point_indices]))
                reference_direction = tangents[min(middle_vertex, len(tangents) - 1)]
            cad_direction = np.asarray(cad_fit['direction'], dtype=float)
            if cad_direction @ reference_direction < 0:
                cad_direction = -cad_direction
            cloud_direction = np.asarray(cloud_fit['direction'], dtype=float)
            if cloud_direction @ cad_direction < 0:
                cloud_direction = -cloud_direction
            angle = float(
                np.degrees(
                    np.arctan2(cloud_direction[1], cloud_direction[0])
                    - np.arctan2(cad_direction[1], cad_direction[0])
                )
            )
            angle = (angle + 180.0) % 360.0 - 180.0
            local_distances = nearest_distances[point_indices]
            measurements.append(
                {
                    'window_index': int(len(measurements)),
                    'segment_index': None,
                    'station_start': float(local_stations[0]),
                    'station_end': float(local_stations[-1]),
                    'length': local_span,
                    'angle_difference_deg': angle,
                    'cad_angle_deg': _angle_degrees(cad_direction),
                    'pointcloud_angle_deg': _angle_degrees(cloud_direction),
                    'rms': float(cloud_fit['rms']),
                    'cad_fit_rms': float(cad_fit['rms']),
                    'point_count': int(len(point_indices)),
                    'pointcloud_linearity': float(cloud_fit['linearity']),
                    'cad_linearity': float(cad_fit['linearity']),
                    'nearest_path_mean': float(np.mean(local_distances)),
                    'nearest_path_p95': float(np.quantile(local_distances, .95)),
                    'search_half_width': search_half_width,
                }
            )
            start_position = end_position

    if not measurements:
        result['reason'] = (
            'CAD 测量路径有界邻域内的点云未形成满足最少点数和跨度要求的局部角度窗口'
        )
        return result

    weights = np.asarray(
        [max(1e-9, item['point_count'] * item['length']) for item in measurements],
        dtype=float,
    )
    mean_angle = _weighted_circular_mean_degrees(
        [item['angle_difference_deg'] for item in measurements],
        weights,
    )
    cad_angle = _weighted_circular_mean_degrees(
        [item['cad_angle_deg'] for item in measurements],
        weights,
    )
    pointcloud_angle = _weighted_circular_mean_degrees(
        [item['pointcloud_angle_deg'] for item in measurements],
        weights,
    )
    result.update(
        status='OK',
        angle_difference_deg=mean_angle,
        cad_angle_deg=cad_angle,
        pointcloud_angle_deg=pointcloud_angle,
        rms=float(
            np.sqrt(
                np.average(
                    [item['rms'] ** 2 for item in measurements],
                    weights=weights,
                )
            )
        ),
        point_count=int(sum(item['point_count'] for item in measurements)),
        fitted_path_span=float(sum(item['length'] for item in measurements)),
        segments=measurements,
        reason=None,
    )
    return result


def polyline_angle(points, polyline):
    result = fit_polyline_angle(points, polyline)
    return (
        result['angle_difference_deg'],
        result['rms'],
        result['point_count'],
        result['segments'],
    )


MEASUREMENT_QUALITY_DEFAULT_THRESHOLDS = {
    'minimum_section_points': 100,
    'minimum_valid_stations': 8,
    'minimum_coverage_ratio': 0.80,
    'minimum_angle_fit_points': 8,
    'maximum_fit_mean_residual': 0.20,
    'maximum_fit_p95_residual': 0.35,
    'maximum_measurement_rms': 0.20,
    'maximum_measurement_p95': 0.35,
}


def evaluate_measurement_quality(
    fit_diagnostics,
    measurements,
    processing_quality=None,
    *,
    section_points=None,
    thresholds=None,
    registration_frame=None,
):
    """Combine stitching and CAD sampling evidence into a formal result gate."""
    limits = dict(MEASUREMENT_QUALITY_DEFAULT_THRESHOLDS)
    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                limits[key] = numeric_value

    failures = []
    insufficient = []
    coverage_diagnostic_warnings = []
    processing_status = None
    if isinstance(processing_quality, dict):
        processing_status = str(processing_quality.get('status', '')).upper() or None
        raw_coverage_warnings = processing_quality.get('coverage_diagnostic_warnings', [])
        if isinstance(raw_coverage_warnings, (list, tuple)):
            coverage_diagnostic_warnings = list(
                dict.fromkeys(str(reason) for reason in raw_coverage_warnings if reason)
            )
        if processing_status == 'FAIL':
            failures.extend(processing_quality.get('failure_reasons', []))
        elif processing_status == 'INSUFFICIENT_DATA':
            insufficient.extend(processing_quality.get('insufficient_data_reasons', []))
    else:
        insufficient.append('缺少点云拼接质量摘要，不能确认最终截面是否可正式测量')

    try:
        point_count = int(section_points) if section_points is not None else None
    except (TypeError, ValueError):
        point_count = None
    if point_count is None and isinstance(processing_quality, dict):
        try:
            point_count = int(processing_quality.get('section_points'))
        except (TypeError, ValueError):
            point_count = None
    if point_count is None:
        insufficient.append('缺少最终截面点数')
    elif point_count < int(limits['minimum_section_points']):
        insufficient.append(
            f'最终截面点数 {point_count} 小于 {int(limits["minimum_section_points"])}'
        )

    fit = fit_diagnostics if isinstance(fit_diagnostics, dict) else {}
    fit_mean = fit.get('selected_cad_to_pointcloud_mean', fit.get('global_cad_to_pointcloud_mean'))
    fit_p95 = fit.get('selected_cad_to_pointcloud_p95', fit.get('global_cad_to_pointcloud_p95'))
    try:
        fit_mean = float(fit_mean)
    except (TypeError, ValueError):
        fit_mean = None
    try:
        fit_p95 = float(fit_p95)
    except (TypeError, ValueError):
        fit_p95 = None
    if fit_mean is None or not np.isfinite(fit_mean) or fit_p95 is None or not np.isfinite(fit_p95):
        insufficient.append('缺少 CAD 配准残差')
    else:
        if fit_mean > limits['maximum_fit_mean_residual']:
            failures.append(
                f'CAD 配准平均残差 {fit_mean:.4f} 超过 {limits["maximum_fit_mean_residual"]:.4f}'
            )
        if fit_p95 > limits['maximum_fit_p95_residual']:
            failures.append(
                f'CAD 配准 P95 残差 {fit_p95:.4f} 超过 {limits["maximum_fit_p95_residual"]:.4f}'
            )
    measurement_results = []
    if isinstance(measurements, dict):
        measurement_items = measurements.items()
    else:
        measurement_items = []
    for measurement_id, measurement in measurement_items:
        if not isinstance(measurement, dict):
            continue
        measurement_failures = []
        measurement_insufficient = []
        try:
            valid_stations = int(measurement.get('valid_stations', 0))
        except (TypeError, ValueError):
            valid_stations = 0
        try:
            total_stations = int(measurement.get('total_stations', 0))
        except (TypeError, ValueError):
            total_stations = 0
        try:
            coverage_ratio = float(measurement.get('coverage_ratio', 0.0))
        except (TypeError, ValueError):
            coverage_ratio = 0.0
        try:
            rms = float(measurement.get('rms'))
        except (TypeError, ValueError):
            rms = None
        try:
            p95 = float(measurement.get('p95_absolute'))
        except (TypeError, ValueError):
            p95 = None
        try:
            angle_difference = float(measurement.get('signed_angle_difference_deg'))
        except (TypeError, ValueError):
            angle_difference = None
        try:
            angle_fit_points = int(measurement.get('angle_fit_point_count', 0))
        except (TypeError, ValueError):
            angle_fit_points = 0

        if total_stations <= 0 or valid_stations < int(limits['minimum_valid_stations']):
            measurement_insufficient.append(
                f'有效采样点不足（{valid_stations}/{total_stations}）'
            )
        if coverage_ratio < limits['minimum_coverage_ratio']:
            measurement_insufficient.append(
                f'测量线覆盖率 {coverage_ratio:.1%} 低于 {limits["minimum_coverage_ratio"]:.1%}'
            )
        if rms is None or not np.isfinite(rms) or p95 is None or not np.isfinite(p95):
            measurement_insufficient.append('缺少测量残差统计')
        else:
            if rms > limits['maximum_measurement_rms']:
                measurement_failures.append(
                    f'测量 RMS {rms:.4f} 超过 {limits["maximum_measurement_rms"]:.4f}'
                )
            if p95 > limits['maximum_measurement_p95']:
                measurement_failures.append(
                    f'测量 P95 {p95:.4f} 超过 {limits["maximum_measurement_p95"]:.4f}'
                )
        if (
            angle_difference is None
            or not np.isfinite(angle_difference)
            or angle_fit_points < int(limits['minimum_angle_fit_points'])
        ):
            measurement_insufficient.append(
                f'角度拟合有效点不足（{angle_fit_points}/'
                f'{int(limits["minimum_angle_fit_points"])}）'
            )

        if measurement_failures:
            measurement_status = 'FAIL'
            failures.extend(measurement_failures)
        elif measurement_insufficient:
            measurement_status = 'INSUFFICIENT_DATA'
            insufficient.extend(measurement_insufficient)
        else:
            measurement_status = 'PASS'
        measurement_results.append(
            {
                'measurement_id': str(measurement_id),
                'status': measurement_status,
                'formal_measurement_allowed': measurement_status == 'PASS',
                'valid_stations': valid_stations,
                'total_stations': total_stations,
                'coverage_ratio': coverage_ratio,
                'rms': rms,
                'p95_absolute': p95,
                'angle_difference_deg': angle_difference,
                'angle_fit_point_count': angle_fit_points,
                'failure_reasons': measurement_failures,
                'insufficient_data_reasons': measurement_insufficient,
            }
        )

    # Preserve order while avoiding repetitive UI messages when the same gate
    # explains several measurements.
    failures = list(dict.fromkeys(str(reason) for reason in failures if reason))
    insufficient = list(dict.fromkeys(str(reason) for reason in insufficient if reason))
    if failures:
        status = 'FAIL'
    elif insufficient:
        status = 'INSUFFICIENT_DATA'
    else:
        status = 'PASS'
    return {
        'status': status,
        'formal_measurement_allowed': status == 'PASS',
        'label': {
            'PASS': '正式测量通过',
            'FAIL': '正式测量不通过',
            'INSUFFICIENT_DATA': '数据不足，仅供参考',
        }[status],
        'fit': {
            'mean_residual': fit_mean,
            'p95_residual': fit_p95,
        },
        'section_points': point_count,
        'processing_status': processing_status,
        'measurements': measurement_results,
        'failure_reasons': failures,
        'insufficient_data_reasons': insufficient,
        'coverage_diagnostic_warnings': coverage_diagnostic_warnings,
        'thresholds': limits,
        'policy': '点云或测量证据不足时允许查看数值，但不输出正式合格结论。',
    }


def _analyze_legacy_axis_measurements(pointcloud_path,dxf_path,out,x_entity=9,y_entity=5,fit_start=10,fit_end=27,step=.05,x_entities=None,y_entities=None,fit_entities=None):
    if 'Microsoft YaHei' in {font.name for font in font_manager.fontManager.ttflist}: plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus']=False
    out=Path(out); out.mkdir(parents=True,exist_ok=True); cloud=o3d.io.read_point_cloud(str(pointcloud_path)); pts=np.asarray(cloud.points)
    if len(pts)<30:
        raise RuntimeError('点云点数太少')
    report_path=Path(pointcloud_path).parent.parent / 'processing_report.json'
    processing_report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else None
    section_contract = validate_section_input_contract(pointcloud_path, processing_report)
    profile=section_profile_from_points(pts, processing_report)
    profile_candidates = []
    if profile is None:
        c=np.median(pts[:,:2],0); _,directions=np.linalg.eigh(np.cov((pts[:,:2]-c).T))
        profile_candidates = [
            np.c_[(pts[:,:2]-c)@directions[:,0],pts[:,2]],
            np.c_[(pts[:,:2]-c)@directions[:,1],pts[:,2]],
        ]
    doc = ezdxf.readfile(str(dxf_path))
    es = list(doc.modelspace())
    indices = list(fit_entities) if fit_entities else list(range(fit_start, fit_end + 1))
    x_indices = list(x_entities) if x_entities else [x_entity]
    y_indices = list(y_entities) if y_entities else [y_entity]
    sampled_entities = [sample_entity(entity) for entity in es]
    cad_parts = [geometry for geometry in sampled_entities if len(geometry)]
    if not cad_parts:
        raise RuntimeError('DXF \u4e2d\u6ca1\u6709\u53ef\u7528\u4e8e\u914d\u51c6\u7684\u7ebf\u6216\u5706\u5f27')
    cad_outline = np.vstack(cad_parts)
    blue = _blue_fit_anchor(sampled_entities, indices)
    blue_anchor = blue
    if profile is None:
        candidates = []
        for profile_candidate in profile_candidates:
            dc_candidate, m_candidate, v_candidate, d_candidate, diagnostics_candidate = fit_profile_global(
                profile_candidate, cad_outline, blue_anchor
            )
            candidates.append((
                diagnostics_candidate['global_score'],
                profile_candidate,
                dc_candidate,
                m_candidate,
                v_candidate,
                d_candidate,
                diagnostics_candidate,
            ))
        _, profile, dc, m, v, _, fit_diagnostics = min(candidates, key=lambda item: item[0])
    else:
        dc, m, v, d, fit_diagnostics = fit_profile_global(profile, cad_outline, blue_anchor)
    profile_tree = cKDTree(_grid_reduce(profile, .025))
    blue_distances = profile_tree.query(transform(blue, dc, m, v[0], v[1:]))[0]
    fit_diagnostics['selected_cad_to_pointcloud_mean'] = float(np.mean(blue_distances))
    fit_diagnostics['selected_cad_to_pointcloud_p95'] = float(np.quantile(blue_distances, .95))
    fit_diagnostics['selected_anchor_entity_indices'] = [int(index) for index in indices]
    fit_diagnostics['selected_anchor_excludes_measurement_edges'] = True
    def geometry_for(entity_indices):
        parts = [sampled_entities[index] for index in entity_indices if 0 <= index < len(sampled_entities) and len(sampled_entities[index])]
        if not parts:
            raise RuntimeError('\u7ed8\u5236\u533a\u57df\u5185\u6ca1\u6709\u8bc6\u522b\u5230 CAD \u5b9e\u4f53')
        return transform(np.vstack(parts), dc, m, v[0], v[1:])
    ge={'x_axis_edge':geometry_for(x_indices),'y_axis_edge':geometry_for(y_indices)}
    x_segment=ge['x_axis_edge'][::-1]; y_segment=ge['y_axis_edge']; xa=(x_segment[-1]-x_segment[0])/np.linalg.norm(x_segment[-1]-x_segment[0]); ya=np.array([-xa[1],xa[0]])
    if ya@(y_segment[-1]-y_segment[0])<0: ya=-ya
    origin=y_segment[0]+ya*np.dot(x_segment[0]-y_segment[0],ya)
    grid=np.floor(profile/.025).astype(int); _,reduced_indices=np.unique(grid,axis=0,return_index=True); reduced=profile[reduced_indices]
    local=np.c_[(reduced-origin)@xa,(reduced-origin)@ya]; local_geometry={k:np.c_[(g-origin)@xa,(g-origin)@ya] for k,g in ge.items()}; results={}
    for key,label in [('x_axis_edge','绿色 X 轴边'),('y_axis_edge','绿色 Y 轴边')]:
        geometry=local_geometry[key]; rows,s=measure(local,geometry[0],geometry[-1],step); ang,rms,count=line_angle(local,geometry[0],geometry[-1]); cadang=0. if s['measurement_axis']=='Y' else 90.; s.update(label=label,cad_axis_angle_deg=cadang,pointcloud_line_angle_deg=cadang+ang,signed_angle_difference_deg=ang,absolute_angle_difference_deg=abs(ang),angle_fit_point_count=count,angle_fit_rms=float(rms)); results[key]=s; write_csv(out/f'{key}.csv',rows)
    angles=[{'edge':k,'label':r['label'],'signed_angle_difference_deg':r['signed_angle_difference_deg'],'absolute_angle_difference_deg':r['absolute_angle_difference_deg']} for k,r in results.items()]; write_csv(out/'green_edge_angles.csv',angles)
    fig,axs=plt.subplots(2,1,figsize=(13,8),dpi=150)
    for ax,(key,color) in zip(axs,[('x_axis_edge','#d13cff'),('y_axis_edge','#00a86b')]):
        rows=[r for r in csv.DictReader(open(out/f'{key}.csv',encoding='utf-8-sig')) if r['valid']=='True']; ss=[float(r['station']) for r in rows]; vv=[float(r['signed_axis_deviation']) for r in rows]; r=results[key]; ax.plot(ss,vv,color=color,marker='o',markersize=3,label='逐点偏差'); ax.axhline(r['mean_signed'],color='#ff7b72',ls='--',label=f'平均 {r["mean_signed"]:+.4f}'); ax.scatter(r['min_station'],r['min_signed'],c='#06c',label=f'最小 {r["min_signed"]:+.4f}'); ax.scatter(r['max_station'],r['max_signed'],c='#e63946',label=f'最大 {r["max_signed"]:+.4f}'); ax.set_title(r['label']+'：全长偏差曲线'); ax.set_xlabel('沿边距离'); ax.set_ylabel(r['measurement_axis']+'向偏差'); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); curve=out/'green_edge_deviation_curves.png'; fig.savefig(curve); plt.close(fig)
    all_cad_profile = {}
    display_indices = sorted(set(indices + x_indices + y_indices))
    display_profile = profile
    if len(display_profile) > 250000:
        random_generator = np.random.default_rng(20260824)
        display_profile = display_profile[random_generator.choice(len(display_profile), 250000, replace=False)]
    cad_alignment_image = save_cad_alignment_image(
        out / 'pointcloud_cad_alignment.png',
        display_profile,
        es,
        indices,
        x_indices,
        y_indices,
        m,
        v[0],
        float(np.mean(blue_distances)),
        fit_diagnostics,
        cad_transform=(dc, m, v[0], v[1:]),
    )
    height_colored_cad_overlay_image = save_height_colored_cad_overlay_image(
        out / 'height_colored_pointcloud_cad_overlay.png',
        profile,
        profile[:, 1],
        es,
        fit_indices=indices,
        x_indices=x_indices,
        y_indices=y_indices,
        cad_transform=(dc, m, v[0], v[1:]),
    )
    for index in display_indices:
        parts = [sample_entity(es[index])] if index < len(es) else []
        if parts and len(parts[0]):
            transformed_part = transform(parts[0], dc, m, v[0], v[1:])
            all_cad_profile[index] = transformed_part
    green_origin_profile = _green_local_to_profile([[0.0, 0.0]], origin, xa, ya)[0]
    x_fit = line_angle(local, local_geometry['x_axis_edge'][0], local_geometry['x_axis_edge'][-1])
    y_fit = line_angle(local, local_geometry['y_axis_edge'][0], local_geometry['y_axis_edge'][-1])
    report_figure=plt.figure(figsize=(18,12),dpi=170,facecolor='#11161d')
    header=report_figure.add_axes([0,0.91,1,0.09]); header.set_facecolor('#222a35'); header.set_xticks([]); header.set_yticks([])
    header.text(.025,.58,'Media \u56fa\u5b9a\u70b9\u4e91\u4e0e CAD \u5bf9\u6bd4\uff1a\u7eff\u8272\u8fb9\u504f\u5dee\u4e0e\u89d2\u5ea6\u5dee',color='white',fontsize=18,weight='bold',va='center')
    header.text(.79,.58,'\u70b9\u4e91\u4e0d\u53d8 \u00b7 CAD \u62df\u5408\u5230\u70b9\u4e91 \u00b7 \u84dd\u8272\u533a\u57df\u8f85\u52a9\u951a\u5b9a',color='#62d8ff',fontsize=12.5,va='center')
    limits=np.vstack([profile, *all_cad_profile.values()])
    minimum=limits.min(0); maximum=limits.max(0); padding=np.maximum((maximum-minimum)*.06,.12); x_limits=(minimum[0]-padding[0],maximum[0]+padding[0]); y_limits=(minimum[1]-padding[1],maximum[1]+padding[1])
    def style_view(view,title):
        view.set_facecolor('#f5f7fa'); view.set_aspect('equal',adjustable='box'); view.set_xlim(*x_limits); view.set_ylim(*y_limits); view.grid(alpha=.22); view.set_xlabel('截面平面内横向坐标'); view.set_ylabel('截面平面内竖向坐标'); view.set_title(title,fontsize=12,weight='bold')
    def draw_cad(view,labels=False,subdued=False):
        for index,geometry in all_cad_profile.items():
            if index in x_indices: color,width,label='#00a651',3.,'CAD X 轴（绿色边）'
            elif index in y_indices: color,width,label='#7acb00',3.,'CAD Y 轴（绿色边）'
            elif index in indices: color,width,label='#075cff',2.,'CAD 蓝色拟合轮廓' if index==indices[0] else None
            else: color,width,label='#9fa8b3',1.,None
            view.plot(geometry[:,0],geometry[:,1],color=color,lw=width,alpha=.58 if subdued else .95,label=label if labels else None)
    cloud_view=report_figure.add_axes([.045,.56,.31,.29]); style_view(cloud_view,'A. 点云截面（与上方清噪截面一致）'); cloud_view.scatter(profile[:,0],profile[:,1],s=1.0,c='#25384e',alpha=.78,lw=0,rasterized=True); cloud_view.scatter(*green_origin_profile,c='#005ce6',s=46,zorder=4,label='绿色基准原点')
    cad_view=report_figure.add_axes([.39,.56,.31,.29]); style_view(cad_view,'B. CAD 轮廓（同一截面坐标）'); draw_cad(cad_view,True); cad_view.scatter(*green_origin_profile,c='#005ce6',s=46,zorder=5,label='绿色基准原点'); cad_view.legend(loc='lower right',fontsize=7.2,framealpha=.92)
    overlay=report_figure.add_axes([.045,.10,.655,.35]); style_view(overlay,'C. 叠加测量视图：点云与 CAD 绿色边偏差'); overlay.scatter(profile[:,0],profile[:,1],s=.8,c='#5b6573',alpha=.3,lw=0,rasterized=True,label='3D 点云截面'); draw_cad(overlay,True)
    for key,rows in [('x_axis_edge',_read_csv(out/'x_axis_edge.csv')),('y_axis_edge',_read_csv(out/'y_axis_edge.csv'))]:
        line_color='#d13cff' if key=='x_axis_edge' else '#00a86b'; valid=[row for row in rows if row['valid']=='True']
        for row in valid[::max(1,len(valid)//18)]:
            cad_point = _green_local_to_profile([[float(row['cad_x']),float(row['cad_y'])]], origin, xa, ya)[0]
            pointcloud_point = _green_local_to_profile([[float(row['pointcloud_x']),float(row['pointcloud_y'])]], origin, xa, ya)[0]
            overlay.plot([cad_point[0],pointcloud_point[0]],[cad_point[1],pointcloud_point[1]],color=line_color,lw=1.,alpha=.9)
    overlay.scatter(*green_origin_profile,c='#005ce6',s=58,zorder=8,label='绿色基准原点'); overlay.scatter(*green_origin_profile,c='#ff2d55',marker='x',s=70,lw=2.2,zorder=9,label='测量原点'); overlay.legend(loc='lower right',fontsize=7.5,framealpha=.92)
    panel=report_figure.add_axes([.73,.10,.245,.75]); panel.set_facecolor('#222a35'); panel.set_xticks([]); panel.set_yticks([])
    panel.text(.06,.95,'\u6d4b\u91cf\u7ed3\u679c',color='white',fontsize=17,weight='bold',va='top'); panel.text(.06,.89,'\u9009\u4e2d CAD \u8f6e\u5ed3\u5bf9\u9f50\u8d28\u91cf',color='#62d8ff',fontsize=12,weight='bold'); panel.text(.08,.84,f'\u9009\u4e2d\u8f6e\u5ed3\u5e73\u5747\uff1a{fit_diagnostics["selected_cad_to_pointcloud_mean"]:.4f}',color='white',fontsize=10.8); panel.text(.08,.80,f'\u9009\u4e2d\u8f6e\u5ed3 P95\uff1a{fit_diagnostics["selected_cad_to_pointcloud_p95"]:.4f}',color='white',fontsize=10.8); panel.text(.08,.76,f'\u84dd\u8272\u8f6e\u5ed3 P95\uff1a{np.quantile(blue_distances,.95):.4f}',color='white',fontsize=10.3)
    gx=results['x_axis_edge']; gy=results['y_axis_edge']; panel.text(.06,.72,'绿色基准边偏移（轴向）',color='#7ee787',fontsize=12,weight='bold'); panel.text(.06,.67,'X 轴边（平行 X，计算 Y 偏移）',color='#d9a7ff',fontsize=11.2,weight='bold'); panel.text(.08,.63,f'平均 Y 偏移：{gx["mean_signed"]:+.4f}',color='white',fontsize=10.5); panel.text(.08,.59,f'绝对 / RMS：{gx["mean_absolute"]:.4f} / {gx["rms"]:.4f}',color='white',fontsize=10.5); panel.text(.08,.55,f'覆盖率：{gx["coverage_ratio"]*100:.1f}%',color='white',fontsize=10.5); panel.text(.08,.51,f'最小 / 最大 Y：{gx["min_signed"]:+.4f} / {gx["max_signed"]:+.4f}',color='white',fontsize=10.2); panel.text(.06,.45,f'角度差（点云-CAD）：{gx["signed_angle_difference_deg"]:+.4f}°',color='#d9a7ff',fontsize=10.2,weight='bold')
    panel.text(.06,.37,'Y 轴边（平行 Y，计算 X 偏移）',color='#80e5a2',fontsize=11.2,weight='bold'); panel.text(.08,.33,f'平均 X 偏移：{gy["mean_signed"]:+.4f}',color='white',fontsize=10.5); panel.text(.08,.29,f'绝对 / RMS：{gy["mean_absolute"]:.4f} / {gy["rms"]:.4f}',color='white',fontsize=10.5); panel.text(.08,.25,f'覆盖率：{gy["coverage_ratio"]*100:.1f}%',color='white',fontsize=10.5); panel.text(.08,.21,f'最小 / 最大 X：{gy["min_signed"]:+.4f} / {gy["max_signed"]:+.4f}',color='white',fontsize=10.2); panel.text(.06,.15,f'角度差（点云-CAD）：{gy["signed_angle_difference_deg"]:+.4f}°',color='#80e5a2',fontsize=10.2,weight='bold'); panel.text(.06,.08,'正值：点云位于 CAD 的 +X / +Y 方向',color='#aab4c2',fontsize=9.5); panel.text(.06,.04,'单位：沿用 DXF / 点云坐标单位',color='#aab4c2',fontsize=9.5)
    report_figure.tight_layout(rect=(0,0,1,.91)); report_image=out/'media_measurement_report.png'; report_figure.savefig(report_image,facecolor=report_figure.get_facecolor(),bbox_inches='tight'); plt.close(report_figure)
    measurement_quality = evaluate_measurement_quality(
        fit_diagnostics,
        results,
        processing_report.get('quality') if isinstance(processing_report, dict) else None,
        section_points=len(pts),
    )
    report = {
        'source_pointcloud': str(pointcloud_path),
        'source_dxf': str(dxf_path),
        'fit': {
            'entities': indices,
            'mirror_x': m,
            'angle_deg': float(np.degrees(v[0])),
            'mean_residual': float(blue_distances.mean()),
            'p95_residual': float(np.quantile(blue_distances, .95)),
            'cad_to_pointcloud_angle_deg': float(np.degrees(v[0])),
            'mirror_applied': bool(m < 0),
            'pointcloud_transformed': False,
            'pointcloud_transform_applied': False,
            'pointcloud_coordinate_system': section_contract.get('coordinate_system') if section_contract else None,
            'pointcloud_source_sha256': section_contract.get('analysis_section_sha256') if section_contract else None,
            'cad_fitted_to_pointcloud': True,
            'method': fit_diagnostics['method'],
            'global_alignment': fit_diagnostics,
            'cad_transform': {
                'center': np.asarray(dc, dtype=float).tolist(),
                'mirror': float(m),
                'angle_rad': float(v[0]),
                'translation': np.asarray(v[1:], dtype=float).tolist(),
            },
        },
        'green_measurements': results,
        'measurement_quality': measurement_quality,
        'processing_quality': processing_report.get('quality') if isinstance(processing_report, dict) else None,
        'outputs': {
            'curve_image': str(curve),
            'report_image': str(report_image),
            'x_csv': str(out / 'x_axis_edge.csv'),
            'y_csv': str(out / 'y_axis_edge.csv'),
            'angle_csv': str(out / 'green_edge_angles.csv'),
            'cad_alignment_image': str(cad_alignment_image),
            'height_colored_cad_overlay_image': str(height_colored_cad_overlay_image),
            'height_colored_cad_overlay_version': 11,
        },
        'units': '\u6cbf\u7528\u70b9\u4e91\u4e0e DXF \u5750\u6807\u5355\u4f4d',
    }
    report['section_contract'] = section_contract
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def analyze(
    pointcloud_path,
    dxf_path,
    out,
    x_entity=9,
    y_entity=5,
    fit_start=10,
    fit_end=27,
    step=.05,
    x_entities=None,
    y_entities=None,
    fit_entities=None,
    registration_entities=None,
    measurement_entities=None,
    registration_paths=None,
    measurement_paths=None,
):
    """Fit CAD to a fixed point-cloud section and measure selected CAD line segments.

    The legacy X/Y/blue-contour call shape remains available for the old web entry.
    The Streamlit workflow uses one registration path plus one or more measurement
    paths, so every measurement is evaluated along its continuous CAD path.
    """
    if (
        registration_entities is None
        and measurement_entities is None
        and registration_paths is None
        and measurement_paths is None
    ):
        return _analyze_legacy_axis_measurements(
            pointcloud_path,
            dxf_path,
            out,
            x_entity=x_entity,
            y_entity=y_entity,
            fit_start=fit_start,
            fit_end=fit_end,
            step=step,
            x_entities=x_entities,
            y_entities=y_entities,
            fit_entities=fit_entities,
        )

    if 'Microsoft YaHei' in {font.name for font in font_manager.fontManager.ttflist}:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    output_directory = Path(out)
    output_directory.mkdir(parents=True, exist_ok=True)
    cloud = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(cloud.points)
    if len(points) < 30:
        raise RuntimeError('点云点数太少')

    processing_report_path = Path(pointcloud_path).parent.parent / 'processing_report.json'
    processing_report = (
        json.loads(processing_report_path.read_text(encoding='utf-8'))
        if processing_report_path.exists()
        else None
    )
    section_contract = validate_section_input_contract(pointcloud_path, processing_report)
    profile = section_profile_from_points(points, processing_report)
    profile_candidates = []
    if profile is None:
        center = np.median(points[:, :2], axis=0)
        _, directions = np.linalg.eigh(np.cov((points[:, :2] - center).T))
        profile_candidates = [
            np.c_[(points[:, :2] - center) @ directions[:, 0], points[:, 2]],
            np.c_[(points[:, :2] - center) @ directions[:, 1], points[:, 2]],
        ]

    document = ezdxf.readfile(str(dxf_path))
    entities = list(document.modelspace())
    if registration_paths is None:
        registration_indices = _unique_entity_indices(registration_entities)
        if len(registration_indices) != 1:
            raise RuntimeError('CAD 配准必须选择 1 条基准线，或提供 1 条连续配准路径')
        registration_paths = [registration_indices]
    else:
        registration_paths = _normalize_entity_path_collection(registration_paths)
        registration_indices = [index for path in registration_paths for index in path]
    if measurement_paths is None:
        measurement_indices = _unique_entity_indices(measurement_entities)
        measurement_paths = [[index] for index in measurement_indices]
    else:
        measurement_paths = _normalize_entity_path_collection(measurement_paths)
        measurement_indices = [index for path in measurement_paths for index in path]
    if len(registration_paths) != 1 or not registration_indices:
        raise RuntimeError('CAD 配准必须且只能选择 1 条连续基准路径')
    if not measurement_indices or not measurement_paths:
        raise RuntimeError('请至少选择 1 条偏差与角度测量路径')
    unknown_indices = sorted(
        set(registration_indices + measurement_indices) - set(range(len(entities)))
    )
    if unknown_indices:
        raise RuntimeError('选择的 CAD 实体不存在：' + ', '.join(map(str, unknown_indices)))
    overlap = sorted(set(registration_indices) & set(measurement_indices))
    if overlap:
        raise RuntimeError('同一条 CAD 线段不能同时作为配准基准线和测量线：' + ', '.join(map(str, overlap)))

    sampled_entities = [sample_entity(entity) for entity in entities]
    cad_parts = [geometry for geometry in sampled_entities if len(geometry)]
    if not cad_parts:
        raise RuntimeError('DXF 中没有可用于配准的线或圆弧')
    for index in registration_indices + measurement_indices:
        if not len(sampled_entities[index]):
            raise RuntimeError(f'选择的 CAD 实体没有可用几何：实体 {index}')
    cad_outline = np.vstack(cad_parts)
    registration_anchor = _stack_sampled_path(sampled_entities, registration_indices)
    if registration_anchor is None:
        raise RuntimeError('选中的 CAD 配准路径没有可用几何')

    if profile is None:
        fit_candidates = []
        for profile_candidate in profile_candidates:
            fit_center, mirror, parameters, _, diagnostics = fit_profile_global(
                profile_candidate,
                cad_outline,
                registration_anchor,
            )
            fit_candidates.append(
                (
                    diagnostics['global_score'],
                    profile_candidate,
                    fit_center,
                    mirror,
                    parameters,
                    diagnostics,
                )
            )
        _, profile, fit_center, mirror, parameters, fit_diagnostics = min(
            fit_candidates,
            key=lambda item: item[0],
        )
    else:
        fit_center, mirror, parameters, _, fit_diagnostics = fit_profile_global(
            profile,
            cad_outline,
            registration_anchor,
        )

    reduced_profile = _grid_reduce(profile, .025)
    profile_tree = cKDTree(reduced_profile)
    registration_distances = profile_tree.query(
        transform(registration_anchor, fit_center, mirror, parameters[0], parameters[1:])
    )[0]
    fit_diagnostics['selected_cad_to_pointcloud_mean'] = float(np.mean(registration_distances))
    fit_diagnostics['selected_cad_to_pointcloud_p95'] = float(np.quantile(registration_distances, .95))
    fit_diagnostics['registration_entity_index'] = int(registration_indices[0])
    fit_diagnostics['selected_anchor_excludes_measurement_lines'] = True
    registration_frame = cad_registration_frame(
        (fit_center, mirror, parameters[0], parameters[1:]),
        registration_anchor,
    )
    fit_diagnostics['registration_frame'] = registration_frame

    measurement_colors = ['#00a651', '#16a34a', '#0f9d58', '#22c55e', '#65a30d', '#10b981']
    measurements = {}
    measurement_csvs = []
    reduced_profile_section = _grid_reduce(profile, .025)
    reduced_profile_cad = inverse_transform(
        reduced_profile_section,
        fit_center,
        mirror,
        parameters[0],
        parameters[1:],
    )
    for position, path_indices in enumerate(measurement_paths, start=1):
        path_geometry = _stack_sampled_path(sampled_entities, path_indices)
        if path_geometry is None:
            raise RuntimeError(f'选择的 CAD 测量路径 {position} 没有可用几何')
        # Measurements use the CAD original XY axes. The frozen point cloud
        # remains untouched; only this derived analysis view is inverse-mapped.
        geometry = path_geometry
        rows, statistics = measure_polyline(
            reduced_profile_cad,
            geometry,
            step,
            allow_empty=True,
        )
        angle_fit = fit_polyline_angle(reduced_profile_cad, geometry)
        angle_difference = angle_fit['angle_difference_deg']
        angle_rms = angle_fit['rms']
        angle_point_count = angle_fit['point_count']
        if len(path_indices) == 1:
            measurement_key = f'measurement_entity_{path_indices[0]}'
            label = f'测量线 {position}（实体 {path_indices[0]}）'
        else:
            measurement_key = f'measurement_path_{position}'
            label = (
                f'测量路径 {position}（实体 {path_indices[0]}→{path_indices[-1]}，'
                f'共 {len(path_indices)} 段）'
            )
        csv_path = output_directory / f'{measurement_key}.csv'
        start_angle = float(np.degrees(np.arctan2(
            geometry[1, 1] - geometry[0, 1],
            geometry[1, 0] - geometry[0, 0],
        )))
        end_angle = float(np.degrees(np.arctan2(
            geometry[-1, 1] - geometry[-2, 1],
            geometry[-1, 0] - geometry[-2, 0],
        )))
        statistics.update(
            measurement_id=measurement_key,
            entity_index=int(path_indices[0]) if len(path_indices) == 1 else None,
            entity_indices=[int(index) for index in path_indices],
            path_index=int(position),
            label=label,
            color=measurement_colors[(position - 1) % len(measurement_colors)],
            cad_line_angle_deg=start_angle,
            cad_path_start_angle_deg=start_angle,
            cad_path_end_angle_deg=end_angle,
            cad_path_angle_deg=angle_fit.get('cad_angle_deg'),
            pointcloud_path_angle_deg=angle_fit.get('pointcloud_angle_deg'),
            pointcloud_line_angle_deg=(
                angle_fit.get('pointcloud_angle_deg')
                if angle_difference is not None
                else None
            ),
            signed_angle_difference_deg=(
                float(angle_difference) if angle_difference is not None else None
            ),
            absolute_angle_difference_deg=(
                float(abs(angle_difference)) if angle_difference is not None else None
            ),
            angle_fit_point_count=int(angle_point_count),
            angle_fit_rms=float(angle_rms) if angle_rms is not None else None,
            angle_fit_status=angle_fit['status'],
            angle_fit_reason=angle_fit['reason'],
            angle_fit_method=angle_fit['method'],
            angle_fit_candidate_point_count=angle_fit['candidate_point_count'],
            angle_fit_path_span=angle_fit['fitted_path_span'],
            angle_fit_search_half_width=angle_fit['search_half_width'],
            angle_fit_minimum_points=angle_fit['minimum_points'],
            angle_fit_minimum_span=angle_fit['minimum_span'],
            angle_fit_maximum_window_span=angle_fit['maximum_window_span'],
            angle_segments=angle_fit['segments'],
            csv_path=str(csv_path),
            coordinate_system='cad_original_xy',
        )
        measurements[measurement_key] = statistics
        write_csv(csv_path, rows)
        measurement_csvs.append(
            {
                'entity_index': int(path_indices[0]) if len(path_indices) == 1 else None,
                'entity_indices': [int(index) for index in path_indices],
                'path_index': int(position),
                'label': label,
                'path': str(csv_path),
            }
        )

    measurement_quality = evaluate_measurement_quality(
        fit_diagnostics,
        measurements,
        processing_report.get('quality') if isinstance(processing_report, dict) else None,
        section_points=len(points),
        registration_frame=registration_frame,
    )
    angle_rows = [
        {
            'measurement_id': key,
            'entity_index': statistics['entity_index'],
            'entity_indices': statistics.get('entity_indices', []),
            'path_index': statistics.get('path_index'),
            'label': statistics['label'],
            'signed_angle_difference_deg': statistics['signed_angle_difference_deg'],
            'absolute_angle_difference_deg': statistics['absolute_angle_difference_deg'],
            'cad_path_angle_deg': statistics.get('cad_path_angle_deg'),
            'pointcloud_path_angle_deg': statistics.get('pointcloud_path_angle_deg'),
            'mean_signed_x': statistics.get('mean_signed_x'),
            'mean_signed_y': statistics.get('mean_signed_y'),
            'rms_x': statistics.get('rms_x'),
            'rms_y': statistics.get('rms_y'),
            'p95_absolute_x': statistics.get('p95_absolute_x'),
            'p95_absolute_y': statistics.get('p95_absolute_y'),
        }
        for key, statistics in measurements.items()
    ]
    angle_csv_path = output_directory / 'measurement_line_angles.csv'
    write_csv(angle_csv_path, angle_rows)

    curve_figure, curve_axes = plt.subplots(
        len(measurements),
        1,
        figsize=(13, max(4.0, 3.6 * len(measurements))),
        dpi=150,
    )
    if len(measurements) == 1:
        curve_axes = [curve_axes]
    for axis, (_, statistics) in zip(curve_axes, measurements.items()):
        rows = _read_csv(statistics['csv_path'])
        stations = [float(row['station']) for row in rows]
        deviations = [
            float(row['signed_normal_deviation'])
            if row['valid'] == 'True' and row['signed_normal_deviation']
            else np.nan
            for row in rows
        ]
        if np.isfinite(deviations).any():
            axis.plot(
                stations,
                deviations,
                color=statistics['color'],
                marker='o',
                markersize=3,
                label='逐点法向偏差',
            )
            axis.axhline(
                statistics['mean_signed'],
                color='#ff7b72',
                linestyle='--',
                label=f'平均 {statistics["mean_signed"]:+.4f}',
            )
            axis.scatter(
                statistics['min_station'],
                statistics['min_signed'],
                c='#0066cc',
                label=f'最小 {statistics["min_signed"]:+.4f}',
            )
            axis.scatter(
                statistics['max_station'],
                statistics['max_signed'],
                c='#e63946',
                label=f'最大 {statistics["max_signed"]:+.4f}',
            )
            axis.legend()
        else:
            axis.text(
                .5,
                .5,
                '测量路径有界邻域内没有有效点云',
                transform=axis.transAxes,
                ha='center',
                va='center',
                color='#6b7280',
            )
        axis.set_title(statistics['label'] + '：全长法向偏差曲线')
        axis.set_xlabel('沿 CAD 路径距离')
        axis.set_ylabel('左法向偏差')
        axis.grid(alpha=.3)
    curve_figure.tight_layout()
    curve_image = output_directory / 'measurement_line_deviation_curves.png'
    curve_figure.savefig(curve_image)
    plt.close(curve_figure)

    display_profile = inverse_transform(
        profile,
        fit_center,
        mirror,
        parameters[0],
        parameters[1:],
    )
    if len(display_profile) > 250000:
        random_generator = np.random.default_rng(20260824)
        display_profile = display_profile[
            random_generator.choice(len(display_profile), 250000, replace=False)
        ]
    cad_alignment_image = save_cad_alignment_image(
        output_directory / 'pointcloud_cad_alignment.png',
        profile,
        entities,
        registration_indices=registration_indices,
        measurement_indices=measurement_indices,
        mirror=mirror,
        angle=parameters[0],
        residual=float(np.mean(registration_distances)),
        alignment=fit_diagnostics,
        cad_transform=(fit_center, mirror, parameters[0], parameters[1:]),
    )
    height_colored_cad_overlay_image = save_height_colored_cad_overlay_image(
        output_directory / 'height_colored_pointcloud_cad_overlay.png',
        profile,
        profile[:, 1],
        entities,
        registration_indices=registration_indices,
        measurement_indices=measurement_indices,
        cad_transform=(fit_center, mirror, parameters[0], parameters[1:]),
    )

    display_indices = sorted(set(registration_indices + measurement_indices))
    all_cad_profile = {
        index: sampled_entities[index]
        for index in display_indices
        if len(sampled_entities[index])
    }
    limits = np.vstack([display_profile, *all_cad_profile.values()])
    minimum = limits.min(axis=0)
    maximum = limits.max(axis=0)
    padding = np.maximum((maximum - minimum) * .06, .12)
    x_limits = (minimum[0] - padding[0], maximum[0] + padding[0])
    y_limits = (minimum[1] - padding[1], maximum[1] + padding[1])
    measurement_color_by_index = {}
    for statistics in measurements.values():
        for entity_index in statistics.get('entity_indices', []):
            measurement_color_by_index[int(entity_index)] = statistics['color']

    report_figure = plt.figure(figsize=(18, 12), dpi=170, facecolor='#11161d')
    header = report_figure.add_axes([0, .91, 1, .09])
    header.set_facecolor('#222a35')
    header.set_xticks([])
    header.set_yticks([])
    header.text(
        .025,
        .58,
        '固定点云与 CAD 对比：配准基准线、测量线偏差与角度差',
        color='white',
        fontsize=18,
        weight='bold',
        va='center',
    )
    header.text(
        .70,
        .58,
        '点云不变 · CAD 拟合到点云 · 连续路径配准 / 连续路径测量',
        color='#62d8ff',
        fontsize=12.5,
        va='center',
    )

    def style_view(axis, title):
        axis.set_facecolor('#f5f7fa')
        axis.set_aspect('equal', adjustable='box')
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.grid(alpha=.22)
        axis.set_xlabel('截面平面内横向坐标')
        axis.set_ylabel('截面平面内竖向坐标')
        axis.set_title(title, fontsize=12, weight='bold')

    def draw_cad(axis, labels=False, subdued=False):
        labels_used = set()
        for index, geometry in all_cad_profile.items():
            if index in registration_indices:
                color, width, label = '#075cff', 3.0, 'CAD 配准基准线'
            elif index in measurement_color_by_index:
                color, width, label = (
                    measurement_color_by_index[index],
                    2.6,
                    'CAD 测量线',
                )
            else:
                color, width, label = '#9fa8b3', 1.0, None
            axis.plot(
                geometry[:, 0],
                geometry[:, 1],
                color=color,
                linewidth=width,
                alpha=.58 if subdued else .95,
                label=(
                    label
                    if labels and label is not None and label not in labels_used
                    else None
                ),
            )
            if label is not None:
                labels_used.add(label)

    cloud_view = report_figure.add_axes([.045, .56, .31, .29])
    style_view(cloud_view, 'A. 点云截面（与上方清噪截面一致）')
    cloud_view.scatter(
        display_profile[:, 0],
        display_profile[:, 1],
        s=1.0,
        c='#25384e',
        alpha=.78,
        linewidths=0,
        rasterized=True,
    )
    cad_view = report_figure.add_axes([.39, .56, .31, .29])
    style_view(cad_view, 'B. CAD 选中线段（同一截面坐标）')
    draw_cad(cad_view, labels=True)
    cad_view.legend(loc='lower right', fontsize=7.2, framealpha=.92)
    overlay = report_figure.add_axes([.045, .10, .655, .35])
    style_view(overlay, 'C. 叠加测量视图：点云与 CAD 测量线法向偏差')
    overlay.scatter(
        display_profile[:, 0],
        display_profile[:, 1],
        s=.8,
        c='#5b6573',
        alpha=.3,
        linewidths=0,
        rasterized=True,
        label='3D 点云截面',
    )
    draw_cad(overlay, labels=True)
    for _, statistics in measurements.items():
        valid_rows = [
            row
            for row in _read_csv(statistics['csv_path'])
            if row['valid'] == 'True'
        ]
        for row in valid_rows[::max(1, len(valid_rows) // 18)]:
            overlay.plot(
                [float(row['cad_x']), float(row['pointcloud_x'])],
                [float(row['cad_y']), float(row['pointcloud_y'])],
                color=statistics['color'],
                linewidth=1.0,
                alpha=.9,
            )
    overlay.legend(loc='lower right', fontsize=7.5, framealpha=.92)

    result_panel = report_figure.add_axes([.73, .10, .245, .75])
    result_panel.set_facecolor('#222a35')
    result_panel.set_xticks([])
    result_panel.set_yticks([])
    result_panel.text(.06, .95, '测量结果', color='white', fontsize=17, weight='bold', va='top')
    result_panel.text(.06, .89, 'CAD 配准基准线对齐质量', color='#62d8ff', fontsize=12, weight='bold')
    result_panel.text(
        .08,
        .84,
        f'平均残差：{fit_diagnostics["selected_cad_to_pointcloud_mean"]:.4f}',
        color='white',
        fontsize=10.8,
    )
    result_panel.text(
        .08,
        .80,
        f'P95 残差：{fit_diagnostics["selected_cad_to_pointcloud_p95"]:.4f}',
        color='white',
        fontsize=10.8,
    )
    quality_status = measurement_quality['status']
    quality_label = measurement_quality['label']
    quality_color = {'PASS': '#7ee787', 'FAIL': '#ff8f8f', 'INSUFFICIENT_DATA': '#ffd166'}[quality_status]
    result_panel.text(.06, .74, f'正式测量结论：{quality_status}', color=quality_color, fontsize=12, weight='bold')
    result_panel.text(.08, .70, quality_label, color='white', fontsize=9.8)
    result_panel.text(.06, .64, '测量线：法向偏差 / 角度差', color='#7ee787', fontsize=12, weight='bold')

    def optional_number(value, format_spec, suffix=''):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return '不可用'
        return f'{format(number, format_spec)}{suffix}' if np.isfinite(number) else '不可用'

    panel_y = .59
    for position, statistics in enumerate(measurements.values(), start=1):
        if panel_y < .13:
            result_panel.text(
                .06,
                panel_y,
                f'另有 {len(measurements) - position + 1} 条测量线，详见 CSV。',
                color='#aab4c2',
                fontsize=9.5,
            )
            break
        result_panel.text(
            .06,
            panel_y,
            statistics['label'],
            color=statistics['color'],
            fontsize=10.5,
            weight='bold',
        )
        result_panel.text(
            .08,
            panel_y - .04,
            '平均 / RMS：'
            f'{optional_number(statistics["mean_signed"], "+.4f")} / '
            f'{optional_number(statistics["rms"], ".4f")}',
            color='white',
            fontsize=9.5,
        )
        result_panel.text(
            .08,
            panel_y - .08,
            f'覆盖：{statistics["coverage_ratio"] * 100:.1f}%  角度差：'
            f'{optional_number(statistics["signed_angle_difference_deg"], "+.4f", "°")}',
            color='white',
            fontsize=9.5,
        )
        panel_y -= .13
    result_panel.text(.06, .04, '正值：点云位于 CAD 线段左法线方向', color='#aab4c2', fontsize=9.5)
    report_image = output_directory / 'media_measurement_report.png'
    report_figure.savefig(report_image, facecolor=report_figure.get_facecolor(), bbox_inches='tight')
    plt.close(report_figure)

    report = {
        'source_pointcloud': str(pointcloud_path),
        'source_dxf': str(dxf_path),
        'fit': {
            'entities': registration_indices,
            'registration_entity_index': int(registration_indices[0]),
            'registration_paths': [
                [int(index) for index in path]
                for path in registration_paths
            ],
            'mirror_x': mirror,
            'angle_deg': float(np.degrees(parameters[0])),
            'mean_residual': float(registration_distances.mean()),
            'p95_residual': float(np.quantile(registration_distances, .95)),
            'cad_to_pointcloud_angle_deg': float(np.degrees(parameters[0])),
            'mirror_applied': bool(mirror < 0),
            'pointcloud_transformed': False,
            'pointcloud_transform_applied': False,
            'pointcloud_coordinate_system': section_contract.get('coordinate_system') if section_contract else None,
            'pointcloud_source_sha256': section_contract.get('analysis_section_sha256') if section_contract else None,
            'cad_fitted_to_pointcloud': True,
            'method': fit_diagnostics['method'],
            'global_alignment': fit_diagnostics,
            'cad_transform': {
                'center': np.asarray(fit_center, dtype=float).tolist(),
                'mirror': float(mirror),
                'angle_rad': float(parameters[0]),
                'translation': np.asarray(parameters[1:], dtype=float).tolist(),
            },
            'coordinate_frame': registration_frame,
        },
        'measurements': measurements,
        'measurement_quality': measurement_quality,
        'processing_quality': processing_report.get('quality') if isinstance(processing_report, dict) else None,
        'measurement_paths': [
            [int(index) for index in path]
            for path in measurement_paths
        ],
        'outputs': {
            'curve_image': str(curve_image),
            'report_image': str(report_image),
            'measurement_csvs': measurement_csvs,
            'measurement_angle_csv': str(angle_csv_path),
            'cad_alignment_image': str(cad_alignment_image),
            'height_colored_cad_overlay_image': str(height_colored_cad_overlay_image),
            'height_colored_cad_overlay_version': 11,
        },
        'units': '沿用点云与 DXF 坐标单位',
        'measurement_sign_convention': '正值表示点云位于 CAD 线段起点→终点的左法线方向',
        'measurement_coordinate_system': 'cad_original_xy',
        'measurement_xy_sign_convention': 'signed_x/y = pointcloud_cad_original_xy - cad_original_xy',
    }
    report['section_contract'] = section_contract
    (output_directory / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    return report


