"""Poisson 点云三维重建入口。

整体流程：
1. 读取点云文件，例如 .pcd/.ply/.xyz/.npy。
2. 可选体素下采样，减少点数并让点云密度更均匀。
3. 可选噪点处理，删除明显离群的孤立点。
4. 可选局部补点，在相近点之间插入中点，让点云稍微更连续。
5. 估计并统一法线方向。Poisson 重建强依赖法线方向。
6. 用 Open3D 的 Poisson Surface Reconstruction 生成三角网格。
7. 清理网格并保存 OBJ/PLY/JSON。

注意：
Poisson 适合较连续的表面。对于教室、Livox 大场景，它通常比单物体
VRCNet 更稳，但仍可能产生漂浮面、封闭假面或过度平滑。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from reconstruction_common import (
    add_common_io_args,
    apply_config_defaults,
    estimate_normals,
    fill_common_defaults,
    o3d,
    print_progress,
    read_point_cloud,
    resolve_output_paths,
    write_json,
    write_meshes,
)

DEFAULT_CONFIG_KEY = "poisson"


@dataclass
class PoissonRunInfo:
    """保存本次运行参数和结果统计，最后写入 JSON，方便复现实验。"""

    method: str
    input: str
    output_meshes: list[str]
    voxel_size: float
    point_count: int
    vertex_count: int
    triangle_count: int
    depth: int
    scale: float
    linear_fit: bool
    normals_radius: float
    normals_max_nn: int
    orient_max_nn: int
    density_quantile: float
    statistical_outlier_nb_neighbors: int
    statistical_outlier_std_ratio: float
    radius_outlier_nb_points: int
    radius_outlier_radius: float
    completion_rounds: int
    completion_max_distance: float
    completion_added_points: int


def remove_noise(
    cloud: "o3d.geometry.PointCloud",
    statistical_nb_neighbors: int,
    statistical_std_ratio: float,
    radius_nb_points: int,
    radius: float,
) -> "o3d.geometry.PointCloud":
    """删除点云噪点。

    这里提供两类 Open3D 内置去噪：

    1. statistical_outlier：
       统计每个点到邻居的平均距离。如果某个点离邻居明显更远，就认为它是离群点。
       - nb_neighbors 越大，判断越稳定，但也可能抹掉小结构。
       - std_ratio 越小，删除越严格；越大，删除越宽松。

    2. radius_outlier：
       检查半径 radius 内是否至少有 nb_points 个邻居。
       半径内邻居太少的点会被认为是孤立点。
       这个方法对尺度很敏感，半径设置不合适会删掉稀疏但真实的结构。
    """
    if statistical_nb_neighbors > 0:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=statistical_nb_neighbors,
            std_ratio=statistical_std_ratio,
        )
    if radius_nb_points > 0 and radius > 0:
        cloud, _ = cloud.remove_radius_outlier(
            nb_points=radius_nb_points,
            radius=radius,
        )
    return cloud


def complete_point_cloud(
    cloud: "o3d.geometry.PointCloud",
    rounds: int,
    max_distance: float,
) -> tuple["o3d.geometry.PointCloud", int]:
    """轻量点云补全：在相近点之间插入中点。

    这不是深度学习补全，也不会凭空猜出被遮挡物体。
    它只做一件很保守的事：

        点 A 和最近邻点 B 足够近 -> 插入中点 M = (A + B) / 2

    作用：
    - 让局部点云更密，法线估计更稳定。
    - 对小裂缝、小空隙有一点缓解。

    风险：
    - completion_rounds 太大时点数会快速膨胀。
    - max_distance 太大时，可能跨越真实空洞乱补点。
    """
    if rounds <= 0 or len(cloud.points) < 2:
        return cloud, 0

    total_added = 0
    for _ in range(rounds):
        points = np.asarray(cloud.points)

        # KDTree 用来快速找每个点的最近邻。k=2 是因为第 1 个邻居是点自己，
        # 第 2 个邻居才是真正的最近点。
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=2)
        neighbor_distances = distances[:, 1]
        neighbor_indices = indices[:, 1]

        if max_distance > 0:
            distance_limit = max_distance
        else:
            # 自动阈值：取最近邻距离的 95% 分位数再放大 1.5 倍。
            # 这样通常只在局部连续区域补点，不会跨很远的空洞。
            positive_distances = neighbor_distances[neighbor_distances > 1e-12]
            if len(positive_distances) == 0:
                break
            distance_limit = float(np.quantile(positive_distances, 0.95) * 1.5)

        mask = (neighbor_distances > 1e-12) & (neighbor_distances <= distance_limit)
        if not np.any(mask):
            break

        # 中点插值补点。如果点云带颜色，则颜色也按两端点平均。
        midpoints = (points[mask] + points[neighbor_indices[mask]]) * 0.5
        merged_points = np.vstack([points, midpoints])

        completed = o3d.geometry.PointCloud()
        completed.points = o3d.utility.Vector3dVector(merged_points)
        if cloud.has_colors():
            colors = np.asarray(cloud.colors)
            midpoint_colors = (colors[mask] + colors[neighbor_indices[mask]]) * 0.5
            completed.colors = o3d.utility.Vector3dVector(np.vstack([colors, midpoint_colors]))

        cloud = completed
        total_added += len(midpoints)

    return cloud, total_added


def reconstruct_poisson(
    cloud: "o3d.geometry.PointCloud",
    depth: int,
    scale: float,
    linear_fit: bool,
    density_quantile: float,
) -> "o3d.geometry.TriangleMesh":
    """执行 Poisson 曲面重建并做基础网格清理。

    Poisson 的核心思想：
    - 输入是带方向的点云法线场。
    - 算法估计一个隐式实体函数。
    - 再从这个隐式函数中提取等值面，得到三角网格。

    关键参数：
    - depth：八叉树深度。越大越细，速度/内存开销越高。
    - scale：重建包围盒缩放。太大可能产生外壳，太小可能裁掉边缘。
    - linear_fit：Open3D Poisson 的线性拟合选项。
    - density_quantile：删除低密度顶点，减少漂浮面。
    """
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=depth,
        scale=scale,
        linear_fit=linear_fit,
    )

    if density_quantile > 0:
        density_values = np.asarray(densities)
        threshold = float(np.quantile(density_values, density_quantile))

        # densities 是 Poisson 返回的“局部支持密度”。低密度点通常位于
        # 扫描稀疏区域或算法猜出来的漂浮面，所以按分位数删掉一部分。
        mesh.remove_vertices_by_mask(density_values < threshold)

    # Poisson 可能在点云包围盒外补出额外表面，这里裁剪回输入点云范围附近。
    bbox = cloud.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # 基础网格清理：删除退化三角形、重复元素和非流形边。
    # 这些操作不会让模型更“像”，但能减少后续查看/导出/打印时的异常。
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def transfer_vertex_colors_from_cloud(
    mesh: "o3d.geometry.TriangleMesh",
    cloud: "o3d.geometry.PointCloud",
) -> None:
    """把输入点云颜色转移到网格顶点。

    Poisson 重建生成的是新顶点，不会自动保留原始点颜色。
    如果输入 .ply/.pcd 本身带颜色，这里用最近邻查询：

        每个网格顶点 -> 找最近的原始点 -> 复制那个点的颜色

    OBJ 对顶点颜色支持不稳定，想看颜色更推荐保存/查看 PLY。
    """
    if not cloud.has_colors() or len(mesh.vertices) == 0:
        return

    cloud_points = np.asarray(cloud.points)
    cloud_colors = np.asarray(cloud.colors)
    mesh_vertices = np.asarray(mesh.vertices)
    if len(cloud_points) == 0 or len(cloud_colors) != len(cloud_points):
        return

    tree = cKDTree(cloud_points)
    _, indices = tree.query(mesh_vertices, k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(cloud_colors[indices])


def parse_args(argv: list[str]) -> argparse.Namespace:
    """定义命令行参数。

    参数既可以从命令行传入，也可以从 reconstruction_config.json 读取。
    命令行传入的值优先级更高。
    """
    parser = argparse.ArgumentParser(description="Poisson surface reconstruction from a point cloud.")
    add_common_io_args(parser, DEFAULT_CONFIG_KEY)
    parser.add_argument("--depth", type=int, help="Poisson octree depth. Higher values preserve more detail.")
    parser.add_argument("--scale", type=float, help="Poisson reconstruction scale.")
    parser.add_argument("--voxel-size", type=float, help="Voxel downsample size before reconstruction.")
    parser.add_argument("--linear-fit", action="store_true", default=None, help="Use linear interpolation in Poisson.")
    parser.add_argument("--normals-radius", type=float, help="Search radius for normal estimation.")
    parser.add_argument("--normals-max-nn", type=int, help="Maximum neighbors for normal estimation.")
    parser.add_argument("--orient-max-nn", type=int, help="Maximum neighbors for consistent normal orientation.")
    parser.add_argument("--density-quantile", type=float, help="Remove vertices below this density quantile.")
    parser.add_argument("--statistical-outlier-nb-neighbors", type=int, help="Neighbors for statistical outlier removal, 0 disables it.")
    parser.add_argument("--statistical-outlier-std-ratio", type=float, help="Standard deviation ratio for statistical outlier removal.")
    parser.add_argument("--radius-outlier-nb-points", type=int, help="Minimum neighbors for radius outlier removal, 0 disables it.")
    parser.add_argument("--radius-outlier-radius", type=float, help="Search radius for radius outlier removal.")
    parser.add_argument("--completion-rounds", type=int, help="Local point-cloud completion/upsampling rounds, 0 disables it.")
    parser.add_argument("--completion-max-distance", type=float, help="Maximum neighbor distance for midpoint completion, 0 uses an automatic threshold.")
    return parser.parse_args(argv)


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """补齐默认参数。

    如果配置文件和命令行都没有给某个参数，就在这里填一个安全默认值。
    注意：项目实际默认值通常来自 reconstruction_config.json；
    这里的值主要用于用户完全手写命令时兜底。
    """
    fill_common_defaults(args)
    if args.voxel_size is None:
        args.voxel_size = 0.05
    if args.depth is None:
        args.depth = 8
    if args.scale is None:
        args.scale = 1.1
    if args.linear_fit is None:
        args.linear_fit = False
    if args.normals_radius is None:
        args.normals_radius = 0.02
    if args.normals_max_nn is None:
        args.normals_max_nn = 30
    if args.orient_max_nn is None:
        args.orient_max_nn = args.normals_max_nn
    if args.density_quantile is None:
        args.density_quantile = 0.02
    if args.statistical_outlier_nb_neighbors is None:
        args.statistical_outlier_nb_neighbors = 0
    if args.statistical_outlier_std_ratio is None:
        args.statistical_outlier_std_ratio = 2.0
    if args.radius_outlier_nb_points is None:
        args.radius_outlier_nb_points = 0
    if args.radius_outlier_radius is None:
        args.radius_outlier_radius = 0.0
    if args.completion_rounds is None:
        args.completion_rounds = 0
    if args.completion_max_distance is None:
        args.completion_max_distance = 0.0
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        # 先把配置文件里的默认值应用到 args，再补齐缺省值。
        args = fill_defaults(apply_config_defaults(args))

        # 参数合法性检查。越早发现错误，越容易知道问题在哪里。
        if args.depth < 4:
            raise ValueError("--depth should be at least 4.")
        if args.normals_radius <= 0:
            raise ValueError("--normals-radius must be positive.")
        if args.normals_max_nn < 3:
            raise ValueError("--normals-max-nn must be at least 3.")
        if args.orient_max_nn < 3:
            raise ValueError("--orient-max-nn must be at least 3.")
        if not 0 <= args.density_quantile < 1:
            raise ValueError("--density-quantile must be in [0, 1).")
        if args.statistical_outlier_nb_neighbors < 0:
            raise ValueError("--statistical-outlier-nb-neighbors must be non-negative.")
        if args.statistical_outlier_std_ratio <= 0:
            raise ValueError("--statistical-outlier-std-ratio must be positive.")
        if args.radius_outlier_nb_points < 0:
            raise ValueError("--radius-outlier-nb-points must be non-negative.")
        if args.radius_outlier_radius < 0:
            raise ValueError("--radius-outlier-radius must be non-negative.")
        if args.completion_rounds < 0:
            raise ValueError("--completion-rounds must be non-negative.")
        if args.completion_max_distance < 0:
            raise ValueError("--completion-max-distance must be non-negative.")

        total_steps = 8
        print_progress(1, total_steps, "读取点云")
        cloud = read_point_cloud(args.input)
        print(f"读取点云: {args.input} 点数={len(cloud.points)}")

        # 体素下采样：
        # 把空间切成 voxel_size 大小的小立方体，每个体素保留一个代表点。
        # 点数越少，Poisson 越快；但 voxel_size 太大会丢细节。
        print_progress(2, total_steps, "体素下采样")
        if args.voxel_size > 0:
            cloud = cloud.voxel_down_sample(args.voxel_size)

        print(f"下采样后点数={len(cloud.points)}")

        print_progress(3, total_steps, "噪点处理")
        before_noise = len(cloud.points)
        cloud = remove_noise(
            cloud=cloud,
            statistical_nb_neighbors=args.statistical_outlier_nb_neighbors,
            statistical_std_ratio=args.statistical_outlier_std_ratio,
            radius_nb_points=args.radius_outlier_nb_points,
            radius=args.radius_outlier_radius,
        )
        print(f"噪点处理后点数={len(cloud.points)} 删除点数={before_noise - len(cloud.points)}")

        print_progress(4, total_steps, "点云补全")
        cloud, completion_added_points = complete_point_cloud(
            cloud=cloud,
            rounds=args.completion_rounds,
            max_distance=args.completion_max_distance,
        )
        print(f"点云补全后点数={len(cloud.points)} 新增点数={completion_added_points}")

        # Poisson 必须依赖法线。法线方向不统一时，算法会不知道哪里是内外，
        # 容易生成破面、尖刺或错误封闭面。
        print_progress(5, total_steps, "估计并统一法线")
        estimate_normals(cloud, args.normals_radius, args.normals_max_nn, args.orient_max_nn)

        print_progress(6, total_steps, "Poisson 重建")
        mesh = reconstruct_poisson(cloud, args.depth, args.scale, args.linear_fit, args.density_quantile)
        transfer_vertex_colors_from_cloud(mesh, cloud)
        print(f"泊松重建完成: 顶点数={len(mesh.vertices)} 三角形数={len(mesh.triangles)}")

        print_progress(7, total_steps, "写出网格和参数")
        mesh_paths, params_path = resolve_output_paths(args)
        write_meshes(mesh, mesh_paths)

        # 把本次运行的输入、输出、关键参数和结果规模写成 JSON。
        # 之后如果某组参数效果好，可以直接从 JSON 里复现。
        info = PoissonRunInfo(
            method="poisson",
            input=str(args.input),
            output_meshes=[str(path) for path in mesh_paths],
            voxel_size=args.voxel_size,
            point_count=len(cloud.points),
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.triangles),
            depth=args.depth,
            scale=args.scale,
            linear_fit=args.linear_fit,
            normals_radius=args.normals_radius,
            normals_max_nn=args.normals_max_nn,
            orient_max_nn=args.orient_max_nn,
            density_quantile=args.density_quantile,
            statistical_outlier_nb_neighbors=args.statistical_outlier_nb_neighbors,
            statistical_outlier_std_ratio=args.statistical_outlier_std_ratio,
            radius_outlier_nb_points=args.radius_outlier_nb_points,
            radius_outlier_radius=args.radius_outlier_radius,
            completion_rounds=args.completion_rounds,
            completion_max_distance=args.completion_max_distance,
            completion_added_points=completion_added_points,
        )
        write_json(params_path, info)

        print_progress(8, total_steps, "完成")
        for mesh_path in mesh_paths:
            print(f"保存重建网格: {mesh_path} 顶点数={len(mesh.vertices)} 三角形数={len(mesh.triangles)}")
        print(f"保存参数: {params_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
