import open3d as o3d
import numpy as np
from pathlib import Path
from reconstruction_common import print_progress
from scipy.spatial import cKDTree


def transfer_vertex_colors_from_cloud(mesh, cloud):
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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Poisson surface reconstruction.")
    parser.add_argument("input", help="输入点云文件路径")
    parser.add_argument("output", help="输出网格文件路径")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="下采样体素大小，默认0.5")
    parser.add_argument("--normal-radius", type=float, default=0.05, help="法线估计搜索半径，默认0.25")
    parser.add_argument("--normal-max-nn", type=int, default=50, help="法线估计最大邻居数，默认50")
    parser.add_argument("--orient-max-nn", type=int, default=50, help="法线统一方向最大邻居数，默认100")
    parser.add_argument("--poisson-depth", type=int, default=7, help="泊松重建深度，默认9")
    parser.add_argument("--scale", type=float, default=1.1, help="泊松重建尺度，默认1.1")
    parser.add_argument("--density-quantile", type=float, default=0.02, help="密度过滤分位数，默认0.02")
    return parser.parse_args()


def poisson_reconstruction(
    input_path: str,
    output_path: str,
    voxel_size: float = 0.01,
    normal_radius: float = 0.05,
    normal_max_nn: int = 50,
    orient_max_nn: int = 100,
    poisson_depth: int = 9,
    scale: float = 1.1,
    density_quantile: float = 0.02,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_steps = 6

    #点云读取
    print_progress(1, total_steps, "读取点云")
    pcd = o3d.io.read_point_cloud(str(input_path))
    #
    if pcd.is_empty():
        raise ValueError(f"输入点云为空: {input_path}")
    print(f"读取点云: {input_path} 点数={len(pcd.points)}")

    #下采样
    print_progress(2, total_steps, "体素下采样")
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    print(f"下采样后点数={len(pcd.points)}")

    #点云法线估计
    print_progress(3, total_steps, "估计点云法线")
    pcd.estimate_normals(
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn
            )
    )

    #统一法线方向
    print_progress(4, total_steps, "统一法线方向")
    pcd.orient_normals_consistent_tangent_plane(k=orient_max_nn)

    #泊松重建
    print_progress(5, total_steps, "Poisson 重建")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=poisson_depth,
        scale=scale,
        linear_fit=False,
    )
    transfer_vertex_colors_from_cloud(mesh, pcd)
    print(f"泊松重建完成: 顶点数={len(mesh.vertices)} 三角形数={len(mesh.triangles)}")

    # # 1. 删除低密度顶点
    # densities = np.asarray(densities)
    # threshold = np.quantile(densities, density_quantile)
    # mesh.remove_vertices_by_mask(densities < threshold)

    # # 2. 删除孤立小连通块，只保留最大主体
    # triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    # triangle_clusters = np.asarray(triangle_clusters)
    # cluster_n_triangles = np.asarray(cluster_n_triangles)

    # largest_cluster_idx = cluster_n_triangles.argmax()
    # mesh.remove_triangles_by_mask(triangle_clusters != largest_cluster_idx)
    # mesh.remove_unreferenced_vertices()

    # # 3. 清理网格
    # mesh.remove_degenerate_triangles()
    # mesh.remove_duplicated_triangles()
    # mesh.remove_duplicated_vertices()
    # mesh.remove_non_manifold_edges()
    # mesh.compute_vertex_normals()

    #保存网格
    print_progress(6, total_steps, "保存网格")
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print(f"保存重建网格: {output_path} 顶点数={len(mesh.vertices)} 三角形数={len(mesh.triangles)}")


if __name__ == "__main__":
    args = parse_args()
    poisson_reconstruction(
        input_path=args.input,
        output_path=args.output,
        voxel_size=args.voxel_size,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        orient_max_nn=args.orient_max_nn,
        poisson_depth=args.poisson_depth,
        scale=args.scale,
        density_quantile=args.density_quantile,
    )

   
