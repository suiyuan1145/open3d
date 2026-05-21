"""View a point-cloud file with Open3D."""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d

from reconstruction_common import read_point_cloud


DEFAULT_INPUT = Path("cloud_registered_20260511_114914.pcd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="显示点云文件，支持 .pcd/.ply/.xyz/.txt/.csv/.npy。")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入点云文件路径，默认 {DEFAULT_INPUT}",
    )
    parser.add_argument("--voxel-size", type=float, default=0.0, help="显示前体素下采样大小，0 表示不下采样。")
    parser.add_argument("--point-size", type=float, default=2.0, help="显示窗口中的点大小。")
    parser.add_argument("--estimate-normals", action="store_true", help="显示前估计点云法线。")
    return parser.parse_args()


def load_point_cloud(path: Path, voxel_size: float, estimate_normals: bool) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(f"点云文件不存在: {path}")

    pcd = read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"点云读取失败或点云为空: {path}")

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(voxel_size * 3.0, 0.03),
                max_nn=50,
            )
        )

    return pcd


def main() -> int:
    args = parse_args()
    pcd = load_point_cloud(args.input, args.voxel_size, args.estimate_normals)

    print(f"读取点云: {args.input}")
    print(f"点数: {len(pcd.points)}")
    print(pcd)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=f"PCD Viewer - {args.input.name}", width=1280, height=720)
    visualizer.add_geometry(pcd)
    render_option = visualizer.get_render_option()
    render_option.point_size = args.point_size
    render_option.background_color = [0.02, 0.02, 0.02]
    visualizer.run()
    visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
