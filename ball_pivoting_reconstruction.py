"""Ball Pivoting surface reconstruction from point clouds."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from reconstruction_common import (
    add_common_io_args,
    apply_config_defaults,
    estimate_normals,
    fill_common_defaults,
    o3d,
    print_output_summary,
    read_point_cloud,
    resolve_output_paths,
    write_json,
    write_meshes,
)


DEFAULT_CONFIG_KEY = "ball_pivoting"


@dataclass
class BallPivotingRunInfo:
    method: str
    input: str
    output_meshes: list[str]
    point_count: int
    vertex_count: int
    triangle_count: int
    radii: list[float]
    normals_radius: float
    normals_max_nn: int


def parse_radii(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        radii = [float(item) for item in value]
    else:
        radii = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("Ball Pivoting radii must be positive numbers.")
    return radii


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ball Pivoting surface reconstruction.")
    add_common_io_args(parser, DEFAULT_CONFIG_KEY)
    parser.add_argument("--radii", help="Comma-separated ball radii, for example 0.004,0.008,0.016.")
    parser.add_argument("--normals-radius", type=float, help="Search radius for normal estimation.")
    parser.add_argument("--normals-max-nn", type=int, help="Maximum neighbors for normal estimation.")
    return parser.parse_args(argv)


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    fill_common_defaults(args)
    if args.radii is None:
        args.radii = [0.003, 0.006, 0.012, 0.024]
    else:
        args.radii = parse_radii(args.radii)
    if args.normals_radius is None:
        args.normals_radius = max(args.radii) * 2.5
    if args.normals_max_nn is None:
        args.normals_max_nn = 30
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        args = fill_defaults(apply_config_defaults(args))
        if o3d is None:
            raise RuntimeError("Ball Pivoting requires open3d.")
        if args.normals_radius <= 0:
            raise ValueError("--normals-radius must be positive.")
        if args.normals_max_nn < 3:
            raise ValueError("--normals-max-nn must be at least 3.")

        cloud = read_point_cloud(args.input)
        estimate_normals(cloud, args.normals_radius, args.normals_max_nn)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud,
            o3d.utility.DoubleVector(args.radii),
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        mesh_paths, params_path = resolve_output_paths(args)
        write_meshes(mesh, mesh_paths)
        info = BallPivotingRunInfo(
            method="ball_pivoting",
            input=str(args.input),
            output_meshes=[str(path) for path in mesh_paths],
            point_count=len(cloud.points),
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.triangles),
            radii=[float(radius) for radius in args.radii],
            normals_radius=args.normals_radius,
            normals_max_nn=args.normals_max_nn,
        )
        write_json(params_path, info)

        print(f"Ball Pivoting reconstructed {len(cloud.points)} points")
        print(f"mesh vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
        print_output_summary(mesh_paths, params_path)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
