"""Alpha Shape surface reconstruction from point clouds."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from reconstruction_common import (
    add_common_io_args,
    apply_config_defaults,
    fill_common_defaults,
    o3d,
    print_output_summary,
    read_point_cloud,
    resolve_output_paths,
    write_json,
    write_meshes,
)


DEFAULT_CONFIG_KEY = "alpha_shape"


@dataclass
class AlphaShapeRunInfo:
    method: str
    input: str
    output_meshes: list[str]
    point_count: int
    vertex_count: int
    triangle_count: int
    alpha: float


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha Shape surface reconstruction.")
    add_common_io_args(parser, DEFAULT_CONFIG_KEY)
    parser.add_argument("--alpha", type=float, help="Alpha radius threshold.")
    return parser.parse_args(argv)


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    fill_common_defaults(args)
    if args.alpha is None:
        args.alpha = 0.03
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        args = fill_defaults(apply_config_defaults(args))
        if o3d is None:
            raise RuntimeError("Alpha Shape reconstruction requires open3d.")
        if args.alpha <= 0:
            raise ValueError("--alpha must be positive.")

        cloud = read_point_cloud(args.input)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, args.alpha)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        mesh_paths, params_path = resolve_output_paths(args)
        write_meshes(mesh, mesh_paths)
        info = AlphaShapeRunInfo(
            method="alpha_shape",
            input=str(args.input),
            output_meshes=[str(path) for path in mesh_paths],
            point_count=len(cloud.points),
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.triangles),
            alpha=args.alpha,
        )
        write_json(params_path, info)

        print(f"Alpha Shape reconstructed {len(cloud.points)} points")
        print(f"mesh vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
        print_output_summary(mesh_paths, params_path)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
