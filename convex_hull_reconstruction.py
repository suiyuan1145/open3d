"""Convex Hull reconstruction from point clouds."""

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


DEFAULT_CONFIG_KEY = "convex_hull"


@dataclass
class ConvexHullRunInfo:
    method: str
    input: str
    output_meshes: list[str]
    point_count: int
    vertex_count: int
    triangle_count: int
    joggle_inputs: bool


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convex Hull reconstruction.")
    add_common_io_args(parser, DEFAULT_CONFIG_KEY)
    parser.add_argument("--joggle-inputs", action="store_true", default=None, help="Perturb nearly degenerate inputs for QHull.")
    return parser.parse_args(argv)


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    fill_common_defaults(args)
    if args.joggle_inputs is None:
        args.joggle_inputs = False
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        args = fill_defaults(apply_config_defaults(args))
        if o3d is None:
            raise RuntimeError("Convex Hull reconstruction requires open3d.")

        cloud = read_point_cloud(args.input)
        mesh, _ = cloud.compute_convex_hull(joggle_inputs=args.joggle_inputs)
        mesh.compute_vertex_normals()

        mesh_paths, params_path = resolve_output_paths(args)
        write_meshes(mesh, mesh_paths)
        info = ConvexHullRunInfo(
            method="convex_hull",
            input=str(args.input),
            output_meshes=[str(path) for path in mesh_paths],
            point_count=len(cloud.points),
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.triangles),
            joggle_inputs=args.joggle_inputs,
        )
        write_json(params_path, info)

        print(f"Convex Hull reconstructed {len(cloud.points)} points")
        print(f"mesh vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
        print_output_summary(mesh_paths, params_path)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
