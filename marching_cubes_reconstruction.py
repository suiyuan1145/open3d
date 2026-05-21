"""Marching Cubes reconstruction from a point-cloud-derived implicit field."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

try:
    from skimage.measure import marching_cubes
except ImportError:  # pragma: no cover - depends on local environment
    marching_cubes = None

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


DEFAULT_CONFIG_KEY = "marching_cubes"


@dataclass
class MarchingCubesRunInfo:
    method: str
    input: str
    output_meshes: list[str]
    point_count: int
    vertex_count: int
    triangle_count: int
    grid_resolution: int
    ball_radius: float
    padding: float
    level: float


def build_implicit_field(points: np.ndarray, grid_resolution: int, ball_radius: float | None, padding: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], float]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    extent = np.maximum(maxs - mins, 1e-6)
    diagonal = float(np.linalg.norm(extent))
    pad = diagonal * padding
    mins = mins - pad
    maxs = maxs + pad

    if ball_radius is None:
        ball_radius = diagonal / grid_resolution * 2.5
    if ball_radius <= 0:
        raise ValueError("--ball-radius must be positive.")

    axes = [np.linspace(mins[i], maxs[i], grid_resolution) for i in range(3)]
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    tree = cKDTree(points)
    distances, _ = tree.query(grid, k=1)
    field = (distances - ball_radius).reshape((grid_resolution, grid_resolution, grid_resolution))
    return field, mins, spacing, ball_radius


def reconstruct_marching_cubes(points: np.ndarray, grid_resolution: int, ball_radius: float | None, padding: float, level: float) -> tuple["o3d.geometry.TriangleMesh", float]:
    if marching_cubes is None:
        raise RuntimeError("Marching Cubes requires scikit-image. Install it with: python -m pip install scikit-image")
    if o3d is None:
        raise RuntimeError("Marching Cubes mesh export requires open3d.")
    if grid_resolution < 16:
        raise ValueError("--grid-resolution must be at least 16.")
    if padding < 0:
        raise ValueError("--padding must be non-negative.")

    field, origin, spacing, used_radius = build_implicit_field(points, grid_resolution, ball_radius, padding)
    if not (np.min(field) <= level <= np.max(field)):
        raise ValueError("Marching Cubes level is outside the scalar field range. Try increasing --ball-radius.")

    vertices, faces, _, _ = marching_cubes(field, level=level, spacing=spacing)
    vertices = vertices + origin

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    return mesh, used_radius


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marching Cubes reconstruction from an implicit point field.")
    add_common_io_args(parser, DEFAULT_CONFIG_KEY)
    parser.add_argument("--grid-resolution", type=int, help="Number of grid samples per axis.")
    parser.add_argument("--ball-radius", type=float, help="Implicit radius around each point. Default derives from cloud size.")
    parser.add_argument("--padding", type=float, help="Bounding-box padding as a fraction of cloud diagonal.")
    parser.add_argument("--level", type=float, help="Isosurface level.")
    return parser.parse_args(argv)


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    fill_common_defaults(args)
    if args.grid_resolution is None:
        args.grid_resolution = 64
    if args.padding is None:
        args.padding = 0.05
    if args.level is None:
        args.level = 0.0
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        args = fill_defaults(apply_config_defaults(args))
        cloud = read_point_cloud(args.input)
        points = np.asarray(cloud.points)
        mesh, used_radius = reconstruct_marching_cubes(
            points,
            args.grid_resolution,
            args.ball_radius,
            args.padding,
            args.level,
        )

        mesh_paths, params_path = resolve_output_paths(args)
        write_meshes(mesh, mesh_paths)
        info = MarchingCubesRunInfo(
            method="marching_cubes",
            input=str(args.input),
            output_meshes=[str(path) for path in mesh_paths],
            point_count=len(points),
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.triangles),
            grid_resolution=args.grid_resolution,
            ball_radius=used_radius,
            padding=args.padding,
            level=args.level,
        )
        write_json(params_path, info)

        print(f"Marching Cubes reconstructed {len(points)} points")
        print(f"mesh vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
        print_output_summary(mesh_paths, params_path)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
