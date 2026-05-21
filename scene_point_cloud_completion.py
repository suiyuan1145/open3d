"""Geometry-based scene point-cloud completion for large LiDAR/indoor scans.

This is intended for scene point clouds such as classrooms or Livox scans.
Unlike object-completion networks such as VRCNet, it does not guess semantic
object shapes. It performs conservative local MLS-style upsampling on existing
surfaces so Poisson reconstruction has denser, more stable input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from reconstruction_common import o3d, read_point_cloud


def estimate_local_frame(neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(neighbors, axis=0)
    centered = neighbors - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent_u = vh[0]
    tangent_v = vh[1]
    normal = vh[2]
    return tangent_u, tangent_v, normal


def geometry_complete_points(
    points: np.ndarray,
    upsample_factor: int,
    k_neighbors: int,
    radius: float,
    seed: int,
    max_source_points: int,
) -> np.ndarray:
    if upsample_factor <= 0:
        return points
    if len(points) < k_neighbors:
        raise ValueError("Not enough points for local completion.")

    rng = np.random.default_rng(seed)
    if max_source_points > 0 and len(points) > max_source_points:
        source_indices = rng.choice(len(points), size=max_source_points, replace=False)
    else:
        source_indices = np.arange(len(points))

    tree = cKDTree(points)
    generated: list[np.ndarray] = []
    for index in source_indices:
        distances, indices = tree.query(points[index], k=k_neighbors)
        valid = np.asarray(indices)[np.asarray(distances) <= radius] if radius > 0 else np.asarray(indices)
        if len(valid) < 6:
            continue
        neighbors = points[valid]
        tangent_u, tangent_v, _ = estimate_local_frame(neighbors)
        local_scale = float(np.median(np.linalg.norm(neighbors - points[index], axis=1)))
        if not np.isfinite(local_scale) or local_scale <= 1e-9:
            continue
        jitter_scale = local_scale * 0.35
        offsets = (
            rng.normal(0.0, jitter_scale, size=(upsample_factor, 1)) * tangent_u
            + rng.normal(0.0, jitter_scale, size=(upsample_factor, 1)) * tangent_v
        )
        generated.append(points[index] + offsets)

    if not generated:
        return points
    return np.vstack([points, *generated])


def save_cloud(path: Path, points: np.ndarray) -> None:
    if o3d is None:
        raise RuntimeError("open3d is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Failed to write point cloud: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry-based scene point-cloud completion.")
    parser.add_argument("--input", type=Path, required=True, help="Input scene point cloud.")
    parser.add_argument("--output", type=Path, required=True, help="Output completed point cloud .ply/.pcd.")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel downsample before completion.")
    parser.add_argument("--statistical-nb-neighbors", type=int, default=10, help="Statistical denoise neighbors, 0 disables.")
    parser.add_argument("--statistical-std-ratio", type=float, default=3.0, help="Statistical denoise std ratio.")
    parser.add_argument("--upsample-factor", type=int, default=2, help="New local points generated per source point.")
    parser.add_argument("--k-neighbors", type=int, default=24, help="Neighbors used for local plane fitting.")
    parser.add_argument("--radius", type=float, default=0.2, help="Maximum local neighborhood radius.")
    parser.add_argument("--max-source-points", type=int, default=60000, help="Maximum source points to upsample.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.voxel_size < 0:
        raise ValueError("--voxel-size must be non-negative.")
    if args.upsample_factor < 0:
        raise ValueError("--upsample-factor must be non-negative.")
    if args.k_neighbors < 6:
        raise ValueError("--k-neighbors must be at least 6.")
    if args.radius < 0:
        raise ValueError("--radius must be non-negative.")

    cloud = read_point_cloud(args.input)
    original_count = len(cloud.points)
    if args.voxel_size > 0:
        cloud = cloud.voxel_down_sample(args.voxel_size)
    downsampled_count = len(cloud.points)

    if args.statistical_nb_neighbors > 0:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=args.statistical_nb_neighbors,
            std_ratio=args.statistical_std_ratio,
        )
    denoised_points = np.asarray(cloud.points)
    completed = geometry_complete_points(
        points=denoised_points,
        upsample_factor=args.upsample_factor,
        k_neighbors=args.k_neighbors,
        radius=args.radius,
        seed=args.seed,
        max_source_points=args.max_source_points,
    )
    save_cloud(args.output, completed)

    print(f"input points: {original_count}")
    print(f"downsampled points: {downsampled_count}")
    print(f"denoised points: {len(denoised_points)}")
    print(f"completed points: {len(completed)}")
    print(f"wrote completed scene cloud: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
