"""Single-superquadric point cloud reconstruction.

This script fits one superellipsoid/superquadric primitive to an input point
cloud and exports both the fitted parameters and a triangle mesh.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from reconstruction_common import (
    DEFAULT_OUTPUT_ROOT,
    add_common_io_args,
    apply_config_defaults,
    companion_mesh_paths,
    organized_mesh_paths,
    organized_params_path,
)

try:
    import open3d as o3d  # type: ignore
except ImportError:  # pragma: no cover - depends on local environment
    o3d = None


SUPPORTED_TEXT_EXTENSIONS = {".xyz", ".txt", ".csv"}
OPEN3D_READ_EXTENSIONS = {".ply", ".pcd"}
MANUAL_MESH_EXTENSIONS = {".obj", ".ply"}
DEFAULT_CONFIG_KEY = "superquadric"


@dataclass
class FitResult:
    center: list[float]
    rotation_vector: list[float]
    axes: list[float]
    epsilon1: float
    epsilon2: float
    rmse: float
    mean_abs_residual: float


def signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    """Power of absolute values, with a tiny floor for numerical stability."""
    return np.power(np.maximum(np.abs(values), 1e-12), exponent)


def superquadric_g(local_points: np.ndarray, axes: np.ndarray, eps1: float, eps2: float) -> np.ndarray:
    """Compute the superquadric inside-outside function G for local points."""
    x = local_points[:, 0] / axes[0]
    y = local_points[:, 1] / axes[1]
    z = local_points[:, 2] / axes[2]

    xy = signed_power(x, 2.0 / eps2) + signed_power(y, 2.0 / eps2)
    return np.power(
        np.power(np.maximum(xy, 1e-12), eps2 / eps1) + signed_power(z, 2.0 / eps1),
        eps1 / 2.0,
    )


def unpack_params(params: np.ndarray) -> tuple[np.ndarray, Rotation, np.ndarray, float, float]:
    center = params[0:3]
    rotation = Rotation.from_rotvec(params[3:6])
    axes = np.exp(params[6:9])
    eps1 = float(params[9])
    eps2 = float(params[10])
    return center, rotation, axes, eps1, eps2


def residuals(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    center, rotation, axes, eps1, eps2 = unpack_params(params)
    local = rotation.inv().apply(points - center)
    g = superquadric_g(local, axes, eps1, eps2)
    return (g - 1.0) * float(np.mean(axes))


def read_text_point_cloud(path: Path) -> np.ndarray:
    """Read xyz-like text files and keep the first three numeric columns."""
    rows: list[list[float]] = []

    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(2048)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",; \t")
            except csv.Error:
                dialect = csv.excel
            reader: Iterable[list[str]] = csv.reader(handle, dialect)
            for row in reader:
                parsed = parse_numeric_triplet(row)
                if parsed is not None:
                    rows.append(parsed)
    else:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parsed = parse_numeric_triplet(stripped.replace(",", " ").split())
                if parsed is not None:
                    rows.append(parsed)

    points = np.asarray(rows, dtype=float)
    validate_points(points, source=str(path))
    return points


def parse_numeric_triplet(fields: Iterable[str]) -> list[float] | None:
    values: list[float] = []
    for field in fields:
        try:
            values.append(float(str(field).strip()))
        except ValueError:
            continue
        if len(values) == 3:
            return values
    return None


def read_point_cloud(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    extension = path.suffix.lower()
    if extension in SUPPORTED_TEXT_EXTENSIONS:
        return read_text_point_cloud(path)

    if extension in OPEN3D_READ_EXTENSIONS:
        if o3d is None:
            raise RuntimeError(
                f"Reading {extension} requires open3d. Install it with: python -m pip install open3d"
            )
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=float)
        validate_points(points, source=str(path))
        return points

    raise ValueError(
        f"Unsupported input extension {extension!r}. Supported: .xyz, .txt, .csv"
        + (", .ply, .pcd" if o3d is not None else " (.ply/.pcd require open3d)")
    )


def validate_points(points: np.ndarray, source: str = "point cloud") -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{source} must contain 3D points with shape (N, 3).")
    if len(points) < 20:
        raise ValueError(f"{source} has only {len(points)} points; at least 20 are recommended.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{source} contains NaN or infinite values.")
    if np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1)) <= 1e-12:
        raise ValueError(f"{source} appears to have no spatial extent.")


def pca_initial_params(points: np.ndarray) -> np.ndarray:
    """Initialize center, rotation, axes, and epsilons from PCA."""
    center = np.mean(points, axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order]

    if np.linalg.det(components) < 0:
        components[:, -1] *= -1.0

    local = centered @ components
    half_ranges = np.percentile(np.abs(local), 95, axis=0)
    fallback_scale = np.maximum(np.std(local, axis=0) * 2.0, 1e-3)
    axes = np.maximum(half_ranges, fallback_scale)
    axes = np.maximum(axes, 1e-4)

    rotation_vector = Rotation.from_matrix(components).as_rotvec()
    return np.concatenate([center, rotation_vector, np.log(axes), np.array([1.0, 1.0])])


def fit_superquadric(points: np.ndarray, f_scale: float, max_nfev: int) -> FitResult:
    initial = pca_initial_params(points)
    centered = points - np.mean(points, axis=0)
    cloud_scale = max(float(np.percentile(np.linalg.norm(centered, axis=1), 95)), 1e-3)

    lower = np.full(11, -np.inf)
    upper = np.full(11, np.inf)
    lower[6:9] = np.log(np.full(3, cloud_scale * 1e-3))
    upper[6:9] = np.log(np.full(3, cloud_scale * 20.0))
    lower[9:11] = 0.1
    upper[9:11] = 2.0

    result = least_squares(
        residuals,
        initial,
        args=(points,),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=f_scale,
        max_nfev=max_nfev,
        x_scale="jac",
    )

    final_residuals = residuals(result.x, points)
    center, rotation, axes, eps1, eps2 = unpack_params(result.x)
    return FitResult(
        center=center.tolist(),
        rotation_vector=rotation.as_rotvec().tolist(),
        axes=axes.tolist(),
        epsilon1=eps1,
        epsilon2=eps2,
        rmse=float(np.sqrt(np.mean(np.square(final_residuals)))),
        mean_abs_residual=float(np.mean(np.abs(final_residuals))),
    )


def superquadric_surface_point(eta: np.ndarray, omega: np.ndarray, axes: np.ndarray, eps1: float, eps2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_eta = np.cos(eta)
    sin_eta = np.sin(eta)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    x = axes[0] * np.sign(cos_eta) * np.abs(cos_eta) ** eps1 * np.sign(cos_omega) * np.abs(cos_omega) ** eps2
    y = axes[1] * np.sign(cos_eta) * np.abs(cos_eta) ** eps1 * np.sign(sin_omega) * np.abs(sin_omega) ** eps2
    z = axes[2] * np.sign(sin_eta) * np.abs(sin_eta) ** eps1
    return x, y, z


def generate_mesh(fit: FitResult, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    if resolution < 8:
        raise ValueError("Mesh resolution must be at least 8.")

    axes = np.asarray(fit.axes, dtype=float)
    center = np.asarray(fit.center, dtype=float)
    rotation = Rotation.from_rotvec(np.asarray(fit.rotation_vector, dtype=float))

    rows = resolution + 1
    cols = resolution * 2
    etas = np.linspace(-math.pi / 2.0, math.pi / 2.0, rows)
    omegas = np.linspace(-math.pi, math.pi, cols, endpoint=False)
    eta_grid, omega_grid = np.meshgrid(etas, omegas, indexing="ij")
    x, y, z = superquadric_surface_point(eta_grid, omega_grid, axes, fit.epsilon1, fit.epsilon2)
    local_vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = rotation.apply(local_vertices) + center

    faces: list[list[int]] = []
    for i in range(rows - 1):
        for j in range(cols):
            a = i * cols + j
            b = i * cols + (j + 1) % cols
            c = (i + 1) * cols + (j + 1) % cols
            d = (i + 1) * cols + j
            faces.append([a, d, c])
            faces.append([a, c, b])
    return vertices, np.asarray(faces, dtype=np.int32)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Superquadric reconstruction\n")
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            handle.write(f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    extension = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if extension == ".obj":
        write_obj(path, vertices, faces)
    elif extension == ".ply":
        write_ply(path, vertices, faces)
    elif extension in {".stl", ".off", ".gltf", ".glb"}:
        if o3d is None:
            raise RuntimeError(
                f"Writing {extension} requires open3d. Install open3d or choose .obj/.ply output."
            )
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(path), mesh):
            raise RuntimeError(f"Open3D failed to write mesh: {path}")
    else:
        raise ValueError(f"Unsupported mesh output extension {extension!r}. Use .obj or .ply.")


def write_params(path: Path, fit: FitResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        import json

        json.dump(asdict(fit), handle, indent=2)
        handle.write("\n")


def create_demo_points(count: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fit = FitResult(
        center=[0.4, -0.2, 0.15],
        rotation_vector=[0.35, -0.25, 0.15],
        axes=[1.4, 0.8, 0.55],
        epsilon1=0.65,
        epsilon2=1.35,
        rmse=0.0,
        mean_abs_residual=0.0,
    )
    vertices, _ = generate_mesh(fit, resolution=max(16, int(math.sqrt(count))))
    selected = vertices[rng.choice(len(vertices), size=count, replace=True)]
    return selected + rng.normal(scale=noise, size=selected.shape)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a single superquadric to a point cloud.")
    add_common_io_args(parser, DEFAULT_CONFIG_KEY)
    parser.add_argument("--demo", action="store_true", help="Run a built-in noisy superquadric demo.")
    parser.add_argument("--resolution", type=int, help="Mesh latitude resolution. Columns are 2x this value.")
    parser.add_argument("--f-scale", type=float, help="soft_l1 robust loss scale for least_squares.")
    parser.add_argument("--max-nfev", type=int, help="Maximum least_squares function evaluations.")
    parser.add_argument("--demo-points", type=int, help="Number of generated demo points.")
    parser.add_argument("--demo-noise", type=float, help="Gaussian noise scale for demo points.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        args = apply_config_defaults(args)
        if args.output is None:
            raise ValueError("--output is required unless it is provided by --config.")
        if args.output_root is None:
            args.output_root = DEFAULT_OUTPUT_ROOT
        if args.flat_output is None:
            args.flat_output = False
        if args.resolution is None:
            args.resolution = 48
        if args.f_scale is None:
            args.f_scale = 0.05
        if args.max_nfev is None:
            args.max_nfev = 1500
        if args.demo_points is None:
            args.demo_points = 2500
        if args.demo_noise is None:
            args.demo_noise = 0.015

        if args.demo:
            points = create_demo_points(args.demo_points, args.demo_noise, seed=7)
        else:
            if args.input is None:
                raise ValueError("--input is required unless --demo is used.")
            points = read_point_cloud(args.input)

        if args.f_scale <= 0:
            raise ValueError("--f-scale must be positive.")

        fit = fit_superquadric(points, f_scale=args.f_scale, max_nfev=args.max_nfev)
        vertices, faces = generate_mesh(fit, resolution=args.resolution)

        mesh_paths = companion_mesh_paths(args.output) if args.flat_output else organized_mesh_paths(args.output, args.output_root)
        for mesh_path in mesh_paths:
            write_mesh(mesh_path, vertices, faces)

        params_path = (
            args.params_out or args.output.with_suffix(".json")
            if args.flat_output
            else organized_params_path(args.output, args.params_out, args.output_root)
        )
        write_params(params_path, fit)

        print(f"Fitted superquadric to {len(points)} points")
        print(f"axes={np.array(fit.axes)} epsilon=({fit.epsilon1:.4f}, {fit.epsilon2:.4f})")
        print(f"rmse={fit.rmse:.6g} mean_abs_residual={fit.mean_abs_residual:.6g}")
        for mesh_path in mesh_paths:
            print(f"wrote mesh: {mesh_path}")
        print(f"wrote params: {params_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
