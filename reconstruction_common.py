"""Shared helpers for point-cloud reconstruction scripts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import open3d as o3d  # type: ignore
except ImportError:  # pragma: no cover - depends on local environment
    o3d = None


SUPPORTED_TEXT_EXTENSIONS = {".xyz", ".txt", ".csv"}
NUMPY_READ_EXTENSIONS = {".npy"}
OPEN3D_READ_EXTENSIONS = {".ply", ".pcd"}
MESH_OUTPUT_EXTENSIONS = {".obj", ".ply", ".stl", ".off", ".gltf", ".glb"}
DEFAULT_OUTPUT_ROOT = Path("model_outputs")
PATH_FIELDS = {"input", "output", "params_out", "output_root"}
CONFIG_KEY_MAPPING = {
    "params-out": "params_out",
    "output-root": "output_root",
    "flat-output": "flat_output",
    "f-scale": "f_scale",
    "max-nfev": "max_nfev",
    "demo-points": "demo_points",
    "demo-noise": "demo_noise",
    "voxel-size": "voxel_size",
    "normals-radius": "normals_radius",
    "normals-max-nn": "normals_max_nn",
    "orient-max-nn": "orient_max_nn",
    "density-quantile": "density_quantile",
    "linear-fit": "linear_fit",
    "grid-resolution": "grid_resolution",
    "ball-radius": "ball_radius",
    "joggle-inputs": "joggle_inputs",
}


def add_common_io_args(parser: argparse.ArgumentParser, default_config_key: str) -> None:
    parser.add_argument("--config", type=Path, help="JSON config file containing method input paths and parameters.")
    parser.add_argument("--config-key", default=default_config_key, help="Method key inside the config file.")
    parser.add_argument("--input", type=Path, help="Input point cloud: .xyz/.txt/.csv/.ply/.pcd.")
    parser.add_argument("--output", type=Path, help="Output mesh path, typically .obj or .ply.")
    parser.add_argument("--params-out", type=Path, help="Output JSON run info file.")
    parser.add_argument("--output-root", type=Path, help="Root folder for organized model outputs.")
    parser.add_argument("--flat-output", action="store_true", default=None, help="Write outputs exactly beside --output.")


def load_method_config(config_path: Path, config_key: str) -> dict[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    methods = data.get("methods", data)
    if not isinstance(methods, dict) or config_key not in methods:
        raise KeyError(f"Config key {config_key!r} was not found in {config_path}.")
    method_config = methods[config_key]
    if not isinstance(method_config, dict):
        raise ValueError(f"Config key {config_key!r} must contain an object.")
    return method_config


def apply_config_defaults(args: argparse.Namespace, extra_mapping: dict[str, str] | None = None) -> argparse.Namespace:
    if args.config is None:
        return args

    config = load_method_config(args.config, args.config_key)
    mapping = dict(CONFIG_KEY_MAPPING)
    if extra_mapping:
        mapping.update(extra_mapping)

    for raw_key, value in config.items():
        key = mapping.get(raw_key, raw_key)
        if not hasattr(args, key) or getattr(args, key) is not None:
            continue
        if key in PATH_FIELDS and value is not None:
            value = Path(str(value))
        setattr(args, key, value)
    return args


def fill_common_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.output is None:
        raise ValueError("--output is required unless it is provided by --config.")
    if args.input is None:
        raise ValueError("--input is required unless it is provided by --config.")
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOT
    if args.flat_output is None:
        args.flat_output = False
    return args


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


def validate_points(points: np.ndarray, source: str, min_points: int = 20) -> None:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{source} must contain 3D points with shape (N, 3).")
    if len(points) < min_points:
        raise ValueError(f"{source} has only {len(points)} points; at least {min_points} are required.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{source} contains NaN or infinite values.")


def depth_image_to_points(depth: np.ndarray, source: str, min_points: int = 20) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"{source} depth image must have shape (H, W).")

    depth_values = depth.astype(float)
    finite_mask = np.isfinite(depth_values) & (depth_values > 0)
    if int(np.count_nonzero(finite_mask)) < min_points:
        raise ValueError(f"{source} has too few valid depth pixels.")

    height, width = depth_values.shape
    max_depth = float(np.nanmax(depth_values[finite_mask]))
    depth_scale = 1000.0 if max_depth > 100.0 else 1.0
    z = depth_values / depth_scale

    yy, xx = np.indices((height, width), dtype=float)
    fx = fy = float(max(width, height))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    points = np.column_stack((x[finite_mask], y[finite_mask], z[finite_mask]))
    validate_points(points, source, min_points=min_points)
    return points


def read_numpy_points(path: Path, min_points: int = 20) -> np.ndarray:
    array = np.load(path)
    if array.ndim == 2 and array.shape[1] in {3, 4, 6, 7}:
        points = np.asarray(array[:, :3], dtype=float)
        validate_points(points, str(path), min_points=min_points)
        return points
    if array.ndim == 3 and array.shape[2] >= 3:
        points = np.asarray(array[..., :3], dtype=float).reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        validate_points(points, str(path), min_points=min_points)
        return points
    if array.ndim == 2:
        return depth_image_to_points(array, str(path), min_points=min_points)
    raise ValueError(f"Unsupported .npy array shape {array.shape}; use (N, 3), (H, W, 3), or a 2D depth image.")


def read_text_points(path: Path, min_points: int = 20) -> np.ndarray:
    rows: list[list[float]] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            sample = handle.read(2048)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",; \t")
            except csv.Error:
                dialect = csv.excel
            for row in csv.reader(handle, dialect):
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
    validate_points(points, str(path), min_points=min_points)
    return points


def read_points(path: Path, min_points: int = 20) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    extension = path.suffix.lower()
    if extension in SUPPORTED_TEXT_EXTENSIONS:
        return read_text_points(path, min_points=min_points)
    if extension in NUMPY_READ_EXTENSIONS:
        return read_numpy_points(path, min_points=min_points)
    if extension in OPEN3D_READ_EXTENSIONS:
        if o3d is None:
            raise RuntimeError(f"Reading {extension} requires open3d. Install it with: python -m pip install open3d")
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=float)
        validate_points(points, str(path), min_points=min_points)
        return points
    raise ValueError("Unsupported input format. Use .xyz, .txt, .csv, .npy, .ply, or .pcd.")


def read_point_cloud(path: Path, min_points: int = 50) -> "o3d.geometry.PointCloud":
    if o3d is None:
        raise RuntimeError("This reconstruction method requires open3d. Install it with: python -m pip install open3d")

    if path.suffix.lower() in OPEN3D_READ_EXTENSIONS:
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        cloud = o3d.io.read_point_cloud(str(path))
        validate_points(np.asarray(cloud.points), str(path), min_points=min_points)
        return cloud

    points = read_points(path, min_points=min_points)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def estimate_normals(cloud: "o3d.geometry.PointCloud", radius: float, max_nn: int, orient_max_nn: int | None = None) -> None:
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    cloud.orient_normals_consistent_tangent_plane(orient_max_nn or max_nn)


def companion_mesh_paths(output: Path) -> list[Path]:
    paths = [output]
    extension = output.suffix.lower()
    if extension == ".obj":
        paths.append(output.with_suffix(".ply"))
    elif extension == ".ply":
        paths.append(output.with_suffix(".obj"))
    return paths


def organized_model_root(output: Path, output_root: Path) -> Path:
    if not output.stem:
        raise ValueError("--output must include a file name such as bunny.obj.")
    root = output.parent if output.parent != Path(".") else output_root
    return root / output.stem


def organized_mesh_paths(output: Path, output_root: Path) -> list[Path]:
    model_root = organized_model_root(output, output_root)
    return [model_root / path.suffix.lower().lstrip(".") / path.name for path in companion_mesh_paths(output)]


def organized_params_path(output: Path, params_out: Path | None, output_root: Path) -> Path:
    params_name = params_out.name if params_out is not None else f"{output.stem}.json"
    return organized_model_root(output, output_root) / "json" / params_name


def resolve_output_paths(args: argparse.Namespace) -> tuple[list[Path], Path]:
    mesh_paths = companion_mesh_paths(args.output) if args.flat_output else organized_mesh_paths(args.output, args.output_root)
    params_path = (
        args.params_out or args.output.with_suffix(".json")
        if args.flat_output
        else organized_params_path(args.output, args.params_out, args.output_root)
    )
    return mesh_paths, params_path


def write_meshes(mesh: "o3d.geometry.TriangleMesh", paths: list[Path]) -> None:
    if o3d is None:
        raise RuntimeError("Mesh export requires open3d.")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() not in MESH_OUTPUT_EXTENSIONS:
            raise ValueError(f"Unsupported mesh output extension: {path.suffix}")
        if not o3d.io.write_triangle_mesh(str(path), mesh):
            raise RuntimeError(f"Open3D failed to write mesh: {path}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(data) if is_dataclass(data) else data
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def print_output_summary(mesh_paths: list[Path], params_path: Path) -> None:
    for mesh_path in mesh_paths:
        print(f"wrote mesh: {mesh_path}")
    print(f"wrote params: {params_path}")


def print_progress(step: int, total: int, message: str, width: int = 30) -> None:
    ratio = step / total if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(ratio * 100)
    print(f"[{bar}] {percent:3d}%  {message}", flush=True)
