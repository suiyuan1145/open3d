"""Tune Poisson reconstruction parameters from point clouds.

The script performs self-supervised parameter search. It reconstructs a mesh
for each candidate parameter set, samples points from the mesh, and scores how
well the reconstructed mesh fits the original point cloud.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from poisson_reconstruction import reconstruct_poisson
from reconstruction_common import estimate_normals, o3d, print_progress, read_point_cloud, write_json


DEFAULT_CONFIG_PATH = Path("reconstruction_config.json")
DEFAULT_OUTPUT_PATH = Path("model_outputs") / "training" / "poisson_training.json"


@dataclass
class PoissonParams:
    depth: int
    scale: float
    normals_radius: float
    normals_max_nn: int
    orient_max_nn: int
    density_quantile: float
    linear_fit: bool


@dataclass
class PoissonCandidateScore:
    params: PoissonParams
    score: float
    chamfer_l1: float
    accuracy: float
    completeness: float
    precision: float
    recall: float
    f_score: float
    component_count: int
    vertex_count: int
    triangle_count: int


@dataclass
class PoissonTrainingCase:
    input: str
    best_params: PoissonParams
    best_score: float
    scores: list[PoissonCandidateScore]


@dataclass
class PoissonTrainingResult:
    method: str
    strategy: str
    recommended_params: PoissonParams
    inputs: list[str]
    sample_points: int
    max_eval_points: int
    voxel_size: float
    max_reconstruction_points: int
    component_penalty: float
    f_score_threshold: float
    f_score_weight: float
    cases: list[PoissonTrainingCase]


def parse_int_list(value: str | list[int]) -> list[int]:
    values = [int(item) for item in value] if isinstance(value, list) else [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("Integer candidates must be positive.")
    return sorted(set(values))


def parse_float_list(value: str | list[float]) -> list[float]:
    values = [float(item) for item in value] if isinstance(value, list) else [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise ValueError("Float candidates must be non-negative.")
    return sorted(set(values))


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def load_training_config(config_path: Path) -> dict[str, object]:
    config = load_json(config_path)
    training = config.get("training", {})
    if not isinstance(training, dict):
        return {}
    poisson_config = training.get("poisson", {})
    return poisson_config if isinstance(poisson_config, dict) else {}


def default_inputs_from_config(config_path: Path) -> list[Path]:
    config = load_json(config_path)
    poisson_config = config.get("methods", {}).get("poisson", {}) if isinstance(config.get("methods"), dict) else {}
    input_path = poisson_config.get("input") if isinstance(poisson_config, dict) else None
    return [Path(str(input_path))] if input_path else []


def update_poisson_config(config_path: Path, params: PoissonParams) -> None:
    config = load_json(config_path)
    methods = config.setdefault("methods", {})
    if not isinstance(methods, dict):
        raise ValueError("Config field 'methods' must be an object.")
    poisson = methods.setdefault("poisson", {})
    if not isinstance(poisson, dict):
        raise ValueError("Config field 'methods.poisson' must be an object.")

    poisson["depth"] = params.depth
    poisson["scale"] = params.scale
    poisson["normals_radius"] = params.normals_radius
    poisson["normals_max_nn"] = params.normals_max_nn
    poisson["orient_max_nn"] = params.orient_max_nn
    poisson["density_quantile"] = params.density_quantile
    poisson["linear_fit"] = params.linear_fit

    training = config.setdefault("training", {})
    if isinstance(training, dict):
        poisson_training = training.setdefault("poisson", {})
        if isinstance(poisson_training, dict):
            poisson_training["recommended_params"] = {
                "depth": params.depth,
                "scale": params.scale,
                "normals_radius": params.normals_radius,
                "normals_max_nn": params.normals_max_nn,
                "orient_max_nn": params.orient_max_nn,
                "density_quantile": params.density_quantile,
                "linear_fit": params.linear_fit,
            }

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def downsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def prepare_training_cloud(
    cloud: "o3d.geometry.PointCloud",
    voxel_size: float,
    max_reconstruction_points: int,
    seed: int,
) -> "o3d.geometry.PointCloud":
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    if max_reconstruction_points > 0 and len(cloud.points) > max_reconstruction_points:
        points = downsample_points(np.asarray(cloud.points), max_reconstruction_points, seed)
        reduced = o3d.geometry.PointCloud()
        reduced.points = o3d.utility.Vector3dVector(points)
        if cloud.has_colors():
            reduced.colors = o3d.utility.Vector3dVector(downsample_points(np.asarray(cloud.colors), max_reconstruction_points, seed))
        return reduced
    return cloud


def mesh_component_count(mesh: "o3d.geometry.TriangleMesh") -> int:
    if len(mesh.triangles) == 0:
        return 0
    labels, _, _ = mesh.cluster_connected_triangles()
    label_array = np.asarray(labels)
    return int(label_array.max() + 1) if len(label_array) else 0


def score_mesh(
    points: np.ndarray,
    mesh: "o3d.geometry.TriangleMesh",
    sample_points: int,
    component_penalty: float,
    f_score_threshold: float,
    f_score_weight: float,
) -> tuple[float, float, float, float, float, float, float, int]:
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return float("inf"), float("inf"), float("inf"), float("inf"), 0.0, 0.0, 0.0, 0

    sampled = mesh.sample_points_uniformly(number_of_points=sample_points)
    mesh_points = np.asarray(sampled.points)
    if len(mesh_points) == 0:
        return float("inf"), float("inf"), float("inf"), float("inf"), 0.0, 0.0, 0.0, 0

    point_tree = cKDTree(points)
    mesh_tree = cKDTree(mesh_points)
    point_to_mesh, _ = mesh_tree.query(points, k=1)
    mesh_to_point, _ = point_tree.query(mesh_points, k=1)

    diagonal = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    scale = max(diagonal, 1e-9)
    completeness = float(np.mean(point_to_mesh)) / scale
    accuracy = float(np.mean(mesh_to_point)) / scale
    chamfer_l1 = accuracy + completeness

    normalized_threshold = max(f_score_threshold, 1e-12)
    precision = float(np.mean((mesh_to_point / scale) <= normalized_threshold))
    recall = float(np.mean((point_to_mesh / scale) <= normalized_threshold))
    f_score = 0.0 if precision + recall <= 0 else float(2.0 * precision * recall / (precision + recall))

    components = mesh_component_count(mesh)
    score = chamfer_l1 + (1.0 - f_score) * f_score_weight + max(0, components - 1) * component_penalty
    return score, chamfer_l1, accuracy, completeness, precision, recall, f_score, components


def build_candidates(args: argparse.Namespace) -> list[PoissonParams]:
    return [
        PoissonParams(
            depth=depth,
            scale=scale,
            normals_radius=normals_radius,
            normals_max_nn=normals_max_nn,
            orient_max_nn=orient_max_nn,
            density_quantile=density_quantile,
            linear_fit=linear_fit,
        )
        for depth, scale, normals_radius, normals_max_nn, orient_max_nn, density_quantile, linear_fit in itertools.product(
            args.depth_values,
            args.scale_values,
            args.normals_radius_values,
            args.normals_max_nn_values,
            args.orient_max_nn_values,
            args.density_quantile_values,
            args.linear_fit_values,
        )
    ]


def tune_one_cloud(
    input_path: Path,
    candidates: list[PoissonParams],
    sample_points: int,
    max_eval_points: int,
    voxel_size: float,
    max_reconstruction_points: int,
    component_penalty: float,
    f_score_threshold: float,
    f_score_weight: float,
    seed: int,
) -> PoissonTrainingCase:
    base_cloud = read_point_cloud(input_path)
    train_cloud = prepare_training_cloud(base_cloud, voxel_size, max_reconstruction_points, seed)
    eval_points = downsample_points(np.asarray(train_cloud.points), max_eval_points, seed)

    scores: list[PoissonCandidateScore] = []
    for index, params in enumerate(candidates, start=1):
        print_progress(index, len(candidates), f"{input_path.name} 训练候选 {index}/{len(candidates)}")
        cloud = o3d.geometry.PointCloud(train_cloud)
        estimate_normals(cloud, params.normals_radius, params.normals_max_nn, params.orient_max_nn)
        mesh = reconstruct_poisson(
            cloud=cloud,
            depth=params.depth,
            scale=params.scale,
            linear_fit=params.linear_fit,
            density_quantile=params.density_quantile,
        )
        score, chamfer_l1, accuracy, completeness, precision, recall, f_score, components = score_mesh(
            points=eval_points,
            mesh=mesh,
            sample_points=sample_points,
            component_penalty=component_penalty,
            f_score_threshold=f_score_threshold,
            f_score_weight=f_score_weight,
        )
        scores.append(
            PoissonCandidateScore(
                params=params,
                score=score,
                chamfer_l1=chamfer_l1,
                accuracy=accuracy,
                completeness=completeness,
                precision=precision,
                recall=recall,
                f_score=f_score,
                component_count=components,
                vertex_count=len(mesh.vertices),
                triangle_count=len(mesh.triangles),
            )
        )

    finite_scores = [item for item in scores if math.isfinite(item.score)]
    if not finite_scores:
        raise RuntimeError(f"No valid Poisson candidate produced a mesh for {input_path}.")
    best = min(finite_scores, key=lambda item: item.score)
    return PoissonTrainingCase(
        input=str(input_path),
        best_params=best.params,
        best_score=best.score,
        scores=scores,
    )


def average_best_params(cases: list[PoissonTrainingCase]) -> PoissonParams:
    best_params = [case.best_params for case in cases]
    return PoissonParams(
        depth=int(round(float(np.mean([params.depth for params in best_params])))),
        scale=float(np.mean([params.scale for params in best_params])),
        normals_radius=float(np.mean([params.normals_radius for params in best_params])),
        normals_max_nn=int(round(float(np.mean([params.normals_max_nn for params in best_params])))),
        orient_max_nn=int(round(float(np.mean([params.orient_max_nn for params in best_params])))),
        density_quantile=float(np.mean([params.density_quantile for params in best_params])),
        linear_fit=bool(round(float(np.mean([1.0 if params.linear_fit else 0.0 for params in best_params])))),
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Poisson reconstruction parameters from point clouds.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Shared reconstruction config.")
    parser.add_argument("--inputs", nargs="*", type=Path, help="Training point clouds. Defaults to config training inputs.")
    parser.add_argument("--depth-values", help="Comma-separated depth candidates.")
    parser.add_argument("--scale-values", help="Comma-separated scale candidates.")
    parser.add_argument("--normals-radius-values", help="Comma-separated normal radius candidates.")
    parser.add_argument("--normals-max-nn-values", help="Comma-separated normal max_nn candidates.")
    parser.add_argument("--orient-max-nn-values", help="Comma-separated normal orientation max_nn candidates.")
    parser.add_argument("--density-quantile-values", help="Comma-separated density quantile candidates.")
    parser.add_argument("--linear-fit-values", help="Comma-separated booleans, e.g. false,true.")
    parser.add_argument("--sample-points", type=int, help="Number of mesh sample points used for scoring.")
    parser.add_argument("--max-eval-points", type=int, help="Maximum input points used for scoring.")
    parser.add_argument("--voxel-size", type=float, help="Voxel downsample size used only during parameter training.")
    parser.add_argument("--max-reconstruction-points", type=int, help="Maximum points used for each training reconstruction.")
    parser.add_argument("--component-penalty", type=float, help="Penalty for each extra connected component.")
    parser.add_argument("--f-score-threshold", type=float, help="Normalized distance threshold for precision/recall/F-score.")
    parser.add_argument("--f-score-weight", type=float, help="Weight of (1 - F-score) in the final minimized score.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Training result JSON path.")
    parser.add_argument("--update-config", action="store_true", help="Write recommended params back to reconstruction_config.json.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for point downsampling.")
    return parser.parse_args(argv)


def parse_bool_list(value: str | list[bool]) -> list[bool]:
    if isinstance(value, list):
        return [bool(item) for item in value]
    parsed: list[bool] = []
    for item in value.split(","):
        token = item.strip().lower()
        if token in {"1", "true", "yes"}:
            parsed.append(True)
        elif token in {"0", "false", "no"}:
            parsed.append(False)
        elif token:
            raise ValueError(f"Invalid boolean candidate: {item}")
    return parsed or [False]


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_training_config(args.config)
    if not args.inputs:
        config_inputs = config.get("inputs")
        if isinstance(config_inputs, list):
            args.inputs = [Path(str(path)) for path in config_inputs]
        else:
            args.inputs = default_inputs_from_config(args.config)
    if not args.inputs:
        raise ValueError("No training inputs provided. Use --inputs or configure training.poisson.inputs.")

    args.depth_values = parse_int_list(args.depth_values or config.get("depth_values", [7, 8, 9]))
    args.scale_values = parse_float_list(args.scale_values or config.get("scale_values", [1.05, 1.1]))
    args.normals_radius_values = parse_float_list(args.normals_radius_values or config.get("normals_radius_values", [0.02, 0.05]))
    args.normals_max_nn_values = parse_int_list(args.normals_max_nn_values or config.get("normals_max_nn_values", [30, 50]))
    args.orient_max_nn_values = parse_int_list(args.orient_max_nn_values or config.get("orient_max_nn_values", args.normals_max_nn_values))
    args.density_quantile_values = parse_float_list(args.density_quantile_values or config.get("density_quantile_values", [0.0, 0.02, 0.05]))
    args.linear_fit_values = parse_bool_list(args.linear_fit_values or config.get("linear_fit_values", [False]))

    if args.sample_points is None:
        args.sample_points = int(config.get("sample_points", 4000))
    if args.max_eval_points is None:
        args.max_eval_points = int(config.get("max_eval_points", 5000))
    if args.voxel_size is None:
        args.voxel_size = float(config.get("voxel_size", 0.05))
    if args.max_reconstruction_points is None:
        args.max_reconstruction_points = int(config.get("max_reconstruction_points", 20000))
    if args.component_penalty is None:
        args.component_penalty = float(config.get("component_penalty", 0.03))
    if args.f_score_threshold is None:
        args.f_score_threshold = float(config.get("f_score_threshold", 0.01))
    if args.f_score_weight is None:
        args.f_score_weight = float(config.get("f_score_weight", 0.05))

    if args.sample_points <= 0:
        raise ValueError("--sample-points must be positive.")
    if args.max_eval_points <= 0:
        raise ValueError("--max-eval-points must be positive.")
    if args.voxel_size < 0:
        raise ValueError("--voxel-size must be non-negative.")
    if args.max_reconstruction_points < 0:
        raise ValueError("--max-reconstruction-points must be non-negative.")
    if args.component_penalty < 0:
        raise ValueError("--component-penalty must be non-negative.")
    if args.f_score_threshold <= 0:
        raise ValueError("--f-score-threshold must be positive.")
    if args.f_score_weight < 0:
        raise ValueError("--f-score-weight must be non-negative.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if o3d is None:
            raise RuntimeError("Poisson training requires open3d.")
        args = fill_defaults(args)
        candidates = build_candidates(args)

        cases = [
            tune_one_cloud(
                input_path=input_path,
                candidates=candidates,
                sample_points=args.sample_points,
                max_eval_points=args.max_eval_points,
                voxel_size=args.voxel_size,
                max_reconstruction_points=args.max_reconstruction_points,
                component_penalty=args.component_penalty,
                f_score_threshold=args.f_score_threshold,
                f_score_weight=args.f_score_weight,
                seed=args.seed + index,
            )
            for index, input_path in enumerate(args.inputs)
        ]
        recommended = average_best_params(cases)

        result = PoissonTrainingResult(
            method="poisson",
            strategy="grid_search_average_best_params",
            recommended_params=recommended,
            inputs=[str(path) for path in args.inputs],
            sample_points=args.sample_points,
            max_eval_points=args.max_eval_points,
            voxel_size=args.voxel_size,
            max_reconstruction_points=args.max_reconstruction_points,
            component_penalty=args.component_penalty,
            f_score_threshold=args.f_score_threshold,
            f_score_weight=args.f_score_weight,
            cases=cases,
        )
        write_json(args.output, result)

        if args.update_config:
            update_poisson_config(args.config, recommended)

        print(f"trained Poisson from {len(cases)} point cloud(s), candidates={len(candidates)}")
        for case in cases:
            print(f"{case.input}: best_params={case.best_params}, score={case.best_score:.6g}")
        print(f"recommended_params={recommended}")
        print(f"wrote training result: {args.output}")
        if args.update_config:
            print(f"updated config methods.poisson in {args.config}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
