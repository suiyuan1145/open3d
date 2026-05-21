"""Tune Alpha Shape alpha values from several point clouds.

This is parameter optimization rather than supervised neural-network training:
without ground-truth meshes, the script searches candidate alpha values and
selects the value that best reconstructs each input point cloud under a
self-supervised Chamfer-style score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from reconstruction_common import o3d, read_point_cloud, write_json


DEFAULT_CONFIG_PATH = Path("reconstruction_config.json")
DEFAULT_OUTPUT_PATH = Path("model_outputs") / "training" / "alpha_shape_training.json"


@dataclass
class AlphaCandidateScore:
    alpha: float
    score: float
    chamfer: float
    component_count: int
    vertex_count: int
    triangle_count: int


@dataclass
class AlphaTrainingCase:
    input: str
    best_alpha: float
    best_score: float
    scores: list[AlphaCandidateScore]


@dataclass
class AlphaTrainingResult:
    method: str
    strategy: str
    recommended_alpha: float
    inputs: list[str]
    alpha_candidates: list[float]
    sample_points: int
    max_eval_points: int
    component_penalty: float
    cases: list[AlphaTrainingCase]


def parse_float_list(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        values = [float(item) for item in value]
    else:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("alpha candidates must be positive numbers.")
    return sorted(set(values))


def load_training_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    training = config.get("training", {})
    if not isinstance(training, dict):
        return {}
    alpha_config = training.get("alpha_shape", {})
    return alpha_config if isinstance(alpha_config, dict) else {}


def default_inputs_from_config(config_path: Path) -> list[Path]:
    if not config_path.exists():
        return []
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    alpha_config = config.get("methods", {}).get("alpha_shape", {})
    input_path = alpha_config.get("input") if isinstance(alpha_config, dict) else None
    return [Path(str(input_path))] if input_path else []


def update_alpha_config(config_path: Path, alpha: float) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    methods = config.setdefault("methods", {})
    if not isinstance(methods, dict):
        raise ValueError("Config field 'methods' must be an object.")
    alpha_shape = methods.setdefault("alpha_shape", {})
    if not isinstance(alpha_shape, dict):
        raise ValueError("Config field 'methods.alpha_shape' must be an object.")
    alpha_shape["alpha"] = alpha

    training = config.setdefault("training", {})
    if isinstance(training, dict):
        alpha_training = training.setdefault("alpha_shape", {})
        if isinstance(alpha_training, dict):
            alpha_training["recommended_alpha"] = alpha

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def downsample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def mesh_component_count(mesh: "o3d.geometry.TriangleMesh") -> int:
    if len(mesh.triangles) == 0:
        return 0
    labels, _, _ = mesh.cluster_connected_triangles()
    label_array = np.asarray(labels)
    return int(label_array.max() + 1) if len(label_array) else 0


def reconstruct_alpha_mesh(cloud: "o3d.geometry.PointCloud", alpha: float) -> "o3d.geometry.TriangleMesh":
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    return mesh


def score_mesh(
    points: np.ndarray,
    mesh: "o3d.geometry.TriangleMesh",
    sample_points: int,
    component_penalty: float,
    seed: int,
) -> AlphaCandidateScore:
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return AlphaCandidateScore(
            alpha=math.nan,
            score=float("inf"),
            chamfer=float("inf"),
            component_count=0,
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.triangles),
        )

    sampled = mesh.sample_points_uniformly(number_of_points=sample_points)
    mesh_points = np.asarray(sampled.points)
    if len(mesh_points) == 0:
        return AlphaCandidateScore(
            alpha=math.nan,
            score=float("inf"),
            chamfer=float("inf"),
            component_count=0,
            vertex_count=len(mesh.vertices),
            triangle_count=len(mesh.triangles),
        )

    point_tree = cKDTree(points)
    mesh_tree = cKDTree(mesh_points)
    point_to_mesh, _ = mesh_tree.query(points, k=1)
    mesh_to_point, _ = point_tree.query(mesh_points, k=1)

    diagonal = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    scale = max(diagonal, 1e-9)
    chamfer = (float(np.mean(point_to_mesh)) + float(np.mean(mesh_to_point))) / scale

    components = mesh_component_count(mesh)
    score = chamfer + max(0, components - 1) * component_penalty
    return AlphaCandidateScore(
        alpha=math.nan,
        score=score,
        chamfer=chamfer,
        component_count=components,
        vertex_count=len(mesh.vertices),
        triangle_count=len(mesh.triangles),
    )


def tune_one_cloud(
    input_path: Path,
    alpha_candidates: list[float],
    sample_points: int,
    max_eval_points: int,
    component_penalty: float,
    seed: int,
) -> AlphaTrainingCase:
    cloud = read_point_cloud(input_path)
    points = downsample_points(np.asarray(cloud.points), max_eval_points, seed)

    scores: list[AlphaCandidateScore] = []
    for alpha in alpha_candidates:
        mesh = reconstruct_alpha_mesh(cloud, alpha)
        candidate_score = score_mesh(points, mesh, sample_points, component_penalty, seed)
        candidate_score.alpha = alpha
        scores.append(candidate_score)

    finite_scores = [item for item in scores if math.isfinite(item.score)]
    if not finite_scores:
        raise RuntimeError(f"No valid alpha candidate produced a mesh for {input_path}.")

    best = min(finite_scores, key=lambda item: item.score)
    return AlphaTrainingCase(
        input=str(input_path),
        best_alpha=best.alpha,
        best_score=best.score,
        scores=scores,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Alpha Shape alpha from several point clouds.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Shared reconstruction config.")
    parser.add_argument("--inputs", nargs="*", type=Path, help="Training point clouds. Defaults to config training inputs.")
    parser.add_argument("--alpha-values", help="Comma-separated alpha candidates.")
    parser.add_argument("--sample-points", type=int, help="Number of mesh sample points used for scoring.")
    parser.add_argument("--max-eval-points", type=int, help="Maximum input points used for scoring.")
    parser.add_argument("--component-penalty", type=float, help="Penalty for each extra connected component.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Training result JSON path.")
    parser.add_argument("--update-config", action="store_true", help="Write recommended alpha back to reconstruction_config.json.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for point downsampling.")
    return parser.parse_args(argv)


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_training_config(args.config)
    if not args.inputs:
        config_inputs = config.get("inputs")
        if isinstance(config_inputs, list):
            args.inputs = [Path(str(path)) for path in config_inputs]
        else:
            args.inputs = default_inputs_from_config(args.config)
    if not args.inputs:
        raise ValueError("No training inputs provided. Use --inputs or configure training.alpha_shape.inputs.")

    if args.alpha_values is None:
        configured = config.get("alpha_candidates")
        args.alpha_values = configured if isinstance(configured, list) else [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07]
    args.alpha_values = parse_float_list(args.alpha_values)

    if args.sample_points is None:
        args.sample_points = int(config.get("sample_points", 4000))
    if args.max_eval_points is None:
        args.max_eval_points = int(config.get("max_eval_points", 5000))
    if args.component_penalty is None:
        args.component_penalty = float(config.get("component_penalty", 0.03))

    if args.sample_points <= 0:
        raise ValueError("--sample-points must be positive.")
    if args.max_eval_points <= 0:
        raise ValueError("--max-eval-points must be positive.")
    if args.component_penalty < 0:
        raise ValueError("--component-penalty must be non-negative.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if o3d is None:
            raise RuntimeError("Alpha training requires open3d.")
        args = fill_defaults(args)

        cases = [
            tune_one_cloud(
                input_path=input_path,
                alpha_candidates=args.alpha_values,
                sample_points=args.sample_points,
                max_eval_points=args.max_eval_points,
                component_penalty=args.component_penalty,
                seed=args.seed + index,
            )
            for index, input_path in enumerate(args.inputs)
        ]
        recommended_alpha = float(np.mean([case.best_alpha for case in cases]))

        result = AlphaTrainingResult(
            method="alpha_shape",
            strategy="grid_search_average_best_alpha",
            recommended_alpha=recommended_alpha,
            inputs=[str(path) for path in args.inputs],
            alpha_candidates=args.alpha_values,
            sample_points=args.sample_points,
            max_eval_points=args.max_eval_points,
            component_penalty=args.component_penalty,
            cases=cases,
        )
        write_json(args.output, result)

        if args.update_config:
            update_alpha_config(args.config, recommended_alpha)

        print(f"trained alpha from {len(cases)} point cloud(s)")
        for case in cases:
            print(f"{case.input}: best_alpha={case.best_alpha:.8g}, score={case.best_score:.6g}")
        print(f"recommended_alpha={recommended_alpha:.8g}")
        print(f"wrote training result: {args.output}")
        if args.update_config:
            print(f"updated config alpha_shape.alpha in {args.config}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
