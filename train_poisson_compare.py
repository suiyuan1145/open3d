"""Compare coarse-to-fine and Optuna tuning for Poisson reconstruction.

The script writes two independent training reports and two independent config
files. Use those config files to reconstruct two meshes, then compare them by
eye or in another tool.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import optuna

from train_poisson_parameters import (
    PoissonCandidateScore,
    PoissonParams,
    average_best_params,
    build_candidates,
    fill_defaults,
    parse_args as parse_base_args,
    tune_one_cloud,
)
from reconstruction_common import write_json


DEFAULT_COMPARE_DIR = Path("model_outputs") / "training" / "poisson_compare"


@dataclass
class CompareResult:
    strategy: str
    recommended_params: PoissonParams
    report_path: str
    config_path: str
    best_score: float


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def write_strategy_config(base_config_path: Path, output_path: Path, params: PoissonParams, output_name: str) -> None:
    config = load_config(base_config_path)
    methods = config.setdefault("methods", {})
    if not isinstance(methods, dict):
        raise ValueError("Config field 'methods' must be an object.")
    poisson = methods.setdefault("poisson", {})
    if not isinstance(poisson, dict):
        raise ValueError("Config field 'methods.poisson' must be an object.")

    poisson["output"] = output_name
    poisson["params_out"] = output_name.replace(".obj", "_params.json").replace(".ply", "_params.json")
    poisson["depth"] = params.depth
    poisson["scale"] = params.scale
    poisson["normals_radius"] = params.normals_radius
    poisson["normals_max_nn"] = params.normals_max_nn
    poisson["orient_max_nn"] = params.orient_max_nn
    poisson["density_quantile"] = params.density_quantile
    poisson["linear_fit"] = params.linear_fit

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def evaluate_params(base_args: argparse.Namespace, params: PoissonParams) -> tuple[float, list[PoissonCandidateScore]]:
    scores: list[PoissonCandidateScore] = []
    case_scores: list[float] = []
    for index, input_path in enumerate(base_args.inputs):
        case = tune_one_cloud(
            input_path=input_path,
            candidates=[params],
            sample_points=base_args.sample_points,
            max_eval_points=base_args.max_eval_points,
            voxel_size=base_args.voxel_size,
            max_reconstruction_points=base_args.max_reconstruction_points,
            component_penalty=base_args.component_penalty,
            f_score_threshold=base_args.f_score_threshold,
            f_score_weight=base_args.f_score_weight,
            seed=base_args.seed + index,
        )
        scores.extend(case.scores)
        case_scores.append(case.best_score)
    return float(np.mean(case_scores)), scores


def run_coarse_to_fine(base_args: argparse.Namespace) -> tuple[PoissonParams, dict[str, object]]:
    coarse_candidates = build_candidates(base_args)
    coarse_cases = [
        tune_one_cloud(
            input_path=input_path,
            candidates=coarse_candidates,
            sample_points=base_args.sample_points,
            max_eval_points=base_args.max_eval_points,
            voxel_size=base_args.voxel_size,
            max_reconstruction_points=base_args.max_reconstruction_points,
            component_penalty=base_args.component_penalty,
            f_score_threshold=base_args.f_score_threshold,
            f_score_weight=base_args.f_score_weight,
            seed=base_args.seed + index,
        )
        for index, input_path in enumerate(base_args.inputs)
    ]
    current = average_best_params(coarse_cases)

    history: list[dict[str, object]] = [
        {
            "stage": "coarse_grid",
            "recommended_params": asdict(current),
            "cases": [asdict(case) for case in coarse_cases],
        }
    ]

    for round_index in range(base_args.fine_rounds):
        param_sets: list[PoissonParams] = []
        for radius_factor in [0.75, 0.9, 1.0, 1.1, 1.25]:
            param_sets.append(PoissonParams(**{**asdict(current), "normals_radius": max(1e-6, current.normals_radius * radius_factor)}))
        for scale_delta in [-0.05, 0.0, 0.05]:
            param_sets.append(PoissonParams(**{**asdict(current), "scale": max(0.5, current.scale + scale_delta)}))
        for density_delta in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            param_sets.append(PoissonParams(**{**asdict(current), "density_quantile": min(0.95, max(0.0, current.density_quantile + density_delta))}))
        for nn_delta in [-20, 0, 20]:
            param_sets.append(PoissonParams(**{**asdict(current), "normals_max_nn": max(3, current.normals_max_nn + nn_delta)}))
        for nn_delta in [-30, 0, 30]:
            param_sets.append(PoissonParams(**{**asdict(current), "orient_max_nn": max(3, current.orient_max_nn + nn_delta)}))

        unique = {tuple(asdict(params).items()): params for params in param_sets}
        candidates = list(unique.values())
        scored: list[dict[str, object]] = []
        best_score = float("inf")
        best_params = current

        for candidate in candidates:
            score, _ = evaluate_params(base_args, candidate)
            scored.append({"params": asdict(candidate), "score": score})
            if score < best_score:
                best_score = score
                best_params = candidate

        current = best_params
        history.append(
            {
                "stage": f"fine_round_{round_index + 1}",
                "recommended_params": asdict(current),
                "candidates": scored,
            }
        )

    final_score, _ = evaluate_params(base_args, current)
    return current, {"strategy": "coarse_to_fine", "best_score": final_score, "history": history}


def run_optuna(base_args: argparse.Namespace) -> tuple[PoissonParams, dict[str, object]]:
    depth_min, depth_max = min(base_args.depth_values), max(base_args.depth_values)
    scale_min, scale_max = min(base_args.scale_values), max(base_args.scale_values)
    radius_min, radius_max = min(base_args.normals_radius_values), max(base_args.normals_radius_values)
    normal_nn_min, normal_nn_max = min(base_args.normals_max_nn_values), max(base_args.normals_max_nn_values)
    orient_nn_min, orient_nn_max = min(base_args.orient_max_nn_values), max(base_args.orient_max_nn_values)
    density_min, density_max = min(base_args.density_quantile_values), max(base_args.density_quantile_values)

    trial_records: list[dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        params = PoissonParams(
            depth=trial.suggest_int("depth", depth_min, depth_max),
            scale=trial.suggest_float("scale", scale_min, scale_max),
            normals_radius=trial.suggest_float("normals_radius", radius_min, radius_max),
            normals_max_nn=trial.suggest_int("normals_max_nn", normal_nn_min, normal_nn_max),
            orient_max_nn=trial.suggest_int("orient_max_nn", orient_nn_min, orient_nn_max),
            density_quantile=trial.suggest_float("density_quantile", density_min, density_max),
            linear_fit=trial.suggest_categorical("linear_fit", base_args.linear_fit_values),
        )
        score, _ = evaluate_params(base_args, params)
        trial_records.append({"number": trial.number, "params": asdict(params), "score": score})
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=base_args.seed))
    study.optimize(objective, n_trials=base_args.optuna_trials)

    best = study.best_params
    best_params = PoissonParams(
        depth=int(best["depth"]),
        scale=float(best["scale"]),
        normals_radius=float(best["normals_radius"]),
        normals_max_nn=int(best["normals_max_nn"]),
        orient_max_nn=int(best["orient_max_nn"]),
        density_quantile=float(best["density_quantile"]),
        linear_fit=bool(best["linear_fit"]),
    )
    return best_params, {"strategy": "optuna_tpe", "best_score": float(study.best_value), "trials": trial_records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare coarse-to-fine and Optuna Poisson parameter tuning.")
    parser.add_argument("--config", type=Path, default=Path("reconstruction_config.json"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    parser.add_argument("--fine-rounds", type=int, default=2)
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--base-args", nargs=argparse.REMAINDER, help="Arguments forwarded to train_poisson_parameters.py parser.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    forwarded = ["--config", str(args.config)]
    if args.base_args:
        forwarded.extend(args.base_args)
    base_args = fill_defaults(parse_base_args(forwarded))
    base_args.fine_rounds = args.fine_rounds
    base_args.optuna_trials = args.optuna_trials

    args.output_dir.mkdir(parents=True, exist_ok=True)

    coarse_params, coarse_report = run_coarse_to_fine(base_args)
    coarse_report_path = args.output_dir / "poisson_coarse_to_fine_report.json"
    coarse_config_path = args.output_dir / "poisson_coarse_to_fine_config.json"
    write_json(coarse_report_path, coarse_report)
    write_strategy_config(args.config, coarse_config_path, coarse_params, "poisson_coarse_to_fine.obj")

    optuna_params, optuna_report = run_optuna(base_args)
    optuna_report_path = args.output_dir / "poisson_optuna_report.json"
    optuna_config_path = args.output_dir / "poisson_optuna_config.json"
    write_json(optuna_report_path, optuna_report)
    write_strategy_config(args.config, optuna_config_path, optuna_params, "poisson_optuna.obj")

    summary = {
        "coarse_to_fine": asdict(
            CompareResult(
                strategy="coarse_to_fine",
                recommended_params=coarse_params,
                report_path=str(coarse_report_path),
                config_path=str(coarse_config_path),
                best_score=float(coarse_report["best_score"]),
            )
        ),
        "optuna": asdict(
            CompareResult(
                strategy="optuna",
                recommended_params=optuna_params,
                report_path=str(optuna_report_path),
                config_path=str(optuna_config_path),
                best_score=float(optuna_report["best_score"]),
            )
        ),
        "reconstruction_commands": [
            f".\\.venv\\Scripts\\python.exe poisson_reconstruction.py --config {coarse_config_path}",
            f".\\.venv\\Scripts\\python.exe poisson_reconstruction.py --config {optuna_config_path}",
        ],
    }
    summary_path = args.output_dir / "poisson_compare_summary.json"
    write_json(summary_path, summary)

    print(f"coarse_to_fine config: {coarse_config_path}")
    print(f"optuna config: {optuna_config_path}")
    print(f"summary: {summary_path}")
    print("生成两个网格的命令：")
    for command in summary["reconstruction_commands"]:
        print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
