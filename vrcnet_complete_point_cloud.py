"""Complete a single point cloud with the pretrained VRCNet model.

This script adapts the original VRCNet test path, which expects MVP H5 files,
to a single local .pcd/.ply/.xyz/.txt/.csv/.npy point cloud.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from reconstruction_common import o3d, read_point_cloud


DEFAULT_MODEL = Path("VRCNet") / "pretrained" / "extracted" / "pretrained_vrcnet_2048.pth"
DEFAULT_CONFIG = Path("VRCNet") / "cfgs" / "vrcnet.yaml"


def farthest_sample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(points) <= count:
        if len(points) == count:
            return points
        rng = np.random.default_rng(seed)
        extra = rng.choice(len(points), size=count - len(points), replace=True)
        return np.vstack([points, points[extra]])

    rng = np.random.default_rng(seed)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(0, len(points)))
    distances = np.full(len(points), np.inf, dtype=np.float64)
    for i in range(1, count):
        current = points[selected[i - 1]]
        distances = np.minimum(distances, np.sum((points - current) ** 2, axis=1))
        selected[i] = int(np.argmax(distances))
    return points[selected]


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.mean(points, axis=0)
    shifted = points - center
    scale = float(np.max(np.linalg.norm(shifted, axis=1)))
    if scale <= 1e-12:
        raise ValueError("Point cloud has near-zero spatial extent.")
    return shifted / scale, center, scale


def write_point_cloud(path: Path, points: np.ndarray) -> None:
    if o3d is None:
        raise RuntimeError("open3d is required to write point clouds.")
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Failed to write point cloud: {path}")


def load_vrcnet_args(config_path: Path, model_path: Path, num_points: int) -> SimpleNamespace:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data["load_model"] = str(model_path)
    data["num_points"] = num_points
    return SimpleNamespace(**data)


def complete_with_vrcnet(points: np.ndarray, args: SimpleNamespace, vrcnet_dir: Path) -> np.ndarray:
    if not torch.cuda.is_available():
        raise RuntimeError("VRCNet pretrained inference requires CUDA because its PointNet++/EMD operators are CUDA based.")

    sys.path.insert(0, str(vrcnet_dir.resolve()))
    model_module = importlib.import_module(f"models.{args.model_name}")
    net = torch.nn.DataParallel(model_module.Model(args)).cuda()
    net.module.skip_metrics = True
    checkpoint = torch.load(args.load_model)
    net.module.load_state_dict(checkpoint["net_state_dict"])
    net.eval()

    tensor = torch.from_numpy(points.astype(np.float32)).unsqueeze(0).cuda()
    inputs = tensor.transpose(2, 1).contiguous()
    gt = tensor.contiguous()
    with torch.no_grad():
        result = net(inputs, gt, is_training=False)
    return result["out2"][0].detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete one point cloud with pretrained VRCNet.")
    parser.add_argument("--input", type=Path, required=True, help="Input point cloud path.")
    parser.add_argument("--output", type=Path, required=True, help="Completed point cloud output .ply path.")
    parser.add_argument("--partial-out", type=Path, help="Optional sampled partial cloud .ply path before completion.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Pretrained VRCNet .pth path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="VRCNet config YAML path.")
    parser.add_argument("--vrcnet-dir", type=Path, default=Path("VRCNet"), help="VRCNet repository folder.")
    parser.add_argument("--num-points", type=int, default=2048, help="Input points expected by pretrained model.")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Optional downsample before model sampling.")
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_points <= 0:
        raise ValueError("--num-points must be positive.")
    if args.voxel_size < 0:
        raise ValueError("--voxel-size must be non-negative.")
    if not args.model.exists():
        raise FileNotFoundError(f"VRCNet model does not exist: {args.model}")

    cloud = read_point_cloud(args.input)
    original_points = np.asarray(cloud.points)
    if args.voxel_size > 0:
        cloud = cloud.voxel_down_sample(args.voxel_size)
    points = np.asarray(cloud.points)
    sampled = farthest_sample(points, args.num_points, args.seed)
    normalized, center, scale = normalize_points(sampled)

    vrc_args = load_vrcnet_args(args.config, args.model, args.num_points)
    completed_normalized = complete_with_vrcnet(normalized, vrc_args, args.vrcnet_dir)
    completed = completed_normalized * scale + center

    if args.partial_out:
        write_point_cloud(args.partial_out, sampled)
    write_point_cloud(args.output, completed)

    print(f"input points: {len(original_points)}")
    print(f"model partial points: {len(sampled)}")
    print(f"completed points: {len(completed)}")
    if args.partial_out:
        print(f"wrote sampled partial: {args.partial_out}")
    print(f"wrote completed cloud: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
