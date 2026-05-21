r"""WebSocket 接收点云、执行三维重建、再转发输出文件。

默认流程：
1. 在 ws://127.0.0.1:8765 接收 stream_id=1 文件流。
2. 调用项目根目录的 poisson_reconstruction.py 生成 PLY 网格。
3. 将生成的 PLY 作为 stream_id=1 文件流发送到 ws://127.0.0.1:8766。

联调时可以加 --passthrough，跳过三维重建，直接转发收到的原始文件。

下一个端口可以直接运行：
    .\.venv\Scripts\python.exe streamin\file_stream_websocket.py --port 8766 --save-dir model_outputs\forwarded
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError as exc:  # pragma: no cover - 运行时依赖提示
    raise SystemExit(
        "缺少依赖 websockets，请先运行: .\\.venv\\Scripts\\python.exe -m pip install websockets"
    ) from exc

from file_stream_websocket import STREAM_ID, parse_json_message, safe_filename, unique_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 8765
DEFAULT_FORWARD_URL = "ws://127.0.0.1:8766"
DEFAULT_SAVE_DIR = REPO_ROOT / "input" / "stream_pipeline"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "model_outputs" / "stream_pipeline"
DEFAULT_CHUNK_SIZE = 1024 * 1024


@dataclass
class PipelineState:
    filename: str
    input_path: Path
    expected_size: int | None
    received_size: int = 0
    started_at: float = 0.0


async def send_json(websocket: WebSocketServerProtocol, payload: dict[str, Any]) -> None:
    await websocket.send(json.dumps(payload, ensure_ascii=False))


def build_output_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    mesh_path = unique_path(output_dir, f"{stem}_poisson.ply")
    params_path = mesh_path.with_suffix(".json")
    return mesh_path, params_path


def run_poisson_reconstruction(input_path: Path, output_path: Path, params_path: Path, extra_args: list[str]) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "poisson_reconstruction.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--params-out",
        str(params_path),
        "--flat-output",
        *extra_args,
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Poisson 重建失败: {detail}")
    if not output_path.exists():
        raise RuntimeError(f"Poisson 重建没有生成输出文件: {output_path}")


async def forward_file(url: str, file_path: Path, chunk_size: int) -> dict[str, Any]:
    size = file_path.stat().st_size
    async with websockets.connect(url, max_size=None) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "stream_id": STREAM_ID,
                    "filename": file_path.name,
                    "size": size,
                },
                ensure_ascii=False,
            )
        )

        first_response = json.loads(await websocket.recv())
        if first_response.get("type") == "error":
            raise RuntimeError(f"下一个端口拒绝接收: {first_response.get('message')}")

        with file_path.open("rb") as source:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                await websocket.send(chunk)

        await websocket.send(json.dumps({"type": "end", "stream_id": STREAM_ID}, ensure_ascii=False))
        while True:
            response = json.loads(await websocket.recv())
            if response.get("type") in {"done", "error"}:
                if response.get("type") == "error":
                    raise RuntimeError(f"下一个端口接收失败: {response.get('message')}")
                return response


async def finish_pipeline(
    websocket: WebSocketServerProtocol,
    state: PipelineState,
    output_dir: Path,
    forward_url: str,
    chunk_size: int,
    poisson_args: list[str],
    passthrough: bool,
) -> None:
    if passthrough:
        await send_json(
            websocket,
            {
                "type": "forwarding",
                "stream_id": STREAM_ID,
                "mode": "passthrough",
                "url": forward_url,
                "path": str(state.input_path),
            },
        )
        forwarded = await forward_file(forward_url, state.input_path, chunk_size)
        await send_json(
            websocket,
            {
                "type": "pipeline_done",
                "stream_id": STREAM_ID,
                "mode": "passthrough",
                "input_path": str(state.input_path),
                "output_path": str(state.input_path),
                "forwarded": forwarded,
            },
        )
        return

    mesh_path, params_path = build_output_paths(state.input_path, output_dir)
    await send_json(
        websocket,
        {
            "type": "reconstructing",
            "stream_id": STREAM_ID,
            "input_path": str(state.input_path),
            "output_path": str(mesh_path),
        },
    )

    await asyncio.to_thread(run_poisson_reconstruction, state.input_path, mesh_path, params_path, poisson_args)
    await send_json(
        websocket,
        {
            "type": "reconstructed",
            "stream_id": STREAM_ID,
            "output_path": str(mesh_path),
            "params_path": str(params_path),
            "bytes": mesh_path.stat().st_size,
        },
    )

    await send_json(
        websocket,
        {
            "type": "forwarding",
            "stream_id": STREAM_ID,
            "url": forward_url,
            "path": str(mesh_path),
        },
    )
    forwarded = await forward_file(forward_url, mesh_path, chunk_size)
    await send_json(
        websocket,
        {
            "type": "pipeline_done",
            "stream_id": STREAM_ID,
            "input_path": str(state.input_path),
            "output_path": str(mesh_path),
            "params_path": str(params_path),
            "forwarded": forwarded,
        },
    )


async def handle_pipeline_client(
    websocket: WebSocketServerProtocol,
    save_dir: Path,
    output_dir: Path,
    forward_url: str,
    chunk_size: int,
    poisson_args: list[str],
    passthrough: bool,
) -> None:
    state: PipelineState | None = None
    output_file = None

    try:
        async for message in websocket:
            if isinstance(message, str):
                data = parse_json_message(message)
                message_type = data.get("type")

                if message_type == "start":
                    if state is not None:
                        raise ValueError("当前连接已有正在接收的文件")

                    stream_id = int(data.get("stream_id", STREAM_ID))
                    if stream_id != STREAM_ID:
                        raise ValueError(f"只支持 stream_id={STREAM_ID}，收到 {stream_id}")

                    filename = safe_filename(str(data.get("filename") or "stream_1.bin"))
                    expected_size = data.get("size")
                    if expected_size is not None:
                        expected_size = int(expected_size)
                        if expected_size < 0:
                            raise ValueError("size 不能为负数")

                    save_dir.mkdir(parents=True, exist_ok=True)
                    input_path = unique_path(save_dir, filename)
                    output_file = input_path.open("wb")
                    state = PipelineState(
                        filename=filename,
                        input_path=input_path,
                        expected_size=expected_size,
                        started_at=time.time(),
                    )
                    await send_json(
                        websocket,
                        {
                            "type": "ack",
                            "stream_id": STREAM_ID,
                            "path": str(input_path),
                        },
                    )

                elif message_type == "end":
                    if state is None or output_file is None:
                        raise ValueError("收到 end，但当前没有正在接收的文件")

                    output_file.close()
                    output_file = None
                    if state.expected_size is not None and state.received_size != state.expected_size:
                        raise ValueError(
                            f"文件大小不匹配: expected={state.expected_size}, received={state.received_size}"
                        )

                    await send_json(
                        websocket,
                        {
                            "type": "received",
                            "stream_id": STREAM_ID,
                            "path": str(state.input_path),
                            "bytes": state.received_size,
                        },
                    )
                    await finish_pipeline(
                        websocket,
                        state,
                        output_dir,
                        forward_url,
                        chunk_size,
                        poisson_args,
                        passthrough,
                    )
                    state = None

                elif message_type == "ping":
                    await send_json(websocket, {"type": "pong", "stream_id": STREAM_ID})

                else:
                    raise ValueError(f"未知文本消息类型: {message_type}")

            else:
                if state is None or output_file is None:
                    raise ValueError("收到二进制数据，但尚未收到 start 消息")

                output_file.write(message)
                state.received_size += len(message)
                if state.received_size == len(message) or state.received_size % (8 * 1024 * 1024) < len(message):
                    await send_json(
                        websocket,
                        {
                            "type": "progress",
                            "stream_id": STREAM_ID,
                            "bytes": state.received_size,
                            "expected_size": state.expected_size,
                        },
                    )

    except Exception as exc:
        if output_file is not None:
            output_file.close()
        await send_json(websocket, {"type": "error", "stream_id": STREAM_ID, "message": str(exc)})


async def run_server(args: argparse.Namespace) -> None:
    async with websockets.serve(
        lambda ws: handle_pipeline_client(
            ws,
            args.save_dir,
            args.output_dir,
            args.forward_url,
            args.chunk_size,
            args.poisson_arg,
            args.passthrough,
        ),
        args.host,
        args.port,
        max_size=None,
    ):
        print(f"重建转发服务已启动: ws://{args.host}:{args.port}")
        print(f"输入保存目录: {args.save_dir.resolve()}")
        print(f"重建输出目录: {args.output_dir.resolve()}")
        print(f"转发到: {args.forward_url}")
        if args.passthrough:
            print("当前模式: passthrough，跳过三维重建，直接转发原始文件")
        else:
            print("当前模式: Poisson 三维重建后转发")
        await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive a point-cloud file, reconstruct it, then forward the mesh.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Listen host.")
    parser.add_argument("--port", type=int, default=DEFAULT_LISTEN_PORT, help="Listen port.")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Directory for received inputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for reconstructed meshes.")
    parser.add_argument("--forward-url", default=DEFAULT_FORWARD_URL, help="Downstream WebSocket URL.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Forward chunk size in bytes.")
    parser.add_argument("--passthrough", action="store_true", help="Forward the received file directly without reconstruction.")
    parser.add_argument(
        "--poisson-arg",
        action="append",
        default=[],
        help="Extra argument passed to poisson_reconstruction.py. Repeat for multiple tokens.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run_server(args))
    except KeyboardInterrupt:
        print("重建转发服务已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
