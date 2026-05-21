"""WebSocket 文件流接收服务。

用途：
- 接收 stream_id=1 的文件流。
- 把二进制分片连续写入本地文件。
- 默认保存到 input/ 目录，适合后续交给点云重建脚本处理。

客户端发送协议：
1. 文本 JSON:
   {"type": "start", "stream_id": 1, "filename": "cloud.ply", "size": 12345}
2. 多个 binary message，每个 message 是文件的一段 bytes。
3. 文本 JSON:
   {"type": "end", "stream_id": 1}

服务器会返回 JSON ack/progress/done/error。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
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


STREAM_ID = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_SAVE_DIR = Path("input")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ReceiveState:
    stream_id: int
    filename: str
    path: Path
    expected_size: int | None
    received_size: int = 0
    started_at: float = 0.0


def safe_filename(name: str) -> str:
    """清理客户端传入的文件名，避免路径穿越。"""
    cleaned = SAFE_FILENAME_RE.sub("_", Path(name).name).strip("._")
    if not cleaned:
        return f"stream_{STREAM_ID}_{int(time.time())}.bin"
    return cleaned


def unique_path(save_dir: Path, filename: str) -> Path:
    """如果文件已存在，自动追加序号，避免覆盖已有输入。"""
    target = save_dir / filename
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    for index in range(1, 10000):
        candidate = save_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法为 {filename} 生成唯一文件名")


async def send_json(websocket: WebSocketServerProtocol, payload: dict[str, Any]) -> None:
    await websocket.send(json.dumps(payload, ensure_ascii=False))


def parse_json_message(message: str) -> dict[str, Any]:
    try:
        data = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 消息格式错误: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON 消息必须是对象")
    return data


async def handle_client(websocket: WebSocketServerProtocol, save_dir: Path) -> None:
    state: ReceiveState | None = None
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
                    path = unique_path(save_dir, filename)
                    output_file = path.open("wb")
                    state = ReceiveState(
                        stream_id=stream_id,
                        filename=filename,
                        path=path,
                        expected_size=expected_size,
                        started_at=time.time(),
                    )
                    await send_json(
                        websocket,
                        {
                            "type": "ack",
                            "stream_id": STREAM_ID,
                            "path": str(path),
                        },
                    )

                elif message_type == "end":
                    if state is None or output_file is None:
                        raise ValueError("收到 end，但当前没有正在接收的文件")

                    output_file.close()
                    output_file = None
                    elapsed = max(time.time() - state.started_at, 1e-6)
                    if state.expected_size is not None and state.received_size != state.expected_size:
                        await send_json(
                            websocket,
                            {
                                "type": "error",
                                "stream_id": STREAM_ID,
                                "message": "文件大小不匹配",
                                "expected_size": state.expected_size,
                                "received_size": state.received_size,
                                "path": str(state.path),
                            },
                        )
                    else:
                        await send_json(
                            websocket,
                            {
                                "type": "done",
                                "stream_id": STREAM_ID,
                                "path": str(state.path),
                                "bytes": state.received_size,
                                "seconds": round(elapsed, 3),
                                "mb_per_second": round(state.received_size / 1024 / 1024 / elapsed, 3),
                            },
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


async def run_server(host: str, port: int, save_dir: Path) -> None:
    async with websockets.serve(lambda ws: handle_client(ws, save_dir), host, port, max_size=None):
        print(f"WebSocket 文件流服务已启动: ws://{host}:{port}")
        print(f"接收 stream_id={STREAM_ID}，保存目录: {save_dir.resolve()}")
        await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive a file stream over WebSocket.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Listen host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Listen port.")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Directory for received files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(run_server(args.host, args.port, args.save_dir))
    except KeyboardInterrupt:
        print("WebSocket 文件流服务已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
