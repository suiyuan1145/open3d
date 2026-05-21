r"""WebSocket 文件流发送客户端。

示例：
    .\.venv\Scripts\python.exe streamout\send_file_websocket.py input\object.xyz
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

try:
    import websockets
except ImportError as exc:  # pragma: no cover - 运行时依赖提示
    raise SystemExit(
        "缺少依赖 websockets，请先运行: .\\.venv\\Scripts\\python.exe -m pip install websockets"
    ) from exc


STREAM_ID = 1
DEFAULT_URL = "ws://127.0.0.1:8765"
DEFAULT_CHUNK_SIZE = 1024 * 1024


async def send_file(url: str, file_path: Path, chunk_size: int) -> None:
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
        print(await websocket.recv())

        sent = 0
        with file_path.open("rb") as source:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                await websocket.send(chunk)
                sent += len(chunk)
                print(f"已发送 {sent}/{size} bytes")

        await websocket.send(json.dumps({"type": "end", "stream_id": STREAM_ID}, ensure_ascii=False))
        while True:
            response = json.loads(await websocket.recv())
            print(json.dumps(response, ensure_ascii=False))
            if response.get("type") in {"done", "error"}:
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a file over WebSocket stream 1.")
    parser.add_argument("file", type=Path, help="File to send.")
    parser.add_argument("--url", default=DEFAULT_URL, help="WebSocket server URL.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Binary chunk size in bytes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.file.is_file():
        raise SystemExit(f"文件不存在: {args.file}")
    asyncio.run(send_file(args.url, args.file, args.chunk_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
