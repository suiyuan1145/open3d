r"""持续发送文件或 helloworld 的 WebSocket 客户端。

默认每 1 秒向 ws://192.168.1.131:8765 发送时间戳元信息和 book_pitch30_500k_poisson.ply。

示例：
    .\.venv\Scripts\python.exe streamout\send_helloworld_loop.py
    .\.venv\Scripts\python.exe streamout\send_helloworld_loop.py --interval 5
    .\.venv\Scripts\python.exe streamout\send_helloworld_loop.py --protocol raw-file
    .\.venv\Scripts\python.exe streamout\send_helloworld_loop.py --protocol text
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

try:
    import websockets
    from websockets.exceptions import ConnectionClosedOK
except ImportError as exc:  # pragma: no cover - 运行时依赖提示
    raise SystemExit(
        "缺少依赖 websockets，请先运行: .\\.venv\\Scripts\\python.exe -m pip install websockets"
    ) from exc


STREAM_ID = 1
DEFAULT_URL = "ws://192.168.1.131:8765"
DEFAULT_INTERVAL = 1.0
DEFAULT_MESSAGE = "helloworld"
DEFAULT_FILE = Path("model_outputs") / "book_pitch30_500k_poisson" / "ply" / "book_pitch30_500k_poisson.ply"
DEFAULT_CHUNK_SIZE = 1024 * 1024


def now_stamp() -> dict:
    sent_at_ns = time.time_ns()
    return {
        "sent_at_unix_ns": sent_at_ns,
        "sent_at_unix_ms": sent_at_ns // 1_000_000,
        "sent_at_perf_ns": time.perf_counter_ns(),
    }


async def send_text_once(url: str, message: str, wait_response: bool) -> dict:
    async with websockets.connect(url, max_size=None) as websocket:
        await websocket.send(message)
        if not wait_response:
            return {"type": "sent"}

        response = await websocket.recv()
        return {"type": "response", "message": response}


async def send_raw_file_once(
    url: str,
    file_path: Path,
    filename: str,
    sequence: int,
    chunk_size: int,
    include_timestamp: bool,
) -> dict:
    sent = 0
    stamp = now_stamp()
    async with websockets.connect(url, max_size=None) as websocket:
        try:
            if include_timestamp:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "file_timestamp",
                            "stream_id": STREAM_ID,
                            "filename": filename,
                            "size": file_path.stat().st_size,
                            "sequence": sequence,
                            **stamp,
                        },
                        ensure_ascii=False,
                    )
                )
            with file_path.open("rb") as source:
                while True:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    await websocket.send(chunk)
                    sent += len(chunk)
        except ConnectionClosedOK:
            return {"type": "closed_ok", "sent_bytes": sent, **stamp}
    return {"type": "sent", "sent_bytes": sent, **stamp}


async def recv_json(websocket) -> dict:
    message = await websocket.recv()
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return {"type": "message", "message": message}
    if isinstance(data, dict):
        return data
    return {"type": "message", "message": data}


async def send_file_stream_once(
    url: str,
    file_path: Path,
    filename: str,
    sequence: int,
    chunk_size: int,
    wait_ack: bool,
    wait_done: bool,
) -> dict:
    size = file_path.stat().st_size
    stamp = now_stamp()
    async with websockets.connect(url, max_size=None) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "stream_id": STREAM_ID,
                    "filename": filename,
                    "size": size,
                    "sequence": sequence,
                    **stamp,
                },
                ensure_ascii=False,
            )
        )

        if wait_ack:
            first_response = await recv_json(websocket)
            if first_response.get("type") == "error":
                return first_response

        sent = 0
        with file_path.open("rb") as source:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                await websocket.send(chunk)
                sent += len(chunk)

        await websocket.send(json.dumps({"type": "end", "stream_id": STREAM_ID}, ensure_ascii=False))

        if not wait_done:
            return {"type": "sent", "sent_bytes": sent, **stamp}

        while True:
            response = await recv_json(websocket)
            if response.get("type") in {"done", "error", "message"}:
                response.setdefault("sent_bytes", sent)
                return response


async def send_loop(
    url: str,
    message: str,
    interval: float,
    filename: str,
    protocol: str,
    wait_response: bool,
    wait_ack: bool,
    wait_done: bool,
    include_timestamp: bool,
    file_path: Path,
    chunk_size: int,
) -> None:
    payload = message.encode("utf-8")
    sequence = 1
    while True:
        try:
            if protocol == "text":
                response = await send_text_once(url, message, wait_response)
                sent_bytes = len(payload)
            else:
                if protocol == "raw-file":
                    response = await send_raw_file_once(
                        url,
                        file_path,
                        filename,
                        sequence,
                        chunk_size,
                        include_timestamp,
                    )
                else:
                    response = await send_file_stream_once(
                        url,
                        file_path,
                        filename,
                        sequence,
                        chunk_size,
                        wait_ack,
                        wait_done,
                    )
                sent_bytes = file_path.stat().st_size
            print(
                json.dumps(
                    {
                        "sequence": sequence,
                        "url": url,
                        "protocol": protocol,
                        "bytes": sent_bytes,
                        "logged_at_unix_ms": time.time_ns() // 1_000_000,
                        "response": response,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "sequence": sequence,
                        "url": url,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                )
            )

        sequence += 1
        await asyncio.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously send a model output file or helloworld over WebSocket.")
    parser.add_argument("--url", default=DEFAULT_URL, help="WebSocket receiver URL.")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Seconds between sends.")
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Text payload to send.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="File to send when --protocol file is used.")
    parser.add_argument("--filename", help="Filename sent in the start message. Defaults to the file basename.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="File chunk size in bytes.")
    parser.add_argument(
        "--protocol",
        choices=["text", "raw-file", "file-stream"],
        default="raw-file",
        help="file-stream sends start/binary/end messages; raw-file sends binary chunks directly; text sends helloworld.",
    )
    parser.add_argument("--wait-response", action="store_true", help="Wait for a server message after text send.")
    parser.add_argument("--wait-ack", action="store_true", help="Wait for ack after file-stream start message.")
    parser.add_argument("--wait-done", action="store_true", help="Wait for done/error after file-stream end message.")
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not send timestamp metadata before raw-file binary chunks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval < 0:
        raise SystemExit("--interval 不能为负数")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size 必须大于 0")
    if args.protocol in {"file-stream", "raw-file"} and not args.file.is_file():
        raise SystemExit(f"文件不存在: {args.file}")
    filename = Path(args.filename).name if args.filename else args.file.name
    asyncio.run(
        send_loop(
            args.url,
            args.message,
            args.interval,
            filename,
            args.protocol,
            args.wait_response,
            args.wait_ack,
            args.wait_done,
            not args.no_timestamp,
            args.file,
            args.chunk_size,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
