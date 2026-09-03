#!/usr/bin/env python3
"""
StreamDiffusion WebSocket server (準リアルタイム img2img)
- 起動時にStreamDiffusionを1回ロードしGPUに常駐
- WebSocketでbase64 JPEGフレームを受信 → img2img → base64 JPEGで返信
- TouchDesigner(Mac)からSSHトンネル経由で 127.0.0.1:8765 に接続する前提

依存: torch, streamdiffusion, diffusers, websockets, pillow
起動: CUDA_VISIBLE_DEVICES=0 python sd_ws_server.py --prompt "..."
"""
import argparse
import asyncio
import base64
import io
import time

import torch
from PIL import Image

# StreamDiffusion のラッパー(公式リポジトリの utils.wrapper)
from utils.wrapper import StreamDiffusionWrapper

import websockets


def build_stream(args):
    """StreamDiffusion を img2img モードで初期化してGPUに載せる"""
    stream = StreamDiffusionWrapper(
        model_id_or_path=args.model,
        t_index_list=args.t_index,       # img2imgのノイズ強度(小さいほどAI解釈が強い)
        frame_buffer_size=1,
        width=args.size,
        height=args.size,
        warmup=10,
        acceleration=args.acceleration,  # 'xformers' or 'tensorrt' or 'none'
        mode="img2img",
        use_denoising_batch=True,
        cfg_type="self",
        seed=args.seed,
        output_type="pil",               # 戻り値をPILにして後処理不要
    )
    stream.prepare(
        prompt=args.prompt,
        negative_prompt="low quality, blurry, distorted",
        num_inference_steps=50,
        guidance_scale=1.2,
    )
    return stream


def b64_to_pil(b64: str, size: int) -> Image.Image:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size))
    return img


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def handler(ws, stream, size):
    print(f"[ws] client connected: {ws.remote_address}")
    try:
        async for message in ws:
            t0 = time.time()
            # 受信: base64 JPEG (先頭に "PROMPT:" が来たらプロンプト更新)
            if message.startswith("PROMPT:"):
                new_prompt = message[len("PROMPT:"):]
                stream.prepare(prompt=new_prompt,
                               negative_prompt="low quality, blurry",
                               num_inference_steps=50, guidance_scale=1.2)
                await ws.send("OK:prompt updated")
                print(f"[ws] prompt -> {new_prompt}")
                continue

            img = b64_to_pil(message, size)
            # img2img 実行 (output_type='pil' なので戻り値はPIL画像)
            out = stream(image=img)
            out_img = out if isinstance(out, Image.Image) else out[0]
            await ws.send(pil_to_b64(out_img))
            dt = (time.time() - t0) * 1000
            print(f"[ws] frame processed {dt:.0f}ms ({1000/dt:.1f} fps)", end="\r")
    except websockets.ConnectionClosed:
        print("\n[ws] client disconnected")


async def main(args, stream):
    async with websockets.serve(lambda ws: handler(ws, stream, args.size),
                                args.host, args.port, max_size=8 * 1024 * 1024):
        print(f"[server] listening on ws://{args.host}:{args.port}  prompt='{args.prompt}'", flush=True)
        await asyncio.Future()  # run forever


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/sd-turbo")
    ap.add_argument("--prompt", default="flowing ink wash painting, glowing smoke, elegant")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--acceleration", default="xformers", choices=["none", "xformers", "tensorrt"])
    ap.add_argument("--t-index", type=int, nargs="+", default=[22, 32])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    return ap.parse_args()


if __name__ == "__main__":
    # モデル構築は asyncio の外(同期)で行う — イベントループ内warmupの干渉クラッシュを回避
    args = parse_args()
    print(f"[init] loading model={args.model} size={args.size} t_index={args.t_index}", flush=True)
    print(f"[init] CUDA: {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    stream = build_stream(args)
    print("[init] model ready (warmup done)", flush=True)
    asyncio.run(main(args, stream))
