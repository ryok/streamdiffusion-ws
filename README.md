# streamdiffusion-ws

*[日本語版はこちら / Japanese version](README.ja.md)*

A minimal WebSocket bridge that keeps [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
img2img resident on a GPU box. Send a base64 JPEG frame, get a stylized frame back,
near real time.

Any client that speaks WebSocket can drive it — TouchDesigner, Unity, Max/MSP,
Processing, a web app. Think of it as a client-agnostic, roll-your-own alternative to
[StreamDiffusionTD](https://github.com/DotSimulate/StreamDiffusion-TD)
(which requires NVIDIA + TensorRT and a TouchDesigner-side operator).

## How it works

- The model (sd-turbo, img2img) is loaded onto the GPU **once at startup** and stays resident
- Each WebSocket message is one frame: receive → img2img → send back
- The prompt and the style strength can be swapped **live**, without reloading the model

## Wire protocol

One WebSocket **text** message = one frame.

| Client → Server | Server → Client |
|---|---|
| base64-encoded JPEG (square, matching `--size`) | base64-encoded JPEG of the result |
| `PROMPT:<new prompt>` | `OK:prompt updated` |
| `TINDEX:<a>,<b>` — change style strength live | `OK:tindex [a, b]` / `ERR:...` |

Notes:

- Text (base64), not binary frames. `max_size` is 8 MB.
- **One request in flight at a time.** Wait for the reply before sending the next frame —
  the server processes sequentially, and queueing more only adds latency.
- The first frame takes a few seconds (CUDA kernel JIT); after that it is steady.
- `TINDEX:` replaces the `--t-index` values at runtime. **Lower values = the AI
  reinterprets the input more aggressively.** The list length must match startup
  (2 by default, e.g. `TINDEX:16,28`) — internally it swaps `t_list` and re-runs
  `prepare()`, so there is no model reload.

## Setup (on the GPU machine)

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/pip install -r <this repo>/requirements.txt
cp <this repo>/sd_ws_server.py .   # must sit next to StreamDiffusion so `utils.wrapper` imports
```

Dependency landmines worth pinning: `numpy<2` (torch is not 2.x-ready),
`huggingface_hub==0.24.6` (diffusers 0.24 still calls `cached_download`),
and `setuptools` for xformers.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python -u sd_ws_server.py \
  --size 512 --acceleration xformers --t-index 22 32 \
  --prompt "flowing ink wash painting, glowing smoke, elegant" --port 8765
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `stabilityai/sd-turbo` | img2img model |
| `--size` | 512 | square input/output size |
| `--t-index` | `22 32` | denoising strength. **Lower = stronger AI reinterpretation** (shape drifts); higher = faithful to the input |
| `--acceleration` | `xformers` | `none` / `xformers` / `tensorrt` |
| `--prompt` | see above | initial prompt |
| `--port` | 8765 | listen port |

> ⚠️ **Build the model outside `asyncio.run()`.** Running warmup inside the event loop
> made the process die during warmup with no traceback at all — CUDA stream sync
> interfering with the loop. This implementation calls `build_stream()` before
> `asyncio.run()`, and the silent crash disappeared completely.

## Keep it resident with tmux

```bash
ssh gpu-host 'tmux new-session -d -s sd "cd ~/StreamDiffusion && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u sd_ws_server.py \
  --size 512 --acceleration xformers --t-index 22 32 --port 8765 > ~/srv.log 2>&1"'
```

> ⚠️ **Never use `pkill -f sd_ws_server.py`.** Over SSH, `-f` matches your own shell
> (`bash -c '...sd_ws_server.py...'`) and kills the session itself — you get exit 255
> with zero output, indistinguishable from a network failure. Stop it with
> `tmux kill-session -t sd`.

## Smoke test

```bash
python3 test_client.py   # sends 3 dummy frames, saves client_out.jpg
```

## Using a remote GPU

From a local client (e.g. a Mac), tunnel the port:

```bash
ssh -N -L 8765:localhost:8765 gpu-host
```

## Measured latency (512px, t_index=[22,32], xformers, RTX 6000 Ada)

- First frame: ~7.3 s (CUDA kernel JIT)
- Steady state: ~55 ms of GPU inference; ~0.3 s/frame (3–4 fps) round trip including
  the SSH tunnel and JPEG/base64 encoding

## Used by

- [hand-trail-ai](https://github.com/ryok/hand-trail-ai) — repaints a MediaPipe hand
  trail in TouchDesigner into a dragon, a phoenix, whatever you prompt

## Write-up

[手の軌跡をAIで塗り替える：TouchDesignerでリアルタイム Hand Tracking × AI Trails を作る](https://zenn.dev/ryok/articles/touchdesigner-hand-trail-ai)
(Japanese) — why I built this instead of paying for a hosted service, and the pitfalls
along the way.

## License

MIT
