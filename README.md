# streamdiffusion-ws

[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) の img2img を
**WebSocket サーバー**として常駐させ、任意のクライアントから base64 JPEG フレームを
送って準リアルタイムに変換して返す最小ブリッジ。

TouchDesigner・Unity・Max/MSP・Processing・Web など、WebSocket を喋れれば
どこからでも使える（[StreamDiffusionTD](https://github.com/DotSimulate/StreamDiffusion-TD) の
自作・クライアント非依存版という位置づけ）。

## しくみ

- 起動時に StreamDiffusion(sd-turbo, img2img) を1回だけGPUに載せて常駐
- WebSocket でフレームを受信 → img2img → 返信
- プロンプトは実行中にライブ差し替え可能

## ワイヤープロトコル

WebSocket テキストメッセージ1本 = 1フレーム。

| 送信(クライアント→サーバー) | 返信(サーバー→クライアント) |
|---|---|
| base64 エンコードした JPEG（`--size` 正方形推奨） | 変換後の base64 JPEG |
| `PROMPT:<新しいプロンプト>` | `OK:prompt updated` |
| `TINDEX:<a>,<b>` スタイル強度をライブ変更 | `OK:tindex [a, b]` / `ERR:...` |

`TINDEX:` は `--t-index` の値を実行中に差し替える（小さいほどAI解釈が強い）。
**要素数は起動時と同じ**でなければならない（既定は2個。例 `TINDEX:16,28`）。
内部で `t_list` を書き換えて `prepare()` し直すだけなのでモデル再ロードは無く軽い。

- バイナリではなく **base64 テキスト**（`max_size` 8MB）
- 1フレームずつのリクエスト/レスポンス。**前の返信を待ってから次を送る**(in-flightは1)のが安全
- 初回フレームだけ CUDA カーネルの JIT で数秒、以降は定常

## セットアップ (GPUマシン)

```bash
git clone https://github.com/cumulo-autumn/StreamDiffusion.git
cd StreamDiffusion
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/pip install -r <このリポジトリ>/requirements.txt
cp <このリポジトリ>/sd_ws_server.py .   # utils.wrapper を import するため StreamDiffusion 直下に置く
```

依存の地雷: `numpy<2`(torchが2.x非対応) / `huggingface_hub==0.24.6`(diffusers 0.24 が cached_download 要求) / xformers 用 `setuptools`。

## 起動

```bash
CUDA_VISIBLE_DEVICES=0 python -u sd_ws_server.py \
  --size 512 --acceleration xformers --t-index 22 32 \
  --prompt "flowing ink wash painting, glowing smoke, elegant" --port 8765
```

主なオプション:

| フラグ | 既定 | 説明 |
|---|---|---|
| `--model` | `stabilityai/sd-turbo` | img2imgモデル |
| `--size` | 512 | 入出力の正方形サイズ |
| `--t-index` | `22 32` | ノイズ強度。**小さいほどAIの再解釈が強い**（形が崩れやすい）／大きいほど入力に忠実 |
| `--acceleration` | `xformers` | `none`/`xformers`/`tensorrt` |
| `--prompt` | (下記) | 初期プロンプト |
| `--port` | 8765 | 待受ポート |

> ⚠️ モデル構築(warmup)は必ず `asyncio.run()` の**外**(同期)で行うこと。イベントループ内で
> warmup を回すと、CUDAストリーム同期と干渉してトレースバック無しでサイレントクラッシュする。
> 本実装は `build_stream()` を `asyncio.run()` の前に呼んでいる。

## tmuxで常駐（推奨）

```bash
ssh gpu-host 'tmux new-session -d -s sd "cd ~/StreamDiffusion && \
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u sd_ws_server.py \
  --size 512 --acceleration xformers --t-index 22 32 --port 8765 > ~/srv.log 2>&1"'
```

> ⚠️ `pkill -f sd_ws_server.py` は使わない。`-f` が SSH の自シェル(`bash -c '...sd_ws_server.py...'`)に
> マッチしてセッションごと落ち(exit 255)、原因不明のまま詰まる。停止は `tmux kill-session -t sd`。

## 疎通テスト

```bash
python3 test_client.py   # ダミーフレームを3枚往復し client_out.jpg を保存
```

## リモートGPUを使う場合

Mac 等ローカルのクライアントからは SSH トンネルで繋ぐ:
```bash
ssh -N -L 8765:localhost:8765 gpu-host
```

## 実測レイテンシ (512px, t_index=[22,32], xformers, RTX 6000 Ada)

- 初回: ~7.3s（CUDAカーネルJIT）
- 定常: GPU推論 ~55ms。SSHトンネル経由の往復込みで ~0.3s/frame(≒3-4fps)

## 利用例

- [hand-trail-ai](https://github.com/ryok/hand-trail-ai) — TouchDesigner の手トレイルをこのサーバーでAI変換

## 解説記事

[手の軌跡をAIで塗り替える：TouchDesignerでリアルタイム Hand Tracking × AI Trails を作る](https://zenn.dev/ryok/articles/touchdesigner-hand-trail-ai)（Zenn）— このサーバーを自作した経緯と、asyncio内warmupのサイレントクラッシュ等の落とし穴。

## ライセンス

MIT
