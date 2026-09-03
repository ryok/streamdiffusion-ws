#!/usr/bin/env python3
"""Mac側テストクライアント: ダミー画像を送り、変換画像を受信して保存"""
import asyncio, base64, io, time
from PIL import Image
import websockets

async def main():
    uri = "ws://localhost:8765"
    # テスト用のカラフルな入力画像(トレイルを模した円)
    img = Image.new("RGB", (512, 512), (10, 10, 30))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for i, c in enumerate([(255,60,60),(60,255,120),(80,120,255)]):
        d.ellipse([180+i*40, 200+i*30, 280+i*40, 300+i*30], fill=c)
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    print(f"connecting {uri} ...")
    async with websockets.connect(uri, max_size=8*1024*1024) as ws:
        print("connected. sending 3 frames...")
        for i in range(3):
            t0 = time.time()
            await ws.send(b64)
            resp = await ws.recv()
            dt = (time.time()-t0)*1000
            print(f"  frame{i}: round-trip {dt:.0f}ms, resp {len(resp)} bytes")
        # 最後の結果を保存
        out = Image.open(io.BytesIO(base64.b64decode(resp)))
        out.save("client_out.jpg")
        print(f"saved client_out.jpg {out.size}")

asyncio.run(main())
