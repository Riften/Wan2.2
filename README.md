# Wan 2.2 http server

To start server

```bash
torchrun --nproc_per_node=8 server/run.py \
    --ckpt_dir ./Wan2.2-I2V-A14B \
    --dit_fsdp --t5_fsdp --ulysses_size 8 \
    --port 8000
```

To use client

```python
from client import WanClient

client = WanClient("http://server-host:8000")
video = client.generate_and_download(
    prompt="A cat surfing on the beach.",
    image_path="input.jpg",       # 本地图片，自动base64上传
    save_path="output.mp4",
)
```

To use server path as input and output

```python
video_path = client.generate_and_download(
    prompt="A cat surfing.",
    server_image_path="examples/i2v_input.JPG",
    server_save_path="/shared/storage/output.mp4",
)
```