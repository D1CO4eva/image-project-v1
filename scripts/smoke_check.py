"""Exercise every image operation through the running HTTP app in Cloud Build."""

from __future__ import annotations

import io
import subprocess
import time
import zipfile

import requests
from PIL import Image


BASE_URL = "http://127.0.0.1:8080"
server = subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8080"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)


def wait_until_ready() -> None:
    for _ in range(40):
        if server.poll() is not None:
            raise RuntimeError("The EtchLab web server stopped during startup.")
        try:
            if requests.get(f"{BASE_URL}/healthz", timeout=1).json() == {"status": "ok"}:
                return
        except requests.RequestException:
            time.sleep(0.5)
    raise RuntimeError("The EtchLab web server did not become ready.")


def process(image_data: bytes, **fields: str) -> requests.Response:
    return requests.post(
        f"{BASE_URL}/api/process",
        files={"file": ("smoke.png", image_data, "image/png")},
        data=fields,
        timeout=240,
    )


try:
    wait_until_ready()
    assert "EtchLab" in requests.get(BASE_URL, timeout=5).text

    source = Image.new("RGB", (32, 32), (128, 128, 128))
    source_bytes = io.BytesIO()
    source.save(source_bytes, format="PNG")
    payload = source_bytes.getvalue()

    upscaled = process(payload, operation="upscale", model="realesrgan-x4plus", scale="2")
    upscaled.raise_for_status()
    assert Image.open(io.BytesIO(upscaled.content)).size == (64, 64)
    print("Real-ESRGAN upscaling passed")

    from_background_remover = process(payload, operation="remove-background")
    from_background_remover.raise_for_status()
    assert Image.open(io.BytesIO(from_background_remover.content)).mode == "RGBA"
    print("U²-Net-P background removal passed")

    tiles = process(payload, operation="split", rows="2", columns="2")
    tiles.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(tiles.content)) as archive:
        assert len(archive.namelist()) == 4
        assert all(Image.open(io.BytesIO(archive.read(name))).size == (16, 16) for name in archive.namelist())
    print("Image tiling passed")

    edited = process(payload, operation="edit", rotation="90", width="64", grayscale="true")
    edited.raise_for_status()
    edited_image = Image.open(io.BytesIO(edited.content))
    assert edited_image.size == (64, 64) and edited_image.mode == "L"
    print("Crop/resize/rotate/tone editing route passed")

    laser = process(payload, operation="laser", threshold="128", laser_method="threshold")
    laser.raise_for_status()
    laser_image = Image.open(io.BytesIO(laser.content))
    assert laser_image.size == (32, 32) and set(laser_image.convert("L").getdata()) <= {0, 255}
    print("Laser threshold preparation passed")
finally:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=5)
