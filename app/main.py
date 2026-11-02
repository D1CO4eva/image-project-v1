from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import deque
from pathlib import Path
from threading import Lock
from time import monotonic
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
UPSCALE_BIN = Path(os.getenv("REALESRGAN_BIN", "/opt/realesrgan/realesrgan-ncnn-vulkan"))
MODEL_DIR = Path(os.getenv("REALESRGAN_MODELS", str(ROOT / "models")))
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
MAX_PIXELS = 12_000_000
MAX_UPSCALE_OUTPUT_PIXELS = 48_000_000
ALLOWED_MODELS = {"realesrgan-x4plus", "realesrgan-x4plus-anime", "realesr-animevideov3"}

app = FastAPI(title="EtchLab", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_rembg_session = None
_rembg_lock = Lock()
_ai_limit_lock = Lock()
_ai_requests: deque[float] = deque()
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
AI_REQUESTS_PER_HOUR = 5
AI_REQUEST_WINDOW_SECONDS = 3600


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


def _image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}:
            raise HTTPException(415, "Use a PNG, JPEG, WebP, BMP, or TIFF image.")
        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(413, "This image is too large. The limit is 12 megapixels.")
        image.load()
        return ImageOps.exif_transpose(image)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(415, "The uploaded file is not a readable image.") from exc


def _encoded(image: Image.Image, fmt: str = "PNG", quality: int = 92) -> bytes:
    stream = io.BytesIO()
    if fmt == "JPEG" and image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, "white")
        if "A" in image.getbands():
            background.paste(image, mask=image.getchannel("A"))
        else:
            background.paste(image.convert("RGB"))
        image = background
    image.save(stream, format=fmt, quality=quality)
    return stream.getvalue()


def _attachment(payload: bytes, name: str, content_type: str = "image/png") -> Response:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "edited-image"
    return Response(
        payload,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"', "Cache-Control": "no-store"},
    )


def _deepseek_plan(image: Image.Image, instruction: str) -> dict:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(503, "AI Edit is not configured on this deployment.")
    instruction = instruction.strip()
    if not instruction or len(instruction) > 500:
        raise HTTPException(400, "Describe the change in 1 to 500 characters.")

    # Send only a downsized visual analysis copy. Apply the returned plan to the
    # full-resolution image held by this request; no upload is persisted.
    preview = image.copy()
    preview.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    preview_payload = _encoded(preview.convert("RGB"), "JPEG", 82)
    image_url = "data:image/jpeg;base64," + base64.b64encode(preview_payload).decode("ascii")
    prompt = (
        "You are an image-editing assistant for laser artwork. Analyze the supplied image and the user's request. "
        "Return only a JSON object with keys summary, brightness, contrast, saturation, sharpness, grayscale. "
        "summary is a short plain-language description. The four numeric values are multipliers in [0.75,1.25]; "
        "use 1.0 for unchanged. grayscale is boolean. Choose conservative global tonal adjustments that best "
        "serve the requested result. Do not claim to crop, remove objects/backgrounds, replace content, draw, "
        "or upscale; those operations are unavailable here. If the request cannot be achieved with these controls, "
        "explain that in summary and return neutral values. User request: " + instruction
    )
    body = json.dumps({
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
        "response_format": {"type": "json_object"},
        "max_tokens": 300,
        "temperature": 0.1,
    }).encode("utf-8")
    request = urllib.request.Request(
        DEEPSEEK_URL, data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = json.loads(response.read(64 * 1024))
        plan = json.loads(raw["choices"][0]["message"]["content"])
    except urllib.error.HTTPError as exc:
        # Never relay provider body; it may contain sensitive diagnostics.
        if exc.code in (401, 403):
            raise HTTPException(503, "DeepSeek rejected the configured API key.") from exc
        raise HTTPException(502, "DeepSeek could not analyze this image. Please retry.") from exc
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(502, "DeepSeek could not analyze this image. Please retry.") from exc

    if not isinstance(plan, dict):
        raise HTTPException(502, "DeepSeek returned an unusable edit plan. Please retry.")
    summary = plan.get("summary")
    if not isinstance(summary, str):
        raise HTTPException(502, "DeepSeek returned an unusable edit plan. Please retry.")
    safe_plan = {"summary": summary[:240], "grayscale": plan.get("grayscale") is True}
    for key in ("brightness", "contrast", "saturation", "sharpness"):
        value = plan.get(key, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.75 <= value <= 1.25:
            raise HTTPException(502, "DeepSeek returned an unsafe edit plan. Please retry.")
        safe_plan[key] = float(value)
    return safe_plan


def _check_ai_rate_limit() -> None:
    now = monotonic()
    with _ai_limit_lock:
        while _ai_requests and now - _ai_requests[0] >= AI_REQUEST_WINDOW_SECONDS:
            _ai_requests.popleft()
        if len(_ai_requests) >= AI_REQUESTS_PER_HOUR:
            raise HTTPException(429, "AI Edit is limited to 5 requests per hour per running instance.")
        _ai_requests.append(now)


@app.post("/api/ai-edit")
def ai_edit(file: UploadFile = File(...), instruction: str = Form(...)) -> Response:
    # Keep the public API-key-backed endpoint bounded. The limit is per instance;
    # Cloud Run currently runs at most two instances.
    _check_ai_rate_limit()
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The upload limit is 24 MB.")
    if not data:
        raise HTTPException(400, "Choose an image first.")
    image = _image(data)
    plan = _deepseek_plan(image, instruction)
    enhancers = {"brightness": "Brightness", "contrast": "Contrast", "saturation": "Color", "sharpness": "Sharpness"}
    for key, enhancer_name in enhancers.items():
        if plan[key] != 1.0:
            image = ImageEnhance.__dict__[enhancer_name](image).enhance(plan[key])
    if plan["grayscale"]:
        image = ImageOps.grayscale(image)
    response = _attachment(_encoded(image), f"{Path(file.filename or 'image').stem[:80]}_ai_edit.png")
    response.headers["X-EtchLab-AI-Summary"] = quote(plan["summary"][:240])
    response.headers["Access-Control-Expose-Headers"] = "X-EtchLab-AI-Summary"
    return response


def _upscale(image: Image.Image, model: str, scale: int) -> bytes:
    if not UPSCALE_BIN.is_file() or not MODEL_DIR.is_dir():
        raise HTTPException(503, "The upscaling engine is not available in this deployment.")
    with tempfile.TemporaryDirectory(prefix="etchlab-") as folder:
        source = Path(folder) / "source.png"
        target = Path(folder) / "upscaled.png"
        image.save(source, format="PNG")
        try:
            result = subprocess.run(
                [str(UPSCALE_BIN), "-i", str(source), "-o", str(target), "-m", str(MODEL_DIR),
                 "-n", model, "-s", str(scale), "-t", "256"],
                capture_output=True, text=True, timeout=540, check=False,
                env={**os.environ, "VK_ICD_FILENAMES": os.getenv(
                    "VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")},
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(504, "Upscaling took too long. Try a smaller source image.") from exc
        if result.returncode != 0 or not target.is_file():
            detail = (result.stderr or result.stdout or "No diagnostic was returned.")[-1200:]
            raise HTTPException(503, f"The upscaling engine could not process this image: {detail}")
        return target.read_bytes()


@app.post("/api/process")
def process_image(
    file: UploadFile = File(...),
    operation: str = Form(...),
    model: str = Form("realesrgan-x4plus"),
    scale: int = Form(4),
    rows: int = Form(2),
    columns: int = Form(2),
    rotation: int = Form(0),
    width: int = Form(0),
    height: int = Form(0),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    grayscale: bool = Form(False),
    crop_x: int = Form(0),
    crop_y: int = Form(0),
    crop_width: int = Form(0),
    crop_height: int = Form(0),
    threshold: int = Form(128),
    laser_method: str = Form("threshold"),
    invert: bool = Form(False),
) -> Response:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The upload limit is 24 MB.")
    if not data:
        raise HTTPException(400, "Choose an image first.")
    image = _image(data)
    stem = Path(file.filename or "image").stem[:80]

    if operation == "upscale":
        if model not in ALLOWED_MODELS or scale not in (2, 3, 4):
            raise HTTPException(400, "Choose a supported model and scale.")
        output_pixels = image.width * image.height * scale * scale
        if output_pixels > MAX_UPSCALE_OUTPUT_PIXELS:
            max_scale = int((MAX_UPSCALE_OUTPUT_PIXELS / (image.width * image.height)) ** 0.5)
            raise HTTPException(
                413,
                f"At {scale}× this image would exceed the 48 megapixel output limit. "
                f"Choose {max_scale}× or lower, or use a smaller image.",
            )
        return _attachment(_upscale(image, model, scale), f"{stem}_{scale}x.png")

    if operation == "remove-background":
        global _rembg_session
        try:
            from rembg import new_session, remove
            with _rembg_lock:
                if _rembg_session is None:
                    _rembg_session = new_session("u2netp")
                cutout = remove(data, session=_rembg_session, force_return_bytes=True)
            return _attachment(cutout, f"{stem}_cutout.png")
        except Exception as exc:
            raise HTTPException(503, "Background removal could not run. Please retry in a moment.") from exc

    if operation == "split":
        if not (1 <= rows <= 12 and 1 <= columns <= 12 and rows * columns <= 64):
            raise HTTPException(400, "Choose a grid of at most 64 tiles.")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for row in range(rows):
                top = round(row * image.height / rows)
                bottom = round((row + 1) * image.height / rows)
                for col in range(columns):
                    left = round(col * image.width / columns)
                    right = round((col + 1) * image.width / columns)
                    tile = image.crop((left, top, right, bottom))
                    bundle.writestr(f"{stem}_r{row + 1:02}_c{col + 1:02}.png", _encoded(tile))
        return _attachment(archive.getvalue(), f"{stem}_tiles.zip", "application/zip")

    if operation == "edit":
        if crop_width > 0 and crop_height > 0:
            x0, y0 = max(0, crop_x), max(0, crop_y)
            x1, y1 = min(image.width, x0 + crop_width), min(image.height, y0 + crop_height)
            if x1 <= x0 or y1 <= y0:
                raise HTTPException(400, "The crop rectangle must overlap the image.")
            image = image.crop((x0, y0, x1, y1))
        if rotation not in (0, 90, 180, 270):
            raise HTTPException(400, "Rotation must be 0, 90, 180, or 270 degrees.")
        if rotation:
            image = image.rotate(-rotation, expand=True)
        if not (0.25 <= brightness <= 2.0 and 0.25 <= contrast <= 2.0):
            raise HTTPException(400, "Brightness and contrast must be between 0.25 and 2.")
        if brightness != 1:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        if contrast != 1:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        if grayscale:
            image = ImageOps.grayscale(image)
        if width or height:
            if not (0 <= width <= 12000 and 0 <= height <= 12000):
                raise HTTPException(400, "Output dimensions must be between 1 and 12,000 pixels.")
            if not width:
                width = max(1, round(image.width * height / image.height))
            if not height:
                height = max(1, round(image.height * width / image.width))
            if width * height > 24_000_000:
                raise HTTPException(413, "The edited output may not exceed 24 megapixels.")
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return _attachment(_encoded(image), f"{stem}_edited.png")

    if operation == "laser":
        if not 0 <= threshold <= 255 or laser_method not in ("threshold", "dither"):
            raise HTTPException(400, "Choose a threshold from 0 to 255 or Floyd–Steinberg dithering.")
        gray = ImageOps.grayscale(image)
        if laser_method == "dither":
            result = gray.convert("1", dither=Image.Dither.FLOYDSTEINBERG).convert("L")
        else:
            result = gray.point(lambda value: 255 if value >= threshold else 0, mode="1").convert("L")
        if invert:
            result = ImageOps.invert(result)
        return _attachment(_encoded(result), f"{stem}_laser.png")

    raise HTTPException(400, "Choose a supported image operation.")


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}
