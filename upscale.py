#!/usr/bin/env python3
"""Portable wrapper for the bundled Real-ESRGAN ncnn Vulkan executable."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "realesrgan-ncnn-vulkan.exe"
MODELS = ROOT / "models"
MODELS_ALLOWED = ("realesrgan-x4plus", "realesrgan-x4plus-anime", "realesr-animevideov3")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upscale images with the shared local Real-ESRGAN install.")
    parser.add_argument("-i", "--input", required=True, help="Input image or input directory")
    parser.add_argument("-o", "--output", required=True, help="Output image or output directory")
    parser.add_argument("-s", "--scale", type=int, choices=(2, 3, 4), default=4, help="Scale for this pass (default: 4)")
    parser.add_argument("-n", "--model", choices=MODELS_ALLOWED, default="realesrgan-x4plus", help="Bundled model (default: realesrgan-x4plus)")
    parser.add_argument("-t", "--tile-size", type=int, default=256, help="Tile size; use 0 for automatic (default: 256)")
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if not EXE.is_file() or not MODELS.is_dir():
        parser.error(f"Bundled executable or models folder is missing under {ROOT}")

    if source.is_dir():
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    command = [str(EXE), "-i", str(source), "-o", str(output), "-m", str(MODELS), "-n", args.model, "-s", str(args.scale), "-t", str(args.tile_size)]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
