# Third-party software and model notices

## Real-ESRGAN

EtchLab uses the Real-ESRGAN ncnn Vulkan executable and model files. The upstream Real-ESRGAN project is distributed under the BSD 3-Clause License. The Linux executable is downloaded from the upstream `v0.2.5.0` release during the container build. Model files in `models/` originate from the Real-ESRGAN ncnn Vulkan release bundled with the original project.

- Project: https://github.com/xinntao/Real-ESRGAN
- Linux release: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0
- License: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE

## U²-Net-P / rembg

Background removal uses the `rembg` Python package and the U²-Net-P model (`u2netp`) distributed by rembg. The rembg source code is MIT-licensed. The U²-Net project and U²-Net-P weights are identified by the upstream project as Apache-2.0.

- rembg: https://github.com/danielgatis/rembg
- U²-Net: https://github.com/xuebinqin/U-2-Net
- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
- MIT License: https://opensource.org/license/mit

## Python packages

FastAPI, Uvicorn, Pillow, python-multipart, ONNX Runtime, and rembg retain their respective upstream licenses and notices. Their packages are pinned in `requirements.txt`.
