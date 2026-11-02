FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    U2NET_HOME=/opt/rembg \
    REALESRGAN_BIN=/opt/realesrgan/realesrgan-ncnn-vulkan \
    REALESRGAN_MODELS=/app/models \
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl unzip libvulkan1 libgomp1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/realesrgan /opt/rembg

ARG REALESRGAN_RELEASE=20220424
RUN curl -fsSL "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-${REALESRGAN_RELEASE}-ubuntu.zip" -o /tmp/realesrgan.zip \
    && mkdir -p /tmp/realesrgan-unpacked \
    && unzip -q /tmp/realesrgan.zip -d /tmp/realesrgan-unpacked \
    && find /tmp/realesrgan-unpacked -type f -name 'realesrgan-ncnn-vulkan' -exec cp {} /opt/realesrgan/realesrgan-ncnn-vulkan \; \
    && test -x /opt/realesrgan/realesrgan-ncnn-vulkan -o -f /opt/realesrgan/realesrgan-ncnn-vulkan \
    && chmod +x /opt/realesrgan/realesrgan-ncnn-vulkan \
    && rm -rf /tmp/realesrgan.zip /tmp/realesrgan-unpacked

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
COPY THIRD_PARTY_NOTICES.md ./THIRD_PARTY_NOTICES.md
COPY README.md ./README.md

# Include the small, Apache-2.0 U2-Net-P model in the image so first use does
# not depend on a runtime model download.
RUN python -c "from rembg import new_session; new_session('u2netp')"

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
