# EtchLab

EtchLab is a browser-based image workbench for preparing artwork for laser workflows. The first release includes Real-ESRGAN upscaling, U²-Net-P background removal, image tiling, basic adjustments, and black-and-white laser preparation. Processing runs in the app container; source uploads and results use temporary memory or disk and are not persisted after the request.

## Tools

- **Upscale:** Real-ESRGAN general, anime, and animation models at 2×, 3×, or 4×.
- **Cut out:** U²-Net-P foreground extraction with a transparent PNG export.
- **Tile:** Even grid splitting with individually numbered PNGs in a ZIP archive.
- **Edit:** Crop by pixel coordinates, rotate, resize, adjust brightness and contrast, or convert to grayscale.
- **Laser prep:** Fixed threshold or Floyd–Steinberg dithering with optional inversion.

The app accepts PNG, JPEG, WebP, BMP, and TIFF files up to 24 MB and 12 megapixels. AI upscaling is limited to 48 megapixels of output; the browser selects the largest scale that fits that limit for each uploaded image. Threshold, dither, and background-removal output should be previewed in the user's laser software before production.

## Run locally with Docker

The container uses the Linux Real-ESRGAN ncnn Vulkan build and Mesa's software Vulkan driver. This allows the bundled model workflow to run on Cloud Run's CPU instances; processing speed depends on image dimensions and available CPU.

```sh
docker build -t etchlab .
docker run --rm -p 8080:8080 etchlab
```

Open `http://localhost:8080`.

The original Windows command-line wrapper remains available in `upscale.py`; see [README_windows.md](README_windows.md).

## Google Cloud deployment

Cloud Build uses `cloudbuild.yaml` to build a Linux container, run an HTTP smoke check across all five image operations, push the commit-tagged image to Artifact Registry, and deploy the Cloud Run service. The intended trigger watches pushes to `main`, so merging a pull request to `main` publishes that revision.

Defaults in the build config:

- Project: `image-project-v1`
- Region: `us-east1`
- Artifact Registry repository: `image-project-v1`
- Cloud Run service: `image-project-v1`
- Cloud Run: 4 vCPU, 8 GiB memory, concurrency 1, 0–2 instances, 300 second request timeout

The service is configured for unauthenticated access with a two-instance cap. Uploaded images are processed in temporary memory or disk and are not saved after a request. Cloud Run logs service events and errors; don't log image contents or user files. Public access means anyone with the URL can submit images, so review the cap and request limits before sharing it broadly.

Before the first deployment, create the Artifact Registry repository and runtime service account, grant the Cloud Build deploy identity Artifact Registry Writer and Cloud Run Admin, allow it to act as the runtime service account, connect GitHub to Cloud Build, and create a push-to-`main` trigger using `cloudbuild.yaml`. Cloud Build, Cloud Run, and Artifact Registry APIs must be enabled.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream software and model attributions.
