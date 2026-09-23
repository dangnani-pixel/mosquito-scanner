# Mosquito Scanner 0.4
Client-only mosquito image detection with YOLito / YOLO11l, ONNX Runtime Web 1.22.0.
Source and original checkpoint metadata license: AGPL-3.0. See LICENSE and dist/NOTICE.txt.

Live site (GitHub Pages): https://dangnani-pixel.github.io/mosquito-scanner/

## Run
Serve `dist` over HTTPS (localhost also works):

```
python -m http.server 8000 --bind 127.0.0.1 --directory dist
```

Enable camera, focus on wall, select AI precise capture. Or select a photograph. Inference is performed on device. The original motion tracking and microphone spectrum features remain separate auxiliary functions, not mosquito classifiers.

`dist/` already contains everything needed: app, converted model (`dist/model`), ONNX Runtime Web (`dist/vendor`), LICENSE.txt and NOTICE.txt.

## Deploy (GitHub Pages)
`.github/workflows/pages.yml` publishes `dist/` on every push to `main`, and adds `source.zip` (this repository at that commit) next to `index.html` for the AGPL source link in the footer.
One-time setup: repository Settings → Pages → Source: **GitHub Actions**.

## Model conversion
Only needed to regenerate `dist/model`. Download weights/best.pt from https://github.com/WildMosquit0/YOLito (Git LFS actual file) as yolito.pt.
Original SHA256: cbcc0812977a81958a849d249d19246069596acaa2799b832de86e0cce7e0841.
Install CPU torch first (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`), then `pip install -r requirements.txt` (ultralytics==8.3.65, onnx, onnxruntime).
Run `python export_model.py yolito.pt`, then `python compact_model.py yolito.onnx dist/model`.
Export uses the legacy torch ONNX exporter (`dynamo=False` when the installed torch supports that argument), opset17, 640x640, one class. Compact stores float initializers as half with Cast nodes back to float, splits the model into 8 MiB parts and writes size + SHA-256 to `manifest.json`. Works on Windows (no temp-path assumptions); old `yolito-*.bin` parts in the output folder are removed first.
Use ONNX Runtime Web 1.22.0 dist/ort.wasm.min.js, ort-wasm-simd-threaded.mjs and ort-wasm-simd-threaded.wasm in dist/vendor. Use one WASM thread for iOS and no cross-origin isolation dependency.
Upstream model training/inference source: https://github.com/WildMosquit0/YOLito and https://github.com/ultralytics/ultralytics/tree/v8.3.65.

## Model loading
The worker downloads the parts listed in `model/manifest.json`, checks total size and SHA-256 (when `crypto.subtle` is available, i.e. HTTPS/localhost), and only then stores the parts in Cache Storage under a hash-specific name. Later visits load from that cache; a new model hash replaces the old cache. A corrupted download is rejected and not cached.

## Validation
Converted model tested on upstream demo/dead.jpeg and a plain image. Native ORT maximum scores: 0.8525 and 0.0011 respectively. WebAssembly ORT test on the same photograph maximum 0.8525; one full-frame pass on the development host took approximately 4.8 seconds. These are execution checks, not accuracy evaluation. Model has NOT been validated on the user's room or iPhone. Source metadata's class label is '1'; mapped to mosquito estimate based on repository task.

## Scope
Detects visual mosquito candidates in captured photos, including stationary objects. No sound-based location, ultrasound sensing, guaranteed identification, or verified household recall. Small or blurred mosquitoes may be missed; patterns and other insects may cause false positives. Per-candidate model scores are not calibrated probabilities.

## Precise capture
The captured frame is scaled to at most 1920 px on the long side (1080p camera frames stay at native resolution), then analyzed as one full-frame pass plus overlapping 640 px tiles (25% overlap). Duplicates are removed by IoU NMS 0.35 plus a containment rule (a box ≥70% inside a higher-scoring box is dropped), which removes tile-edge clipped copies. Progress shows an estimated remaining time. Leaving the page cancels a running scan but keeps finished results.

## v0.4 live motion classification
Enable camera, then live mosquito AI. A dedicated Web Worker runs one ROI inference at a time. AI detections attach only to a motion track that intersects the detected box in the analyzed frame and remains alive with the same ID and camera epoch when inference returns. Red boxes follow current track positions and expire; lost tracks and camera movement clear boxes. Rotating the device resets tracking to the new frame geometry. AI refresh is seconds apart, not every video frame. No new household accuracy claims.
