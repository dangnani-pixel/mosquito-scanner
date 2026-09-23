# Mosquito Scanner 0.3
Client-only mosquito image detection with YOLito / YOLO11l, ONNX Runtime Web 1.22.0.
Source and original checkpoint metadata license: AGPL-3.0. See LICENSE and dist/NOTICE.txt.

## Run
Serve dist over HTTPS (localhost also works). Enable camera, focus on wall, select AI precise capture. Or select a photograph. Inference is performed on device. The original motion tracking and microphone spectrum features remain separate auxiliary functions, not mosquito classifiers.

## Model conversion
Download weights/best.pt from https://github.com/WildMosquit0/YOLito (Git LFS actual file) as yolito.pt.
Original SHA256: cbcc0812977a81958a849d249d19246069596acaa2799b832de86e0cce7e0841.
Install CPU torch, torchvision, ultralytics==8.3.65, onnx, onnxruntime.
Run `python export_model.py yolito.pt`, then `python compact_model.py yolito.onnx dist/model`.
Export uses legacy torch ONNX export (dynamo=False), opset17, 640x640, one class. Compact stores float initializers as half with Cast nodes back to float, splits model into 8 MiB parts.
Use ONNX Runtime Web 1.22.0 dist/ort.wasm.min.js, ort-wasm-simd-threaded.mjs and ort-wasm-simd-threaded.wasm in dist/vendor. Use one WASM thread for iOS and no cross-origin isolation dependency.
Upstream model training/inference source: https://github.com/WildMosquit0/YOLito and https://github.com/ultralytics/ultralytics/tree/v8.3.65.

## Validation
Converted model tested on upstream demo/dead.jpeg and a plain image. Native ORT maximum scores: 0.8525 and 0.0011 respectively. WebAssembly ORT test on the same photograph maximum 0.8525; one full-frame pass on the development host took approximately 4.8 seconds. These are execution checks, not accuracy evaluation. Model has NOT been validated on the user's room or iPhone. Source metadata's class label is '1'; mapped to mosquito estimate based on repository task.

## Scope
Detects visual mosquito candidates in captured photos, including stationary objects. No sound-based location, ultrasound sensing, guaranteed identification, or verified household recall. Small or blurred mosquitoes may be missed; patterns and other insects may cause false positives. Per-candidate model scores are not calibrated probabilities.

## v0.4 live motion classification
Enable camera, then live mosquito AI. A dedicated Web Worker runs one ROI inference at a time. AI detections attach only to a motion track that intersects the detected box in the analyzed frame and remains alive with the same ID and camera epoch when inference returns. Red boxes follow current track positions and expire; lost tracks and camera movement clear boxes. AI refresh is seconds apart, not every video frame. No new household accuracy claims.
