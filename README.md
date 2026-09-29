# Mosquito Scanner 0.4
Client-only mosquito image detection with YOLito / YOLO11l and wingbeat sound detection with HumBug MozzBNN, both on ONNX Runtime Web 1.22.0.
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

## Sound AI (HumBug MozzBNN)
The microphone panel runs the HumBug mosquito event detector (University of Oxford, MIT license; https://github.com/HumBug-Mosquito/MozzBNN, paper: HumBugDB, NeurIPS 2021) on device. Only detection (mosquito vs background) is used; species classification is not included (its models are GB-scale and trained on lab/African field species).

- `dist/audio-tap.js` (AudioWorklet) forwards raw mic samples; `dist/sound-ai.js` keeps the last 2.1 s and analyses it every 0.5 s in `dist/sound-worker.js`.
- The worker resamples to 8 kHz (windowed sinc), reproduces MozzBNN's librosa features exactly (STFT 2048/512 Hann centred reflect, 128 Slaney mel, power_to_db ref=max top_db=80, standardisation) over one 1.92 s window (30 frames), and runs the model 10 times in one batch with dropout active (Monte Carlo, like the original BNN) to get the mean probability, predictive entropy and mutual information.
- The original pipeline standardises over a whole recording; live we standardise per window. On MozzBNN's sample recording this agrees with the published output for 96.8% of windows (99.1% with whole-recording normalisation).
- Per-window standardisation removes loudness, and in synthetic tests broadband low-frequency (brown) noise was scored as mosquito. The app therefore also requires a 250–1000 Hz spectral peak ≥ 9.5 dB above the band median (kept 96–99% of mosquito windows in the sample, rejected white/brown noise), and two hits in the last three analyses.
- JS features were checked against librosa: identical at 8 kHz; mean abs difference 0.06 (standardised units) after 48 kHz → 8 kHz resampling. About 65 ms per analysis on the development PC.

Proximity meter ("가까움"): the app reads one microphone channel, so direction cannot be computed. After an AI hit (the same AI + spectral-peak rule above), the meter locks onto the wingbeat frequency (tracked ±40 Hz) and shows, on every animation frame, how far that tone stands above the 250–1000 Hz band median (0–30 dB bar, 0.6 s time-constant smoothing). A 1.5 s change of more than 3 dB is shown as stronger/weaker, so the user can sweep the phone and follow the stronger direction. Only AI hits refresh the lock (5 s), so fans or tones the AI rejects cannot hold the meter.
Test (headless Chromium, fake microphone): MozzBNN sample recording 400–436 s (labelled mosquito) with a -12 → 0 → -24 dB volume envelope. Meter level correlated 0.69 with the envelope; stronger/weaker labels pointed the wrong way in about 20% of 0.5 s readings (the recording's own loudness fluctuates as the mosquito moves). A 500 Hz tone + 120 Hz hum + brown noise never turned the meter on. Not tested on a phone or a live mosquito; loudness also changes with phone orientation and hand position.

Rebuild the ONNX model (2 MB) from the original Keras weights (the Keras file's MC-dropout lambdas are Python 3.7 bytecode, so the graph is rebuilt from weights instead of loaded):

```
pip install h5py numpy onnx
curl -LO https://github.com/HumBug-Mosquito/HumBugDB/releases/download/v1.0/neurips_2021_humbugdb_keras_bnn_best.hdf5
python export_humbug.py neurips_2021_humbugdb_keras_bnn_best.hdf5 dist/model
```

## Model loading
The worker downloads the parts listed in `model/manifest.json`, checks total size and SHA-256 (when `crypto.subtle` is available, i.e. HTTPS/localhost), and only then stores the parts in Cache Storage under a hash-specific name. Later visits load from that cache; a new model hash replaces the old cache. A corrupted download is rejected and not cached.

## Validation
Converted model tested on upstream demo/dead.jpeg and a plain image. Native ORT maximum scores: 0.8525 and 0.0011 respectively. WebAssembly ORT test on the same photograph maximum 0.8525; one full-frame pass on the development host took approximately 4.8 seconds. These are execution checks, not accuracy evaluation. Model has NOT been validated on the user's room or iPhone. Source metadata's class label is '1'; mapped to mosquito estimate based on repository task.

## Scope
Detects visual mosquito candidates in captured photos, including stationary objects. No sound-based direction finding (only a loudness-based proximity meter), ultrasound sensing, guaranteed identification, or verified household recall. Small or blurred mosquitoes may be missed; patterns and other insects may cause false positives. Per-candidate model scores are not calibrated probabilities.

## Precise capture
The captured frame is scaled to at most 1920 px on the long side (1080p camera frames stay at native resolution), then analyzed as one full-frame pass plus overlapping 640 px tiles (25% overlap). Duplicates are removed by IoU NMS 0.35 plus a containment rule (a box ≥70% inside a higher-scoring box is dropped), which removes tile-edge clipped copies.

Small-spot zoom: mosquitoes only 10–20 px across in the analyzed photo score 0 at tile scale. On two user photos (wall, 9×16 px and ~10 px insects) the same model scored 0.65–0.88 once the crop was enlarged 4–8×. After the tile pass the app therefore finds compact spots darker than their 31 px neighbourhood (threshold max(18, min(40, 8×MAD)), 3–60 px, aspect ≤ 5, fill ≥ 20%), takes the 10 darkest, and runs the model on two enlarged crops per spot (spot scaled to ~60 and ~100 px of the 640 input). A zoomed detection counts only if it lies on the spot and its size is 0.4–2.5× the spot (larger boxes on window-screen texture and on-screen text were false positives). This adds up to 20 model runs; in a headless Chromium test (container CPU) analysis time went from 32–61 s to 122–149 s on 1920 px photos (2.4–3.8×).

4K camera: the camera is requested at 3840×2160 (the browser falls back to the best it has; the granted size is shown in the top-left label). The tile pass still runs on the photo scaled to 1920 px (4K tiles would need about 41 model runs instead of 9), but small-spot and tap zoom crops are cut from a full-resolution copy (long side ≤ 4096 px, also used for saved photos). Automatic zoom hits need a score ≥ 0.6 (in tests real insects scored 0.65–0.89 and specks/scuffs 0.53–0.77; small sample).
Test (headless Chromium, fake 4K camera from the user's wall photo): the 1080p build found only a wall speck (0.53) and missed the insect; the 4K build found the insect (0.82) and no speck, in about the same time (126 s vs 132 s). On the YOLito lab photo as a 4K feed both builds gave the same result. Not tested on an iPhone; whether Safari grants 4K to a web page is unverified.

Tap zoom: on the result photo, tapping snaps to the nearest dark spot within ~16 CSS px and runs four enlarged crops (spot at ~45/60/80/100 px). Taps on blank areas are not analyzed, because enlarging a blank patch made the model report a mosquito (0.77 on a plain wall).

Known limits: on busy backgrounds (grids, patterns) the darkest spots are usually not insects, so automatic zoom rarely helps there. Dark specks such as dirt can still be reported (one 0.77 false positive on a wall speck in the test photo). Progress shows an estimated remaining time. Leaving the page cancels a running scan but keeps finished results.

## v0.4 live motion classification
Enable camera, then live mosquito AI. A dedicated Web Worker runs one ROI inference at a time. AI detections attach only to a motion track that intersects the detected box in the analyzed frame and remains alive with the same ID and camera epoch when inference returns. Red boxes follow current track positions and expire; lost tracks and camera movement clear boxes. Rotating the device resets tracking to the new frame geometry. AI refresh is seconds apart, not every video frame. No new household accuracy claims.
