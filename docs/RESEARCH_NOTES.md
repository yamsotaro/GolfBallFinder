# Research notes and reusable technology

Research refreshed: 2026-08-22.

## Apple device/runtime

### iPhone 16 Pro

Apple technical specs:

- A18 Pro;
- 6-core CPU;
- 6-core GPU;
- 16-core Neural Engine;
- 48MP Fusion 24mm;
- 12MP 2x 48mm;
- 5x telephoto;
- LiDAR.

Source: https://support.apple.com/ja-jp/121031

Implication: local object detection is the correct first architecture. 2x is worth testing for search/detail. LiDAR is optional later, not primary detection.

### Core ML compute units

Apple documents `.cpuAndNeuralEngine` as allowing CPU + Neural Engine but excluding GPU. `.all` allows the OS to choose all available units.

Sources:

- https://developer.apple.com/documentation/coreml/mlcomputeunits
- https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine

### Vision

`VNCoreMLRequest` runs Core ML models as Vision image-analysis requests. `VNTrackObjectRequest` tracks a previously identified bounding box across frames.

Sources:

- https://developer.apple.com/documentation/vision/vncoremlrequest
- https://developer.apple.com/documentation/vision/vntrackobjectrequest

The current MVP repeatedly re-runs the detector on an ROI instead of using Vision tracking because it gives both tracking and re-classification. Vision tracking remains a legitimate later optimization.

## Ultralytics iOS

Repository: https://github.com/ultralytics/yolo-ios-app

Pinned MVP dependency: v8.9.13.

Relevant capabilities:

- native Swift package;
- Core ML/Vision inference;
- custom `.mlpackage` support;
- YOLO26 and legacy YOLO11-style outputs;
- synchronous image inference API and real-time camera components;
- normalized `Box.xywhn` results;
- v8.9.13 includes Float16 detection-output safety and standardized 640x640 non-classification model inputs.

Performance notes (important caveat): current published measurements use iPhone 17 Pro/A19 Pro, not iPhone 16 Pro. They show YOLO26n detect can be real-time in this stack and that CPU + Neural Engine preferred is substantially faster than CPU-only in their test. Do not copy those latency numbers into iPhone 16 Pro claims; benchmark the target device.

Sources:

- https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md
- https://github.com/ultralytics/yolo-ios-app/releases/tag/v8.9.13
- https://docs.ultralytics.com/integrations/coreml

License: AGPL-3.0. Personal prototype is the immediate use. Revisit before distribution/commercialization.

## SAHI / sliced inference

SAHI addresses small-object detection by slicing a larger image and running a detector on crops. The paper reports AP improvements on small-object benchmarks. The Python project is MIT licensed.

Sources:

- https://github.com/obss/sahi
- https://arxiv.org/abs/2202.06934

Use here: concept only. The iOS implementation has a tiny native `ScanScheduler` that alternates full-frame and overlapping crops and then locks to an ROI. This avoids embedding a Python runtime.

## Golf-ball public model — bootstrap

Hugging Face: https://huggingface.co/notjulietxd/golf-ball-tracker

Model card reports:

- YOLOv8-nano;
- 559 real ball images + 500 synthetic images;
- 640x640 input;
- mAP50 81.2%;
- mAP50-95 58.6%;
- Apache-2.0;
- limitation: mixed sports-ball data and real golf-ball data would improve it.

Published `best.pt` SHA256: `45e8f8bd8975dc7f437919a11c3f6ee1fe7c8ae40b0f49910d0677d1c0326791`.

Use: bootstrap only, to validate deployment before collecting the target-domain dataset.

## Golf-ball datasets worth inspecting

These are leads, not automatically approved training inputs. Inspect label quality and record license/attribution before merging.

### Roboflow golf-ball smartphone

https://universe.roboflow.com/yukiwatanabaes-workspace/golf-ball-smartphone

As of research date:

- 109 images;
- YOLOv11n model listed;
- mAP@50 83.8%;
- precision 83.3%;
- recall 76.2%;
- CC BY 4.0.

Small dataset, but directly relevant as a smartphone-oriented example.

### CreateML golf-ball detection

https://universe.roboflow.com/createml-ew7k2/golf-ball-detection-dqzau

- roughly 3.2k images reported;
- one `golfball` class;
- CC BY 4.0.

### RatssTech Golf-Ball Detection V2

https://universe.roboflow.com/ratsstech/golf-ball-detection-v2

- roughly 6.9k images reported;
- CC BY 4.0.

### Other datasets

Roboflow Universe search contains multiple golf-ball datasets. Quantity is less important than domain similarity and annotation quality. Avoid blindly concatenating duplicates or video-adjacent frames from unknown sources.

## Older golf-ball detector/tracker research implementation

`rucv/golf_ball` is a useful conceptual reference for detection + tracking, but it is not the recommended iOS runtime architecture. Prefer the current Core ML/YOLO path and borrow only general tracking ideas after checking its license/source.

Repository: https://github.com/rucv/golf_ball

## Apple personal-device installation

Apple's developer-account documentation says a free Personal Team can install/test apps on a personal device, but App IDs/devices/provisioning profiles expire after seven days and the app must be rebuilt/reinstalled. This is workable for initial testing; a paid developer membership is more convenient for routine use.

Source: https://developer.apple.com/help/account/basics/about-your-developer-account
