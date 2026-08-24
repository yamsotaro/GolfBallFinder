import CoreImage
import Foundation
import UltralyticsYOLO

final class GolfBallDetector: @unchecked Sendable {
    private let stateLock = NSLock()
    private var model: YOLO?
    private var modelReady = false
    private var storedLoadError: Error?

    var isLoaded: Bool {
        withStateLock { modelReady }
    }

    var loadError: Error? {
        withStateLock { storedLoadError }
    }

    func load(completion: @escaping @Sendable (Result<Void, Error>) -> Void) {
        withStateLock {
            model = nil
            modelReady = false
            storedLoadError = nil
        }
        var instance: YOLO?
        instance = YOLO(
            AppConfig.modelName,
            task: .detect,
            useGpu: true,
            numItemsThreshold: AppConfig.maxDetections
        ) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let loaded):
                loaded.setThresholds(
                    numItems: AppConfig.maxDetections,
                    confidence: AppConfig.modelConfidenceThreshold,
                    iou: AppConfig.modelIoUThreshold
                )
                self.withStateLock {
                    self.model = loaded
                    self.modelReady = true
                    self.storedLoadError = nil
                }
                completion(.success(Void()))
            case .failure(let error):
                self.withStateLock {
                    self.modelReady = false
                    self.storedLoadError = error
                }
                completion(.failure(error))
            }
        }
        withStateLock {
            // Model loading is normally asynchronous, but preserve a synchronous success result.
            if !modelReady {
                model = instance
            }
        }
    }

    func predict(
        image: CIImage,
        cropRegion: CGRect,
        fullFrameImageSize: CGSize,
        source: ScanSource,
        timestamp: TimeInterval
    ) -> (observations: [DetectionObservation], inferenceMs: Double) {
        guard let model = withStateLock({ modelReady ? model : nil }) else { return ([], 0) }

        let result = model(image)
        let observations = result.boxes.compactMap { box -> DetectionObservation? in
            guard shouldAcceptClass(box.cls, allNames: result.names) else { return nil }
            // UltralyticsYOLO v8.9.13 has already decoded raw Core ML center-based
            // [cx, cy, w, h] model-pixel channels, removed letterbox padding, converted the
            // origin to top-left, and normalized the result. Do not center-decode xywhn again.
            guard let detectorLocalRect = validatedNormalizedTopLeftDetectionRect(
                box.xywhn,
                imageSize: result.orig_shape
            ) else { return nil }
            let mapped = mapCropRectToFullFrame(detectorLocalRect, cropRegion: cropRegion)
            guard let fullRect = validatedNormalizedTopLeftDetectionRect(
                mapped,
                imageSize: fullFrameImageSize
            ) else { return nil }
            return DetectionObservation(
                normalizedRect: fullRect,
                confidence: box.conf,
                className: box.cls,
                source: source,
                timestamp: timestamp
            )
        }
        .sorted { $0.confidence > $1.confidence }

        return (observations, result.inferenceMs)
    }

    private func shouldAcceptClass(_ raw: String, allNames: [String]) -> Bool {
        let normalized = raw.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if AppConfig.acceptedClassTokens.contains(normalized) { return true }

        // A final custom model should be one-class. Accept its sole label even if named differently.
        return Set(allNames).count == 1
    }

    private func withStateLock<T>(_ operation: () -> T) -> T {
        stateLock.lock()
        defer { stateLock.unlock() }
        return operation()
    }
}
