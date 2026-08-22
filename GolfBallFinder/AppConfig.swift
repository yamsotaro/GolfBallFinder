import CoreGraphics
import Foundation

/// Single place for tuning the MVP without scattering magic numbers through the app.
enum AppConfig {
    /// Core ML model resource name. Xcode compiles GolfBall.mlpackage into the app bundle.
    static let modelName = "GolfBall"

    /// Ask the detector for low-confidence candidates, then suppress false positives temporally.
    static let modelConfidenceThreshold = 0.12
    static let modelIoUThreshold = 0.45
    static let maxDetections = 8

    /// Temporal confirmation. A low detector threshold + repeated spatial agreement is preferred
    /// to a high one-frame threshold because the target is frequently tiny or partially occluded.
    static let stabilizerWindow = 5
    static let requiredHits = 3
    static let candidateMinConfidence: Float = 0.14
    static let confirmedAverageConfidence: Float = 0.20
    static let maxCenterStep: CGFloat = 0.20
    static let foundPersistenceFrames = 5

    /// Inference scheduling. One model invocation per processed frame keeps thermals bounded.
    static let processedFPS: Double = 20
    static let roiLockFrames = 8
    static let roiLostTolerance = 3
    static let minimumROISide: CGFloat = 0.34
    static let roiExpansion: CGFloat = 5.0

    /// Search tiles overlap on purpose so a ball near an edge is not repeatedly split by crops.
    static let searchTiles: [CGRect] = [
        CGRect(x: 0.00, y: 0.00, width: 0.62, height: 0.62),
        CGRect(x: 0.38, y: 0.00, width: 0.62, height: 0.62),
        CGRect(x: 0.00, y: 0.38, width: 0.62, height: 0.62),
        CGRect(x: 0.38, y: 0.38, width: 0.62, height: 0.62),
        CGRect(x: 0.19, y: 0.19, width: 0.62, height: 0.62),
    ]

    /// Normalize label spelling from seed/public models. Final training should use `golf_ball` only.
    static let acceptedClassTokens: Set<String> = [
        "golfball", "golf_ball", "golf-ball", "golf ball", "ball"
    ]
}
