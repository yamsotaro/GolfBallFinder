import CoreGraphics
import Foundation

enum ScanSource: String, Sendable {
    case fullFrame
    case tile
    case roi
}

struct DetectionObservation: Equatable, Sendable {
    let normalizedRect: CGRect
    let confidence: Float
    let className: String
    let source: ScanSource
    let timestamp: TimeInterval

    var center: CGPoint {
        CGPoint(x: normalizedRect.midX, y: normalizedRect.midY)
    }
}

enum FinderState: Equatable, Sendable {
    case loading
    case scanning
    case candidate(DetectionObservation)
    case found(DetectionObservation)
    case error(String)
}
