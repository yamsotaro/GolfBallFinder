import CoreGraphics
import Foundation

enum ScanSource: String, Sendable {
    case fullFrame = "full"
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
    case paused(String)
    case error(String)
}

extension FinderState {
    var diagnosticName: String {
        switch self {
        case .loading: return "loading"
        case .scanning: return "scanning"
        case .candidate: return "candidate"
        case .found: return "found"
        case .paused: return "paused"
        case .error: return "error"
        }
    }
}
