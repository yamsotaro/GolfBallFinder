import Foundation

struct InferenceFrameStatistics: Equatable, Sendable {
    let received: Int
    let admitted: Int
    let droppedBusy: Int
    let droppedCadence: Int
}

/// Video-queue-only admission gate. It never stores a pending frame: frames received while an
/// invocation is running are discarded, so the next admitted frame is always a fresh camera frame.
struct InferenceFrameGate: Sendable {
    private var busy = false
    private var lastAdmittedTimestamp: TimeInterval?
    private var received = 0
    private var admitted = 0
    private var droppedBusy = 0
    private var droppedCadence = 0

    mutating func admit(timestamp: TimeInterval, targetFPS: Double, detectorReady: Bool) -> Bool {
        received += 1

        guard detectorReady else {
            droppedCadence += 1
            return false
        }
        guard !busy else {
            droppedBusy += 1
            return false
        }

        let safeFPS = max(1, targetFPS)
        if let lastAdmittedTimestamp {
            if timestamp >= lastAdmittedTimestamp,
               timestamp - lastAdmittedTimestamp + 1e-9 < 1.0 / safeFPS {
                droppedCadence += 1
                return false
            }
            // A camera/session restart can move the presentation timestamp backwards. In that
            // case admit immediately instead of waiting for the old timestamp to be reached.
        }

        busy = true
        lastAdmittedTimestamp = timestamp
        admitted += 1
        return true
    }

    mutating func complete() {
        busy = false
    }

    var statistics: InferenceFrameStatistics {
        InferenceFrameStatistics(
            received: received,
            admitted: admitted,
            droppedBusy: droppedBusy,
            droppedCadence: droppedCadence
        )
    }
}

struct ThermalInferencePolicy: Equatable, Sendable {
    let processedFPS: Double

    static func current(for state: ProcessInfo.ThermalState) -> ThermalInferencePolicy {
        switch state {
        case .nominal:
            return ThermalInferencePolicy(processedFPS: AppConfig.processedFPS)
        case .fair:
            return ThermalInferencePolicy(processedFPS: AppConfig.fairThermalFPS)
        case .serious:
            return ThermalInferencePolicy(processedFPS: AppConfig.seriousThermalFPS)
        case .critical:
            return ThermalInferencePolicy(processedFPS: AppConfig.criticalThermalFPS)
        @unknown default:
            return ThermalInferencePolicy(processedFPS: AppConfig.seriousThermalFPS)
        }
    }
}

extension ProcessInfo.ThermalState {
    var diagnosticName: String {
        switch self {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
