import CoreGraphics
import Foundation

struct StabilizerOutput: Equatable, Sendable {
    let state: FinderState
    let shouldTriggerFeedback: Bool
}

/// Lightweight temporal confirmation for a single expected target.
/// It deliberately favors recall in the detector and precision in the temporal layer.
struct DetectionStabilizer: Sendable {
    private var history: [DetectionObservation?] = []
    private var lastAccepted: DetectionObservation?
    private var lastConfirmed: DetectionObservation?
    private var wasConfirmed = false
    private var persistenceRemaining = 0

    mutating func reset() {
        history.removeAll(keepingCapacity: true)
        lastAccepted = nil
        lastConfirmed = nil
        wasConfirmed = false
        persistenceRemaining = 0
    }

    mutating func update(candidates: [DetectionObservation]) -> StabilizerOutput {
        let accepted = chooseCandidate(from: candidates)
        append(accepted)

        if let accepted {
            lastAccepted = accepted
        }

        let recent = history.compactMap { $0 }
        let hits = clusteredHits(in: recent, around: accepted ?? lastAccepted)
        let averageConfidence: Float = hits.isEmpty
            ? 0
            : hits.reduce(0) { $0 + $1.confidence } / Float(hits.count)

        let isConfirmed = hits.count >= AppConfig.requiredHits &&
            averageConfidence >= AppConfig.confirmedAverageConfidence

        if isConfirmed, let best = hits.max(by: { $0.confidence < $1.confidence }) {
            persistenceRemaining = AppConfig.foundPersistenceFrames
            lastConfirmed = best
            let trigger = !wasConfirmed
            wasConfirmed = true
            return StabilizerOutput(state: .found(best), shouldTriggerFeedback: trigger)
        }

        // Keep the last *confirmed* location briefly through detector flicker. Do not let an
        // unrelated one-frame candidate move the green "found" indication during this grace period.
        if persistenceRemaining > 0, wasConfirmed, let lastConfirmed {
            persistenceRemaining -= 1
            return StabilizerOutput(state: .found(lastConfirmed), shouldTriggerFeedback: false)
        }

        wasConfirmed = false
        lastConfirmed = nil
        if let accepted {
            return StabilizerOutput(state: .candidate(accepted), shouldTriggerFeedback: false)
        }

        return StabilizerOutput(state: .scanning, shouldTriggerFeedback: false)
    }

    private mutating func append(_ observation: DetectionObservation?) {
        history.append(observation)
        if history.count > AppConfig.stabilizerWindow {
            history.removeFirst(history.count - AppConfig.stabilizerWindow)
        }
    }

    private func chooseCandidate(from candidates: [DetectionObservation]) -> DetectionObservation? {
        let filtered = candidates.filter { $0.confidence >= AppConfig.candidateMinConfidence }
        guard !filtered.isEmpty else { return nil }

        if let lastAccepted {
            let nearby = filtered.filter {
                $0.normalizedRect.centerDistance(to: lastAccepted.normalizedRect) <= AppConfig.maxCenterStep
            }
            if let bestNearby = nearby.max(by: { score($0, relativeTo: lastAccepted) < score($1, relativeTo: lastAccepted) }) {
                return bestNearby
            }
        }

        return filtered.max(by: { $0.confidence < $1.confidence })
    }

    private func score(_ detection: DetectionObservation, relativeTo previous: DetectionObservation) -> Float {
        let distance = Float(detection.normalizedRect.centerDistance(to: previous.normalizedRect))
        return detection.confidence - distance * 0.75
    }

    private func clusteredHits(
        in recent: [DetectionObservation],
        around reference: DetectionObservation?
    ) -> [DetectionObservation] {
        guard let reference else { return [] }
        return recent.filter {
            $0.normalizedRect.centerDistance(to: reference.normalizedRect) <= AppConfig.maxCenterStep
        }
    }
}
