import CoreGraphics
import Foundation

enum ScanRegion: Equatable, Sendable {
    case full
    case tile(CGRect)
    case roi(CGRect)

    var normalizedRect: CGRect {
        switch self {
        case .full:
            return CGRect(x: 0, y: 0, width: 1, height: 1)
        case .tile(let rect), .roi(let rect):
            return rect
        }
    }

    var source: ScanSource {
        switch self {
        case .full: return .fullFrame
        case .tile: return .tile
        case .roi: return .roi
        }
    }
}

/// Adaptive search pattern:
/// 1. Alternate full frame and an overlapping tile while searching.
/// 2. As soon as a candidate appears, spend the next frames repeatedly on a compact ROI.
/// 3. Return to broad search if the candidate is lost.
struct ScanScheduler: Sendable {
    private var tileIndex = 0
    private var useFullNext = true
    private var lockedROI: CGRect?
    private var remainingLockFrames = 0
    private var consecutiveLockMisses = 0

    mutating func nextRegion() -> ScanRegion {
        if let roi = lockedROI, remainingLockFrames > 0 {
            remainingLockFrames -= 1
            return .roi(roi)
        }

        if lockedROI != nil {
            unlock()
        }

        if useFullNext {
            useFullNext = false
            return .full
        }

        useFullNext = true
        let tiles = AppConfig.searchTiles
        guard !tiles.isEmpty else { return .full }
        let tile = tiles[tileIndex % tiles.count]
        tileIndex = (tileIndex + 1) % tiles.count
        return .tile(tile)
    }

    mutating func lock(on detection: DetectionObservation) {
        lockedROI = detection.normalizedRect.expandedAroundCenter(
            multiplier: AppConfig.roiExpansion,
            minimumSide: AppConfig.minimumROISide
        )
        remainingLockFrames = AppConfig.roiLockFrames
        consecutiveLockMisses = 0
    }

    mutating func updateLockedROI(with detection: DetectionObservation?) {
        guard lockedROI != nil else { return }

        if let detection {
            let newROI = detection.normalizedRect.expandedAroundCenter(
                multiplier: AppConfig.roiExpansion,
                minimumSide: AppConfig.minimumROISide
            )
            if let old = lockedROI {
                // Damp movement to avoid an unstable crop jumping around false-positive edges.
                let blended = CGRect(
                    x: old.minX * 0.55 + newROI.minX * 0.45,
                    y: old.minY * 0.55 + newROI.minY * 0.45,
                    width: old.width * 0.55 + newROI.width * 0.45,
                    height: old.height * 0.55 + newROI.height * 0.45
                ).clampedPreservingSize()
                lockedROI = blended
            } else {
                lockedROI = newROI
            }
            remainingLockFrames = AppConfig.roiLockFrames
            consecutiveLockMisses = 0
        } else {
            consecutiveLockMisses += 1
            if consecutiveLockMisses > AppConfig.roiLostTolerance {
                unlock()
            }
        }
    }

    mutating func unlock() {
        lockedROI = nil
        remainingLockFrames = 0
        consecutiveLockMisses = 0
    }
}
