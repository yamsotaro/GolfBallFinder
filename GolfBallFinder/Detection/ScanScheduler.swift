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
    private var tileVisitCount = 0
    private var activeTileOrder: [Int]?
    private var pendingTileOrder: [Int]?
    private var useFullNext = true
    private var lockedROI: CGRect?
    private var remainingLockFrames = 0
    private var consecutiveLockMisses = 0

    var isROILocked: Bool { lockedROI != nil }

    var currentTileOrder: [Int] {
        activeTileOrder ?? Array(AppConfig.searchTiles.indices)
    }

    mutating func reset() {
        self = ScanScheduler()
    }

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
        let position = tileVisitCount % tiles.count
        if position == 0, let pendingTileOrder {
            activeTileOrder = pendingTileOrder
            self.pendingTileOrder = nil
        }
        let baselineOrder = Array(tiles.indices)
        let order = activeTileOrder ?? baselineOrder
        let selectedIndex = order[position]
        tileVisitCount += 1
        return .tile(tiles[selectedIndex])
    }

    /// Updates only the order of the next broad-search tile cycle. A cycle always visits every
    /// tile exactly once, so changing saliency every frame cannot starve a low-scoring tile.
    /// Passing `enabled == false` restores the original round-robin sequence at the same visit
    /// count; this makes the default/off behavior identical to the pre-experiment scheduler.
    @discardableResult
    mutating func updateTilePriorities(scores: [Double]?, enabled: Bool) -> [Int] {
        let baseline = Array(AppConfig.searchTiles.indices)
        guard enabled,
              let scores,
              scores.count == baseline.count,
              scores.allSatisfy(\.isFinite) else {
            activeTileOrder = nil
            pendingTileOrder = nil
            return baseline
        }

        let ordered = baseline.sorted { left, right in
            if scores[left] == scores[right] { return left < right }
            return scores[left] > scores[right]
        }
        if baseline.isEmpty || tileVisitCount % baseline.count == 0 {
            activeTileOrder = ordered
            pendingTileOrder = nil
        } else {
            pendingTileOrder = ordered
        }
        return ordered
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
