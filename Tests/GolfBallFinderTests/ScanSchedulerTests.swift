import CoreGraphics
import XCTest
@testable import GolfBallFinder

final class ScanSchedulerTests: XCTestCase {
    func testBroadSearchAlternatesFullAndTile() {
        var scheduler = ScanScheduler()
        XCTAssertEqual(scheduler.nextRegion(), .full)
        if case .tile = scheduler.nextRegion() {
            // expected
        } else {
            XCTFail("second region should be a tile")
        }
        XCTAssertEqual(scheduler.nextRegion(), .full)
    }

    func testCandidateLocksROI() {
        var scheduler = ScanScheduler()
        let obs = DetectionObservation(
            normalizedRect: CGRect(x: 0.4, y: 0.4, width: 0.04, height: 0.04),
            confidence: 0.5,
            className: "golf_ball",
            source: .tile,
            timestamp: 0
        )
        scheduler.lock(on: obs)
        if case .roi(let roi) = scheduler.nextRegion() {
            XCTAssertTrue(roi.contains(obs.center))
        } else {
            XCTFail("expected ROI after lock")
        }
    }

    func testROILockReleasesAfterConfiguredMissTolerance() {
        var scheduler = ScanScheduler()
        let obs = DetectionObservation(
            normalizedRect: CGRect(x: 0.4, y: 0.4, width: 0.04, height: 0.04),
            confidence: 0.5,
            className: "golf_ball",
            source: .tile,
            timestamp: 0
        )
        scheduler.lock(on: obs)

        for _ in 0...AppConfig.roiLostTolerance {
            guard case .roi = scheduler.nextRegion() else {
                return XCTFail("ROI should remain selected through the configured miss tolerance")
            }
            scheduler.updateLockedROI(with: nil)
        }

        XCTAssertEqual(scheduler.nextRegion(), .full)
    }

    func testTilesAdvanceAcrossBroadSearchCycles() {
        var scheduler = ScanScheduler()
        XCTAssertEqual(scheduler.nextRegion(), .full)
        XCTAssertEqual(scheduler.nextRegion(), .tile(AppConfig.searchTiles[0]))
        XCTAssertEqual(scheduler.nextRegion(), .full)
        XCTAssertEqual(scheduler.nextRegion(), .tile(AppConfig.searchTiles[1]))
    }

    func testColorAssistOrdersTilesByScoreWithoutStarvation() {
        var scheduler = ScanScheduler()
        let order = scheduler.updateTilePriorities(
            scores: [0.1, 0.7, 0.2, 0.9, 0.4],
            enabled: true
        )
        XCTAssertEqual(order, [3, 1, 4, 2, 0])

        var visited: [CGRect] = []
        for _ in AppConfig.searchTiles.indices {
            _ = scheduler.nextRegion()
            guard case .tile(let tile) = scheduler.nextRegion() else {
                return XCTFail("expected prioritized tile")
            }
            visited.append(tile)
        }
        XCTAssertEqual(visited, order.map { AppConfig.searchTiles[$0] })
        XCTAssertEqual(Set(visited.map(\.debugDescription)).count, AppConfig.searchTiles.count)
    }

    func testColorAssistOffPreservesBaselineRoundRobinOrder() {
        var scheduler = ScanScheduler()
        _ = scheduler.updateTilePriorities(scores: [0, 0, 0, 1, 0], enabled: false)
        XCTAssertEqual(scheduler.nextRegion(), .full)
        XCTAssertEqual(scheduler.nextRegion(), .tile(AppConfig.searchTiles[0]))
        XCTAssertEqual(scheduler.nextRegion(), .full)
        XCTAssertEqual(scheduler.nextRegion(), .tile(AppConfig.searchTiles[1]))
    }

    func testPriorityUpdateWaitsForNextCycleAndKeepsEveryCurrentTile() {
        var scheduler = ScanScheduler()
        _ = scheduler.updateTilePriorities(scores: [0.9, 0.8, 0.7, 0.6, 0.5], enabled: true)
        _ = scheduler.nextRegion()
        XCTAssertEqual(scheduler.nextRegion(), .tile(AppConfig.searchTiles[0]))

        _ = scheduler.updateTilePriorities(scores: [0.1, 0.2, 0.3, 0.4, 0.9], enabled: true)
        var restOfFirstCycle: [CGRect] = []
        for _ in 1..<AppConfig.searchTiles.count {
            _ = scheduler.nextRegion()
            guard case .tile(let tile) = scheduler.nextRegion() else {
                return XCTFail("expected tile")
            }
            restOfFirstCycle.append(tile)
        }
        XCTAssertEqual(restOfFirstCycle, Array(AppConfig.searchTiles[1...]))

        _ = scheduler.nextRegion()
        XCTAssertEqual(scheduler.nextRegion(), .tile(AppConfig.searchTiles[4]))
    }

    func testROINearEveryScreenCornerIsClippedToUnitFrame() {
        let edgeOrigins: [CGPoint] = [
            CGPoint(x: -0.03, y: -0.03),
            CGPoint(x: 0.98, y: -0.03),
            CGPoint(x: -0.03, y: 0.98),
            CGPoint(x: 0.98, y: 0.98),
        ]

        for origin in edgeOrigins {
            var scheduler = ScanScheduler()
            scheduler.lock(on: DetectionObservation(
                normalizedRect: CGRect(origin: origin, size: CGSize(width: 0.05, height: 0.05)),
                confidence: 0.5,
                className: "golf_ball",
                source: .tile,
                timestamp: 0
            ))
            guard case .roi(let roi) = scheduler.nextRegion() else {
                return XCTFail("expected ROI at edge")
            }
            XCTAssertGreaterThanOrEqual(roi.minX, 0)
            XCTAssertGreaterThanOrEqual(roi.minY, 0)
            XCTAssertLessThanOrEqual(roi.maxX, 1)
            XCTAssertLessThanOrEqual(roi.maxY, 1)
            XCTAssertGreaterThan(roi.width, 0)
            XCTAssertGreaterThan(roi.height, 0)
        }
    }

    func testUnlockReturnsToDeterministicBroadSearch() {
        var scheduler = ScanScheduler()
        _ = scheduler.nextRegion()
        scheduler.lock(on: DetectionObservation(
            normalizedRect: CGRect(x: 0.5, y: 0.5, width: 0.04, height: 0.04),
            confidence: 0.5,
            className: "golf_ball",
            source: .fullFrame,
            timestamp: 0
        ))
        scheduler.unlock()

        guard case .tile = scheduler.nextRegion() else {
            return XCTFail("unlock should resume the broad-search phase that was interrupted")
        }
        XCTAssertEqual(scheduler.nextRegion(), .full)
    }

    func testResetClearsROIAndRestartsFullFrameSequence() {
        var scheduler = ScanScheduler()
        scheduler.lock(on: DetectionObservation(
            normalizedRect: CGRect(x: 0.4, y: 0.4, width: 0.04, height: 0.04),
            confidence: 0.5,
            className: "golf_ball",
            source: .tile,
            timestamp: 0
        ))
        XCTAssertTrue(scheduler.isROILocked)

        scheduler.reset()

        XCTAssertFalse(scheduler.isROILocked)
        XCTAssertEqual(scheduler.nextRegion(), .full)
        XCTAssertEqual(scheduler.nextRegion(), .tile(AppConfig.searchTiles[0]))
    }
}
