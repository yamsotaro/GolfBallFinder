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
}
