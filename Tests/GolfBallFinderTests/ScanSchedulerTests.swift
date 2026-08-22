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
}
