import XCTest
@testable import GolfBallFinder

final class DiagnosticsLogicTests: XCTestCase {
    func testRateMeterUsesCompletedInferenceIntervals() {
        var meter = InferenceRateMeter(windowSeconds: 2)
        XCTAssertEqual(meter.record(at: 0), 0)
        XCTAssertEqual(meter.record(at: 0.5), 2, accuracy: 0.0001)
        XCTAssertEqual(meter.record(at: 1.0), 2, accuracy: 0.0001)
        XCTAssertEqual(meter.record(at: 3.1), 0, accuracy: 0.0001)
    }
}
