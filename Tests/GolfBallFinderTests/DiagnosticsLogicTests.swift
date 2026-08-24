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


    func testTimingUsesStabilizerTrackLatencyInsteadOfOldCandidateWallTime() {
        var timing = DetectionTimingState()
        timing.beginScene(at: 10)
        _ = timing.update(
            state: .candidate(observation(at: 0, x: 0.1)),
            candidateTrackStartedAt: 0,
            confirmationLatencyMs: nil,
            at: 11
        )
        _ = timing.update(
            state: .candidate(observation(at: 305.6, x: 0.8)),
            candidateTrackStartedAt: 305.6,
            confirmationLatencyMs: nil,
            at: 315.6
        )
        let confirmed = timing.update(
            state: .found(observation(at: 305.733, x: 0.8)),
            candidateTrackStartedAt: 305.6,
            confirmationLatencyMs: 133,
            at: 315.733
        )

        XCTAssertTrue(confirmed)
        XCTAssertEqual(timing.candidateToConfirmedMs ?? -1, 133)
        XCTAssertNotEqual(timing.candidateToConfirmedMs ?? -1, 305_733)
    }

    func testReSearchResetClearsCandidateConfirmedAndRestartsActiveSceneClock() {
        var timing = DetectionTimingState()
        timing.beginScene(at: 10)
        _ = timing.update(
            state: .candidate(observation(at: 20, x: 0.4)),
            candidateTrackStartedAt: 20,
            confirmationLatencyMs: nil,
            at: 20
        )
        _ = timing.update(
            state: .found(observation(at: 20.14, x: 0.4)),
            candidateTrackStartedAt: 20,
            confirmationLatencyMs: 140,
            at: 20.14
        )

        timing.resetSearch(at: 30, sceneIsActive: true)

        XCTAssertNil(timing.candidateTrackStartedAt)
        XCTAssertNil(timing.candidateToConfirmedMs)
        XCTAssertNil(timing.sceneStartToFirstCandidateMs)
        XCTAssertNil(timing.sceneStartToConfirmedMs)
        XCTAssertFalse(timing.previousWasFound)
        XCTAssertEqual(timing.sceneBeganAt ?? -1, 30)
    }

    private func observation(at timestamp: TimeInterval, x: CGFloat) -> DetectionObservation {
        DetectionObservation(
            normalizedRect: CGRect(x: x, y: 0.4, width: 0.04, height: 0.04),
            confidence: 0.4,
            className: "golf_ball",
            source: .roi,
            timestamp: timestamp
        )
    }
}
