import CoreGraphics
import XCTest
@testable import GolfBallFinder

final class DetectionStabilizerTests: XCTestCase {
    func testReportedExtremeVerticalSliverCanNeverConfirm() {
        var stabilizer = DetectionStabilizer()
        let invalid = DetectionObservation(
            normalizedRect: CGRect(x: 0.996, y: 0.017, width: 0.004, height: 0.980),
            confidence: 0.99,
            className: "golf_ball",
            source: .roi,
            timestamp: 0
        )

        for index in 0..<(AppConfig.requiredHits + AppConfig.stabilizerWindow) {
            let repeated = DetectionObservation(
                normalizedRect: invalid.normalizedRect,
                confidence: invalid.confidence,
                className: invalid.className,
                source: invalid.source,
                timestamp: Double(index)
            )
            let output = stabilizer.update(candidates: [repeated])
            XCTAssertEqual(output.state, .scanning)
            XCTAssertFalse(output.shouldTriggerFeedback)
        }
    }

    func testConfirmsRepeatedSpatialCandidate() {
        var stabilizer = DetectionStabilizer()
        var last: StabilizerOutput?

        for i in 0..<3 {
            let obs = DetectionObservation(
                normalizedRect: CGRect(x: 0.45 + CGFloat(i) * 0.005, y: 0.48, width: 0.03, height: 0.03),
                confidence: 0.50,
                className: "golf_ball",
                source: .roi,
                timestamp: Double(i)
            )
            last = stabilizer.update(candidates: [obs])
        }

        guard let last else { return XCTFail("missing output") }
        if case .found = last.state {
            XCTAssertTrue(last.shouldTriggerFeedback)
        } else {
            XCTFail("expected confirmed detection")
        }
    }

    func testConfidencePoint370RemainsCandidateBelowValidatedConfirmationThreshold() {
        XCTAssertLessThanOrEqual(AppConfig.candidateMinConfidence, 0.370)
        XCTAssertGreaterThan(AppConfig.confirmedAverageConfidence, 0.370)
        var stabilizer = DetectionStabilizer()
        var output: StabilizerOutput?
        for index in 0..<AppConfig.requiredHits {
            output = stabilizer.update(candidates: [DetectionObservation(
                normalizedRect: CGRect(x: 0.45, y: 0.45, width: 0.04, height: 0.04),
                confidence: 0.370,
                className: "golf_ball",
                source: .roi,
                timestamp: Double(index) / 15
            )])
        }
        guard let output, case .candidate = output.state else {
            return XCTFail("0.370 should remain a candidate below the validated confirmation threshold")
        }
        XCTAssertFalse(output.shouldTriggerFeedback)
    }

    func testConfirmationLatencyUsesOnlyCurrentSpatialTrack() {
        var stabilizer = DetectionStabilizer()
        _ = stabilizer.update(candidates: [DetectionObservation(
            normalizedRect: CGRect(x: 0.05, y: 0.05, width: 0.04, height: 0.04),
            confidence: 0.50,
            className: "golf_ball",
            source: .fullFrame,
            timestamp: 0
        )])

        var output: StabilizerOutput?
        for timestamp in [305.600, 305.667, 305.733] {
            output = stabilizer.update(candidates: [DetectionObservation(
                normalizedRect: CGRect(x: 0.75, y: 0.75, width: 0.04, height: 0.04),
                confidence: 0.50,
                className: "golf_ball",
                source: .roi,
                timestamp: timestamp
            )])
        }

        guard let output, case .found = output.state else {
            return XCTFail("new spatial track should confirm")
        }
        XCTAssertEqual(output.candidateTrackStartedAt ?? -1, 305.600, accuracy: 0.0001)
        XCTAssertEqual(output.confirmationLatencyMs ?? -1, 133, accuracy: 0.01)
        XCTAssertLessThan(output.confirmationLatencyMs ?? .infinity, 1_000)
    }

    func testOneFrameFalsePositiveDoesNotConfirm() {
        var stabilizer = DetectionStabilizer()
        let obs = DetectionObservation(
            normalizedRect: CGRect(x: 0.1, y: 0.1, width: 0.03, height: 0.03),
            confidence: 0.90,
            className: "golf_ball",
            source: .fullFrame,
            timestamp: 0
        )
        let output = stabilizer.update(candidates: [obs])
        if case .candidate = output.state {
            XCTAssertFalse(output.shouldTriggerFeedback)
        } else {
            XCTFail("a single detection must remain a candidate")
        }
    }

    func testSpatiallyInconsistentHitsDoNotConfirm() {
        var stabilizer = DetectionStabilizer()
        let positions: [CGFloat] = [0.05, 0.45, 0.85]
        var output: StabilizerOutput?

        for (index, position) in positions.enumerated() {
            let observation = DetectionObservation(
                normalizedRect: CGRect(x: position, y: position, width: 0.03, height: 0.03),
                confidence: 0.9,
                className: "golf_ball",
                source: .fullFrame,
                timestamp: Double(index)
            )
            output = stabilizer.update(candidates: [observation])
        }

        guard let output else { return XCTFail("missing output") }
        if case .found = output.state {
            XCTFail("distant one-frame candidates must not form a confirmed cluster")
        }
        XCTAssertFalse(output.shouldTriggerFeedback)
    }

    func testIntermittentFalsePositiveDoesNotConfirm() {
        var stabilizer = DetectionStabilizer()
        let observation = DetectionObservation(
            normalizedRect: CGRect(x: 0.45, y: 0.45, width: 0.03, height: 0.03),
            confidence: 0.95,
            className: "golf_ball",
            source: .fullFrame,
            timestamp: 0
        )
        let sequence: [[DetectionObservation]] = [[observation], [], [], [observation], []]

        for candidates in sequence {
            let output = stabilizer.update(candidates: candidates)
            if case .found = output.state {
                XCTFail("two intermittent hits in the five-frame window must not confirm")
            }
            XCTAssertFalse(output.shouldTriggerFeedback)
        }
    }

    func testMultipleCandidatesPreferSpatiallyConsistentTrack() {
        var stabilizer = DetectionStabilizer()
        let near = CGRect(x: 0.30, y: 0.30, width: 0.04, height: 0.04)
        let far = CGRect(x: 0.82, y: 0.08, width: 0.04, height: 0.04)

        for index in 0..<3 {
            let tracked = DetectionObservation(
                normalizedRect: near.offsetBy(dx: CGFloat(index) * 0.003, dy: 0),
                confidence: 0.50,
                className: "golf_ball",
                source: .tile,
                timestamp: Double(index)
            )
            let distractor = DetectionObservation(
                normalizedRect: far,
                confidence: 0.98,
                className: "golf_ball",
                source: .fullFrame,
                timestamp: Double(index)
            )
            let output = stabilizer.update(candidates: index == 0 ? [tracked] : [distractor, tracked])
            if index == 2 {
                guard case .found(let confirmed) = output.state else {
                    return XCTFail("spatially consistent candidate should confirm")
                }
                XCTAssertLessThan(confirmed.center.x, 0.5)
                XCTAssertTrue(output.shouldTriggerFeedback)
            }
        }
    }

    func testResetRequiresFreshConfirmationAndRetriggersOnce() {
        var stabilizer = DetectionStabilizer()
        let observation = DetectionObservation(
            normalizedRect: CGRect(x: 0.50, y: 0.50, width: 0.03, height: 0.03),
            confidence: 0.50,
            className: "golf_ball",
            source: .roi,
            timestamp: 0
        )
        for _ in 0..<AppConfig.requiredHits {
            _ = stabilizer.update(candidates: [observation])
        }

        stabilizer.reset()
        let firstAfterReset = stabilizer.update(candidates: [observation])
        guard case .candidate = firstAfterReset.state else {
            return XCTFail("reset must discard prior confirmation history")
        }
        XCTAssertFalse(firstAfterReset.shouldTriggerFeedback)

        var reconfirmed = firstAfterReset
        for _ in 1..<AppConfig.requiredHits {
            reconfirmed = stabilizer.update(candidates: [observation])
        }
        guard case .found = reconfirmed.state else { return XCTFail("fresh hits should reconfirm") }
        XCTAssertTrue(reconfirmed.shouldTriggerFeedback)

        let repeated = stabilizer.update(candidates: [observation])
        XCTAssertFalse(repeated.shouldTriggerFeedback, "continuous found state must not repeat haptic/audio")
    }
}

extension DetectionStabilizerTests {
    func testFoundPersistenceDoesNotJumpToUnrelatedCandidate() {
        var stabilizer = DetectionStabilizer()
        let confirmedRect = CGRect(x: 0.45, y: 0.45, width: 0.04, height: 0.04)

        for i in 0..<3 {
            let obs = DetectionObservation(
                normalizedRect: confirmedRect.offsetBy(dx: CGFloat(i) * 0.002, dy: 0),
                confidence: 0.50,
                className: "golf_ball",
                source: .roi,
                timestamp: Double(i)
            )
            _ = stabilizer.update(candidates: [obs])
        }

        let unrelated = DetectionObservation(
            normalizedRect: CGRect(x: 0.82, y: 0.12, width: 0.04, height: 0.04),
            confidence: 0.95,
            className: "golf_ball",
            source: .fullFrame,
            timestamp: 4
        )
        let output = stabilizer.update(candidates: [unrelated])

        guard case .found(let persisted) = output.state else {
            return XCTFail("expected short found persistence")
        }
        XCTAssertLessThan(persisted.center.x, 0.6, "persisted found location must stay on the previously confirmed ball")
        XCTAssertFalse(output.shouldTriggerFeedback)
    }
}
