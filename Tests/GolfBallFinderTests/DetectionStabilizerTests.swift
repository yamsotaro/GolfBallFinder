import CoreGraphics
import XCTest
@testable import GolfBallFinder

final class DetectionStabilizerTests: XCTestCase {
    func testConfirmsRepeatedSpatialCandidate() {
        var stabilizer = DetectionStabilizer()
        var last: StabilizerOutput?

        for i in 0..<3 {
            let obs = DetectionObservation(
                normalizedRect: CGRect(x: 0.45 + CGFloat(i) * 0.005, y: 0.48, width: 0.03, height: 0.03),
                confidence: 0.35,
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
}

extension DetectionStabilizerTests {
    func testFoundPersistenceDoesNotJumpToUnrelatedCandidate() {
        var stabilizer = DetectionStabilizer()
        let confirmedRect = CGRect(x: 0.45, y: 0.45, width: 0.04, height: 0.04)

        for i in 0..<3 {
            let obs = DetectionObservation(
                normalizedRect: confirmedRect.offsetBy(dx: CGFloat(i) * 0.002, dy: 0),
                confidence: 0.40,
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
