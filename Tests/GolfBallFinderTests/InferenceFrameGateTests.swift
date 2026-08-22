import Foundation
import XCTest
@testable import GolfBallFinder

final class InferenceFrameGateTests: XCTestCase {
    func testBusyFramesAreDroppedWithoutPendingBacklog() {
        var gate = InferenceFrameGate()
        XCTAssertTrue(gate.admit(timestamp: 1.0, targetFPS: 20, detectorReady: true))
        XCTAssertFalse(gate.admit(timestamp: 1.1, targetFPS: 20, detectorReady: true))
        XCTAssertFalse(gate.admit(timestamp: 1.2, targetFPS: 20, detectorReady: true))

        gate.complete()
        XCTAssertTrue(gate.admit(timestamp: 1.3, targetFPS: 20, detectorReady: true))
        XCTAssertEqual(
            gate.statistics,
            InferenceFrameStatistics(received: 4, admitted: 2, droppedBusy: 2, droppedCadence: 0)
        )
    }

    func testCadenceLimitAndCameraTimestampRestart() {
        var gate = InferenceFrameGate()
        XCTAssertTrue(gate.admit(timestamp: 10.0, targetFPS: 10, detectorReady: true))
        gate.complete()
        XCTAssertFalse(gate.admit(timestamp: 10.05, targetFPS: 10, detectorReady: true))
        XCTAssertTrue(gate.admit(timestamp: 10.10, targetFPS: 10, detectorReady: true))
        gate.complete()
        XCTAssertTrue(gate.admit(timestamp: 0.01, targetFPS: 10, detectorReady: true))
    }

    func testDetectorNotReadyNeverCreatesBusyState() {
        var gate = InferenceFrameGate()
        XCTAssertFalse(gate.admit(timestamp: 0, targetFPS: 20, detectorReady: false))
        XCTAssertTrue(gate.admit(timestamp: 0.1, targetFPS: 20, detectorReady: true))
    }

    func testThermalPolicyReducesLoadMonotonically() {
        let nominal = ThermalInferencePolicy.current(for: .nominal).processedFPS
        let fair = ThermalInferencePolicy.current(for: .fair).processedFPS
        let serious = ThermalInferencePolicy.current(for: .serious).processedFPS
        let critical = ThermalInferencePolicy.current(for: .critical).processedFPS

        XCTAssertGreaterThan(nominal, fair)
        XCTAssertGreaterThan(fair, serious)
        XCTAssertGreaterThan(serious, critical)
        XCTAssertGreaterThanOrEqual(critical, 1)
    }
}
