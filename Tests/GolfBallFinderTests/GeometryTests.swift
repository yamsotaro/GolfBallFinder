import CoreGraphics
import XCTest
@testable import GolfBallFinder

final class GeometryTests: XCTestCase {
    func testCropMappingToFullFrame() {
        let crop = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        let local = CGRect(x: 0.2, y: 0.4, width: 0.1, height: 0.2)
        let mapped = mapCropRectToFullFrame(local, cropRegion: crop)

        XCTAssertEqual(mapped.minX, 0.35, accuracy: 0.0001)
        XCTAssertEqual(mapped.minY, 0.45, accuracy: 0.0001)
        XCTAssertEqual(mapped.width, 0.05, accuracy: 0.0001)
        XCTAssertEqual(mapped.height, 0.10, accuracy: 0.0001)
    }

    func testExpandedROIStaysInsideUnitRect() {
        let ball = CGRect(x: 0.94, y: 0.92, width: 0.03, height: 0.03)
        let roi = ball.expandedAroundCenter(multiplier: 5, minimumSide: 0.34)
        XCTAssertGreaterThanOrEqual(roi.minX, 0)
        XCTAssertGreaterThanOrEqual(roi.minY, 0)
        XCTAssertLessThanOrEqual(roi.maxX, 1)
        XCTAssertLessThanOrEqual(roi.maxY, 1)
        XCTAssertEqual(roi.width, 0.34, accuracy: 0.0001)
        XCTAssertEqual(roi.height, 0.34, accuracy: 0.0001)
    }

    func testCropMappingClampsDetectionAtFrameEdge() {
        let crop = CGRect(x: 0.8, y: 0.8, width: 0.2, height: 0.2)
        let local = CGRect(x: 0.9, y: 0.9, width: 0.4, height: 0.4)
        let mapped = mapCropRectToFullFrame(local, cropRegion: crop)

        XCTAssertEqual(mapped.minX, 0.98, accuracy: 0.0001)
        XCTAssertEqual(mapped.minY, 0.98, accuracy: 0.0001)
        XCTAssertEqual(mapped.maxX, 1.0, accuracy: 0.0001)
        XCTAssertEqual(mapped.maxY, 1.0, accuracy: 0.0001)
    }

    func testAspectFillMappingAccountsForHorizontalCrop() {
        let mapped = aspectFillDisplayRect(
            normalizedRect: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5),
            imageSize: CGSize(width: 720, height: 1280),
            viewSize: CGSize(width: 390, height: 844)
        )

        XCTAssertEqual(mapped.midX, 195, accuracy: 0.0001)
        XCTAssertEqual(mapped.midY, 422, accuracy: 0.0001)
        XCTAssertEqual(mapped.height, 422, accuracy: 0.0001)
        XCTAssertEqual(mapped.width, 237.375, accuracy: 0.0001)
    }
}
