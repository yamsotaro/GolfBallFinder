import CoreGraphics
import XCTest
@testable import GolfBallFinder

final class GeometryTests: XCTestCase {
    func testValidatesUltralyticsTopLeftNormalizedBox() {
        let rect = CGRect(x: 0.40, y: 0.45, width: 0.05, height: 0.04)
        assertRectEqual(
            validatedNormalizedTopLeftDetectionRect(
                rect,
                imageSize: CGSize(width: 720, height: 1280)
            ),
            rect
        )
    }

    func testAcceptsValidBoxesOnNormalizedFrameEdges() {
        let imageSize = CGSize(width: 720, height: 1280)
        let boxes = [
            CGRect(x: 0, y: 0, width: 0.02, height: 0.02),
            CGRect(x: 0.98, y: 0.98, width: 0.02, height: 0.02),
        ]

        for box in boxes {
            assertRectEqual(
                validatedNormalizedTopLeftDetectionRect(box, imageSize: imageSize),
                box
            )
        }
    }

    func testRejectsReportedExtremeVerticalSliver() {
        let reported = CGRect(x: 0.996, y: 0.017, width: 0.004, height: 0.980)
        XCTAssertNil(
            validatedNormalizedTopLeftDetectionRect(
                reported,
                imageSize: CGSize(width: 720, height: 1280)
            )
        )
    }

    func testRejectsExtremeHorizontalSliver() {
        let reported = CGRect(x: 0.017, y: 0.996, width: 0.980, height: 0.004)
        XCTAssertNil(
            validatedNormalizedTopLeftDetectionRect(
                reported,
                imageSize: CGSize(width: 720, height: 1280)
            )
        )
    }

    func testRejectsNegativeWidthBeforeCGRectStandardization() {
        let raw = CGRect(x: 0.2, y: 0.2, width: -0.1, height: 0.1)
        XCTAssertLessThan(raw.size.width, 0)
        XCTAssertNil(validatedNormalizedTopLeftDetectionRect(raw))
    }

    func testRejectsNegativeHeightBeforeCGRectStandardization() {
        let raw = CGRect(x: 0.2, y: 0.2, width: 0.1, height: -0.1)
        XCTAssertLessThan(raw.size.height, 0)
        XCTAssertNil(validatedNormalizedTopLeftDetectionRect(raw))
    }

    func testRejectsNonFiniteZeroSizedAndOutOfBoundsBoxesBeforeClamping() {
        let invalid: [CGRect] = [
            CGRect(x: .nan, y: 0.2, width: 0.1, height: 0.1),
            CGRect(x: 0.2, y: .infinity, width: 0.1, height: 0.1),
            CGRect(x: 0.2, y: 0.2, width: .infinity, height: 0.1),
            CGRect(x: 0.2, y: 0.2, width: 0.1, height: .nan),
            CGRect(x: 0.2, y: 0.2, width: 0, height: 0.1),
            CGRect(x: 0.2, y: 0.2, width: 0.1, height: 0),
            CGRect(x: -0.1, y: 0.2, width: 0.2, height: 0.2),
            CGRect(x: 0.9, y: 0.2, width: 0.2, height: 0.2),
        ]
        for rect in invalid {
            XCTAssertNil(validatedNormalizedTopLeftDetectionRect(rect), "accepted \(rect)")
        }
    }

    func testValidFullTileAndROILocalBoxesAreAcceptedAndMapToSameFrameBox() {
        let expected = CGRect(x: 0.40, y: 0.40, width: 0.04, height: 0.04)
        let cases: [(name: String, crop: CGRect, local: CGRect)] = [
            (
                name: "full",
                crop: CGRect(x: 0, y: 0, width: 1, height: 1),
                local: expected
            ),
            (
                name: "tile",
                crop: CGRect(x: 0.20, y: 0.10, width: 0.50, height: 0.50),
                local: CGRect(x: 0.40, y: 0.60, width: 0.08, height: 0.08)
            ),
            (
                name: "roi",
                crop: CGRect(x: 0.35, y: 0.35, width: 0.20, height: 0.20),
                local: CGRect(x: 0.25, y: 0.25, width: 0.20, height: 0.20)
            ),
        ]

        for testCase in cases {
            let cropImageSize = CGSize(
                width: 1_000 * testCase.crop.width,
                height: 1_000 * testCase.crop.height
            )
            guard let local = validatedNormalizedTopLeftDetectionRect(
                testCase.local,
                imageSize: cropImageSize
            ) else {
                XCTFail("rejected valid \(testCase.name) local bbox")
                continue
            }
            let mapped = mapCropRectToFullFrame(local, cropRegion: testCase.crop)
            assertRectEqual(
                validatedNormalizedTopLeftDetectionRect(
                    mapped,
                    imageSize: CGSize(width: 1_000, height: 1_000)
                ),
                expected
            )
        }
    }

    func testCropMappingToFullFrame() {
        let crop = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        let local = CGRect(x: 0.2, y: 0.4, width: 0.1, height: 0.2)
        let mapped = mapCropRectToFullFrame(local, cropRegion: crop)

        XCTAssertEqual(mapped.minX, 0.35, accuracy: 0.0001)
        XCTAssertEqual(mapped.minY, 0.45, accuracy: 0.0001)
        XCTAssertEqual(mapped.width, 0.05, accuracy: 0.0001)
        XCTAssertEqual(mapped.height, 0.10, accuracy: 0.0001)
    }

    func testFullFrameMapsCenterAndAllFourCornersWithoutChangingCoordinates() {
        let boxes = [
            CGRect(x: 0.45, y: 0.45, width: 0.10, height: 0.10),
            CGRect(x: 0.00, y: 0.00, width: 0.05, height: 0.05),
            CGRect(x: 0.95, y: 0.00, width: 0.05, height: 0.05),
            CGRect(x: 0.00, y: 0.95, width: 0.05, height: 0.05),
            CGRect(x: 0.95, y: 0.95, width: 0.05, height: 0.05),
        ]
        for box in boxes {
            assertRectEqual(
                mapCropRectToFullFrame(box, cropRegion: CGRect(x: 0, y: 0, width: 1, height: 1)),
                box
            )
        }
    }

    func testOverlappingTilesMapSameCameraFrameBoxToSameLocation() {
        let expected = CGRect(x: 0.40, y: 0.40, width: 0.04, height: 0.04)
        let topLeftTile = AppConfig.searchTiles[0]
        let centerTile = AppConfig.searchTiles[4]

        func localRect(in crop: CGRect) -> CGRect {
            CGRect(
                x: (expected.minX - crop.minX) / crop.width,
                y: (expected.minY - crop.minY) / crop.height,
                width: expected.width / crop.width,
                height: expected.height / crop.height
            )
        }

        assertRectEqual(
            mapCropRectToFullFrame(localRect(in: topLeftTile), cropRegion: topLeftTile),
            expected
        )
        assertRectEqual(
            mapCropRectToFullFrame(localRect(in: centerTile), cropRegion: centerTile),
            expected
        )
    }

    func testROINearBottomRightEdgeMapsLocalBoxToFullFrame() {
        let roi = CGRect(x: 0.66, y: 0.66, width: 0.34, height: 0.34)
        let local = CGRect(x: 0.80, y: 0.85, width: 0.10, height: 0.08)
        let mapped = mapCropRectToFullFrame(local, cropRegion: roi)

        XCTAssertEqual(mapped.minX, 0.932, accuracy: 0.0001)
        XCTAssertEqual(mapped.minY, 0.949, accuracy: 0.0001)
        XCTAssertEqual(mapped.width, 0.034, accuracy: 0.0001)
        XCTAssertEqual(mapped.height, 0.0272, accuracy: 0.0001)
        XCTAssertLessThanOrEqual(mapped.maxX, 1)
        XCTAssertLessThanOrEqual(mapped.maxY, 1)
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

    func testPortraitPreviewPlacesValidCenterBoxAtViewCenter() {
        let mapped = aspectFillDisplayRect(
            normalizedRect: CGRect(x: 0.45, y: 0.45, width: 0.10, height: 0.10),
            imageSize: CGSize(width: 720, height: 1280),
            viewSize: CGSize(width: 393, height: 852)
        )

        XCTAssertEqual(mapped.midX, 196.5, accuracy: 0.0001)
        XCTAssertEqual(mapped.midY, 426, accuracy: 0.0001)
        XCTAssertGreaterThan(mapped.width, 0)
        XCTAssertGreaterThan(mapped.height, 0)
    }

    func testPortraitPreviewTopAndBottomScreenEdgesKeepTopLeftOrigin() {
        let top = aspectFillDisplayRect(
            normalizedRect: CGRect(x: 0.48, y: 0.00, width: 0.04, height: 0.04),
            imageSize: CGSize(width: 720, height: 1280),
            viewSize: CGSize(width: 393, height: 852)
        )
        let bottom = aspectFillDisplayRect(
            normalizedRect: CGRect(x: 0.48, y: 0.96, width: 0.04, height: 0.04),
            imageSize: CGSize(width: 720, height: 1280),
            viewSize: CGSize(width: 393, height: 852)
        )

        XCTAssertLessThan(top.midY, bottom.midY)
        XCTAssertLessThanOrEqual(top.minY, 1)
        XCTAssertGreaterThanOrEqual(bottom.maxY, 851)
    }

    func testOversizedROIClipsToWholeFrame() {
        let roi = CGRect(x: -2, y: -1, width: 4, height: 3).clampedPreservingSize()
        assertRectEqual(roi, CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    func testCropMappingClipsNegativeLocalCoordinates() {
        let mapped = mapCropRectToFullFrame(
            CGRect(x: -0.5, y: -0.25, width: 0.75, height: 0.50),
            cropRegion: CGRect(x: 0, y: 0, width: 0.4, height: 0.4)
        )
        XCTAssertEqual(mapped.minX, 0, accuracy: 0.0001)
        XCTAssertEqual(mapped.minY, 0, accuracy: 0.0001)
        XCTAssertEqual(mapped.maxX, 0.1, accuracy: 0.0001)
        XCTAssertEqual(mapped.maxY, 0.1, accuracy: 0.0001)
    }

    func testAspectFillMapsFrameEdgesBeyondCroppedViewEdges() {
        let leftEdge = aspectFillDisplayRect(
            normalizedRect: CGRect(x: 0, y: 0.4, width: 0.02, height: 0.2),
            imageSize: CGSize(width: 720, height: 1280),
            viewSize: CGSize(width: 390, height: 844)
        )
        let rightEdge = aspectFillDisplayRect(
            normalizedRect: CGRect(x: 0.98, y: 0.4, width: 0.02, height: 0.2),
            imageSize: CGSize(width: 720, height: 1280),
            viewSize: CGSize(width: 390, height: 844)
        )
        XCTAssertLessThan(leftEdge.minX, 0)
        XCTAssertGreaterThan(rightEdge.maxX, 390)
        XCTAssertEqual(leftEdge.midY, rightEdge.midY, accuracy: 0.0001)
    }

    func testZeroImageOrViewSizeProducesZeroDisplayRect() {
        assertRectEqual(
            aspectFillDisplayRect(
                normalizedRect: CGRect(x: 0.2, y: 0.2, width: 0.1, height: 0.1),
                imageSize: .zero,
                viewSize: CGSize(width: 390, height: 844)
            ),
            .zero
        )
    }

    private func assertRectEqual(
        _ actual: CGRect,
        _ expected: CGRect,
        accuracy: CGFloat = 1e-9,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.origin.x, expected.origin.x, accuracy: accuracy, file: file, line: line)
        XCTAssertEqual(actual.origin.y, expected.origin.y, accuracy: accuracy, file: file, line: line)
        XCTAssertEqual(actual.size.width, expected.size.width, accuracy: accuracy, file: file, line: line)
        XCTAssertEqual(actual.size.height, expected.size.height, accuracy: accuracy, file: file, line: line)
    }

    private func assertRectEqual(
        _ actual: CGRect?,
        _ expected: CGRect,
        accuracy: CGFloat = 1e-9,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard let actual = actual else {
            XCTFail("expected CGRect, got nil", file: file, line: line)
            return
        }
        assertRectEqual(actual, expected, accuracy: accuracy, file: file, line: line)
    }
}
