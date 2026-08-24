import CoreImage
import CoreVideo
import XCTest
@testable import GolfBallFinder

final class ColorAssistEngineTests: XCTestCase {
    func testAllModesProduceFiniteScoresAndACompleteOrder() {
        let image = syntheticGrassWithWhiteBall()
        let engine = ColorAssistEngine()

        for mode in ColorAssistMode.allCases {
            guard let analysis = engine.analyze(
                image: image,
                mode: mode,
                tileRegions: AppConfig.searchTiles,
                includePreview: false
            ) else {
                return XCTFail("Color Assist failed for \(mode.rawValue)")
            }
            XCTAssertEqual(analysis.tileScores.count, AppConfig.searchTiles.count)
            XCTAssertEqual(Set(analysis.selectedTileOrder), Set(AppConfig.searchTiles.indices))
            XCTAssertTrue(analysis.tileScores.allSatisfy {
                $0.score.isFinite && $0.score >= 0 && $0.score <= 1
            })
        }
    }

    func testWhiteSaliencyPrioritizesTopLeftBallAndCreatesDebugPreviews() {
        let engine = ColorAssistEngine()
        let analysis = engine.analyze(
            image: syntheticGrassWithWhiteBall(),
            mode: .whiteBallSaliency,
            tileRegions: AppConfig.searchTiles,
            includePreview: true
        )

        XCTAssertEqual(analysis?.selectedTileOrder.first, 0)
        XCTAssertNotNil(analysis?.preview)
    }

    func testRawDebugPreviewPreservesGreenRGBAndPortraitDimensions() {
        let extent = CGRect(x: 0, y: 0, width: 180, height: 320)
        let green = CIImage(color: CIColor(red: 0.05, green: 0.80, blue: 0.10))
            .cropped(to: extent)
        let analysis = ColorAssistEngine().analyze(
            image: green,
            mode: .raw,
            tileRegions: AppConfig.searchTiles,
            includePreview: true
        )

        guard let raw = analysis?.preview?.raw,
              let pixel = centerPixel(of: raw) else {
            return XCTFail("missing raw debug preview")
        }
        XCTAssertEqual(raw.width, 180)
        XCTAssertEqual(raw.height, 320)
        XCTAssertGreaterThan(pixel.green, pixel.red * 4)
        XCTAssertGreaterThan(pixel.green, pixel.blue * 4)
    }

    func testBGRA32CameraBufferIsRenderedAsRGBWithoutChannelSwap() {
        var buffer: CVPixelBuffer?
        XCTAssertEqual(
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                180,
                320,
                kCVPixelFormatType_32BGRA,
                nil,
                &buffer
            ),
            kCVReturnSuccess
        )
        guard let buffer else { return XCTFail("could not create BGRA test buffer") }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            return XCTFail("BGRA test buffer has no base address")
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<CVPixelBufferGetHeight(buffer) {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
            for x in 0..<CVPixelBufferGetWidth(buffer) {
                row[x * 4] = 20       // B
                row[x * 4 + 1] = 210 // G
                row[x * 4 + 2] = 10  // R
                row[x * 4 + 3] = 255 // A
            }
        }

        let analysis = ColorAssistEngine().analyze(
            image: CIImage(cvPixelBuffer: buffer),
            mode: .raw,
            tileRegions: AppConfig.searchTiles,
            includePreview: true
        )
        guard let raw = analysis?.preview?.raw,
              let pixel = centerPixel(of: raw) else {
            return XCTFail("missing BGRA raw debug preview")
        }
        XCTAssertGreaterThan(pixel.green, pixel.red * 5)
        XCTAssertGreaterThan(pixel.green, pixel.blue * 5)
    }

    private func syntheticGrassWithWhiteBall() -> CIImage {
        let extent = CGRect(x: 0, y: 0, width: 320, height: 180)
        let grass = CIImage(color: CIColor(red: 0, green: 0.6, blue: 0)).cropped(to: extent)
        let ball = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(
            to: CGRect(x: 24, y: 146, width: 12, height: 12)
        )
        return ball.composited(over: grass).cropped(to: extent)
    }

    private func centerPixel(of image: CGImage) -> (red: Double, green: Double, blue: Double)? {
        var bytes = [UInt8](repeating: 0, count: 4)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: &bytes,
                width: 1,
                height: 1,
                bitsPerComponent: 8,
                bytesPerRow: 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else { return nil }
        context.draw(
            image,
            in: CGRect(x: 0, y: 0, width: 1, height: 1),
            byTiling: false
        )
        return (
            red: Double(bytes[0]),
            green: Double(bytes[1]),
            blue: Double(bytes[2])
        )
    }
}
