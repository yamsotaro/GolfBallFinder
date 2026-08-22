import CoreImage
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

    private func syntheticGrassWithWhiteBall() -> CIImage {
        let extent = CGRect(x: 0, y: 0, width: 320, height: 180)
        let grass = CIImage(color: CIColor(red: 0, green: 0.6, blue: 0)).cropped(to: extent)
        let ball = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(
            to: CGRect(x: 24, y: 146, width: 12, height: 12)
        )
        return ball.composited(over: grass).cropped(to: extent)
    }
}
