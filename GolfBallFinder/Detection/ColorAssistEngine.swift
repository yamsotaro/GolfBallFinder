import CoreGraphics
import CoreImage
import Foundation
import Metal

enum ColorAssistMode: String, CaseIterable, Hashable, Identifiable, Sendable {
    case raw
    case golfContrast = "golf_contrast"
    case excessGreen = "excess_green"
    case whiteBallSaliency = "white_ball_saliency"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .raw: return "Raw luminance"
        case .golfContrast: return "Golf Contrast"
        case .excessGreen: return "Excess Green rejection"
        case .whiteBallSaliency: return "White-ball saliency"
        }
    }
}

struct ColorAssistTileScore: Equatable, Sendable {
    let tileIndex: Int
    let score: Double
}

/// Small CGImages rendered only while Field Diagnostics is visible. They never enter YOLO.
struct ColorAssistPreview: @unchecked Sendable {
    let raw: CGImage
    let golfContrast: CGImage
    let saliency: CGImage
}

struct ColorAssistAnalysis: @unchecked Sendable {
    let mode: ColorAssistMode
    let processingLatencyMs: Double
    let tileScores: [ColorAssistTileScore]
    let selectedTileOrder: [Int]
    let preview: ColorAssistPreview?
}

/// Low-resolution, GPU-backed visual heuristic for broad-search tile ordering.
///
/// This component is deliberately separate from GolfBallDetector. Its output is metadata and a
/// tile order only; detector input remains the original raw RGB CIImage/crop.
final class ColorAssistEngine {
    private static let debugPreviewColorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private static let luminanceKernel = CIColorKernel(source: """
        kernel vec4 luminanceMap(__sample pixel) {
            float value = dot(pixel.rgb, vec3(0.2126, 0.7152, 0.0722));
            return vec4(value, value, value, pixel.a);
        }
        """)

    private static let golfContrastKernel = CIColorKernel(source: """
        kernel vec4 golfContrast(__sample pixel) {
            float luminance = dot(pixel.rgb, vec3(0.2126, 0.7152, 0.0722));
            vec3 contrasted = vec3(
                1.12 * pixel.r + 0.04 * luminance,
                0.72 * pixel.g,
                1.08 * pixel.b + 0.03 * luminance
            );
            return vec4(clamp(contrasted, 0.0, 1.0), pixel.a);
        }
        """)

    private static let excessGreenKernel = CIColorKernel(source: """
        kernel vec4 excessGreen(__sample pixel) {
            float vegetation = clamp(2.0 * pixel.g - pixel.r - pixel.b, 0.0, 1.0);
            return vec4(vegetation, vegetation, vegetation, pixel.a);
        }
        """)

    private static let whiteBallSaliencyKernel = CIColorKernel(source: """
        kernel vec4 whiteBallSaliency(__sample pixel) {
            float luminance = dot(pixel.rgb, vec3(0.2126, 0.7152, 0.0722));
            float maximum = max(pixel.r, max(pixel.g, pixel.b));
            float minimum = min(pixel.r, min(pixel.g, pixel.b));
            float chroma = maximum - minimum;
            float vegetation = clamp(2.0 * pixel.g - pixel.r - pixel.b, 0.0, 1.0);
            float bright = smoothstep(0.55, 0.90, luminance);
            float lowChroma = 1.0 - smoothstep(0.04, 0.24, chroma);
            float value = clamp(bright * lowChroma * (1.0 - 0.85 * vegetation), 0.0, 1.0);
            return vec4(value, value, value, pixel.a);
        }
        """)

    private let context: CIContext
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

    init() {
        let options: [CIContextOption: Any] = [.cacheIntermediates: false]
        if let device = MTLCreateSystemDefaultDevice() {
            context = CIContext(mtlDevice: device, options: options)
        } else {
            // Simulator fallback remains Core Image and still processes only the downscaled frame.
            context = CIContext(options: options)
        }
    }

    func analyze(
        image: CIImage,
        mode: ColorAssistMode,
        tileRegions: [CGRect],
        includePreview: Bool
    ) -> ColorAssistAnalysis? {
        let startedAt = ProcessInfo.processInfo.systemUptime
        let raw = downscaled(image)
        guard let luminance = apply(Self.luminanceKernel, to: raw),
              let golfContrast = apply(Self.golfContrastKernel, to: raw),
              let golfLuminance = apply(Self.luminanceKernel, to: golfContrast),
              let excessGreen = apply(Self.excessGreenKernel, to: raw),
              let saliency = apply(Self.whiteBallSaliencyKernel, to: raw) else {
            return nil
        }
        let vegetationRejection = excessGreen.applyingFilter("CIColorInvert")

        let priorityMap: CIImage
        switch mode {
        case .raw: priorityMap = luminance
        case .golfContrast: priorityMap = golfLuminance
        case .excessGreen: priorityMap = vegetationRejection
        case .whiteBallSaliency: priorityMap = saliency
        }

        let scores = tileRegions.enumerated().map { index, region in
            ColorAssistTileScore(
                tileIndex: index,
                score: score(map: priorityMap, normalizedTopLeft: region)
            )
        }
        let order = scores.sorted { left, right in
            if left.score == right.score { return left.tileIndex < right.tileIndex }
            return left.score > right.score
        }.map(\.tileIndex)

        let preview: ColorAssistPreview?
        if includePreview,
           let rawImage = makeDebugPreview(from: raw),
           let contrastImage = makeDebugPreview(from: golfContrast),
           let saliencyImage = makeDebugPreview(from: saliency) {
            preview = ColorAssistPreview(
                raw: rawImage,
                golfContrast: contrastImage,
                saliency: saliencyImage
            )
        } else {
            preview = nil
        }

        let latency = max(0, (ProcessInfo.processInfo.systemUptime - startedAt) * 1_000)
        return ColorAssistAnalysis(
            mode: mode,
            processingLatencyMs: latency,
            tileScores: scores,
            selectedTileOrder: order,
            preview: preview
        )
    }

    private func downscaled(_ image: CIImage) -> CIImage {
        let translated = image.transformed(
            by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY)
        )
        let longSide = max(translated.extent.width, translated.extent.height)
        let shortSide = min(translated.extent.width, translated.extent.height)
        guard longSide > 0, shortSide > 0 else { return translated }
        let scale = min(
            1,
            min(
                AppConfig.colorAssistAnalysisLongSide / longSide,
                AppConfig.colorAssistAnalysisShortSide / shortSide
            )
        )
        return translated.applyingFilter(
            "CILanczosScaleTransform",
            parameters: [
                kCIInputScaleKey: scale,
                kCIInputAspectRatioKey: 1.0,
            ]
        )
    }

    private func apply(_ kernel: CIColorKernel?, to image: CIImage) -> CIImage? {
        kernel?.apply(extent: image.extent, arguments: [image])
    }

    /// Materialize diagnostics-only images with an explicit RGBA byte order and sRGB output
    /// profile. The camera/YOLO CIImage is never replaced or rendered through this path.
    private func makeDebugPreview(from image: CIImage) -> CGImage? {
        context.createCGImage(
            image,
            from: image.extent,
            format: .RGBA8,
            colorSpace: Self.debugPreviewColorSpace
        )
    }

    private func score(map: CIImage, normalizedTopLeft region: CGRect) -> Double {
        let crop = map.cropped(normalizedTopLeft: region)
        let mean = areaStatistic(filterName: "CIAreaAverage", image: crop)
        let peak = areaStatistic(filterName: "CIAreaMaximum", image: crop)
        let combined = AppConfig.colorAssistMeanWeight * mean
            + AppConfig.colorAssistPeakWeight * peak
        return min(max(combined, 0), 1)
    }

    private func areaStatistic(filterName: String, image: CIImage) -> Double {
        guard !image.extent.isEmpty,
              let filter = CIFilter(name: filterName) else { return 0 }
        filter.setValue(image, forKey: kCIInputImageKey)
        filter.setValue(CIVector(cgRect: image.extent), forKey: kCIInputExtentKey)
        guard let output = filter.outputImage else { return 0 }

        var pixel = [UInt8](repeating: 0, count: 4)
        pixel.withUnsafeMutableBytes { buffer in
            guard let address = buffer.baseAddress else { return }
            context.render(
                output,
                toBitmap: address,
                rowBytes: 4,
                bounds: output.extent,
                format: .RGBA8,
                colorSpace: colorSpace
            )
        }
        return Double(pixel[0]) / 255
    }
}
