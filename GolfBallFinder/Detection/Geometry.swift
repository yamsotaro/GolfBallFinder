import CoreGraphics
import CoreImage

extension CGRect {
    func clampedToUnitRect() -> CGRect {
        let x1 = max(0, min(1, minX))
        let y1 = max(0, min(1, minY))
        let x2 = max(0, min(1, maxX))
        let y2 = max(0, min(1, maxY))
        return CGRect(x: x1, y: y1, width: max(0, x2 - x1), height: max(0, y2 - y1))
    }

    func expandedAroundCenter(multiplier: CGFloat, minimumSide: CGFloat) -> CGRect {
        let side = max(minimumSide, max(width, height) * multiplier)
        return CGRect(x: midX - side / 2, y: midY - side / 2, width: side, height: side)
            .clampedPreservingSize()
    }

    /// Clamp a normalized crop while trying to preserve the requested width/height.
    func clampedPreservingSize() -> CGRect {
        let w = min(max(width, 0), 1)
        let h = min(max(height, 0), 1)
        let x = min(max(minX, 0), 1 - w)
        let y = min(max(minY, 0), 1 - h)
        return CGRect(x: x, y: y, width: w, height: h)
    }

    func centerDistance(to other: CGRect) -> CGFloat {
        let dx = midX - other.midX
        let dy = midY - other.midY
        return sqrt(dx * dx + dy * dy)
    }
}

private func finiteComponents(of rect: CGRect) -> Bool {
    rect.origin.x.isFinite && rect.origin.y.isFinite &&
        rect.size.width.isFinite && rect.size.height.isFinite
}

/// Validates the public Ultralytics `Box.xywhn` contract: a top-left-origin normalized CGRect.
/// Validation deliberately happens before clamping so a wildly out-of-bounds model result cannot
/// be converted into a plausible-looking sliver at a screen edge.
func validatedNormalizedTopLeftDetectionRect(
    _ rect: CGRect,
    imageSize: CGSize? = nil
) -> CGRect? {
    guard finiteComponents(of: rect),
          rect.width > 0,
          rect.height > 0 else { return nil }

    let tolerance = AppConfig.normalizedBoundsTolerance
    guard rect.minX >= -tolerance,
          rect.minY >= -tolerance,
          rect.maxX <= 1 + tolerance,
          rect.maxY <= 1 + tolerance else { return nil }

    let normalizedAspect = max(rect.width / rect.height, rect.height / rect.width)
    guard normalizedAspect.isFinite,
          normalizedAspect <= AppConfig.maximumNormalizedBoxAspectRatio else { return nil }

    let clipped = rect.clampedToUnitRect()
    guard clipped.width > 0, clipped.height > 0 else { return nil }

    if let imageSize {
        guard imageSize.width.isFinite,
              imageSize.height.isFinite,
              imageSize.width > 0,
              imageSize.height > 0 else { return nil }
        let pixelWidth = clipped.width * imageSize.width
        let pixelHeight = clipped.height * imageSize.height
        guard pixelWidth >= AppConfig.minimumDetectionPixelSide,
              pixelHeight >= AppConfig.minimumDetectionPixelSide else { return nil }
        let pixelAspect = max(pixelWidth / pixelHeight, pixelHeight / pixelWidth)
        guard pixelAspect.isFinite,
              pixelAspect <= AppConfig.maximumPixelBoxAspectRatio else { return nil }
    }

    return clipped
}

extension CIImage {
    /// Crop using top-left normalized coordinates, which matches UI/object-detection conventions.
    /// Core Image itself uses a bottom-left pixel coordinate system, hence the Y conversion.
    func cropped(normalizedTopLeft region: CGRect) -> CIImage {
        let r = region.clampedPreservingSize()
        let pixelRect = CGRect(
            x: extent.minX + r.minX * extent.width,
            y: extent.minY + (1 - r.maxY) * extent.height,
            width: r.width * extent.width,
            height: r.height * extent.height
        ).integral.intersection(extent)

        let cropped = self.cropped(to: pixelRect)
        return cropped.transformed(
            by: CGAffineTransform(translationX: -cropped.extent.minX, y: -cropped.extent.minY)
        )
    }
}

func mapCropRectToFullFrame(_ cropLocalRect: CGRect, cropRegion: CGRect) -> CGRect {
    CGRect(
        x: cropRegion.minX + cropLocalRect.minX * cropRegion.width,
        y: cropRegion.minY + cropLocalRect.minY * cropRegion.height,
        width: cropLocalRect.width * cropRegion.width,
        height: cropLocalRect.height * cropRegion.height
    ).clampedToUnitRect()
}

/// Maps a normalized top-left image rect into a SwiftUI view that uses aspect-fill camera preview.
func aspectFillDisplayRect(normalizedRect: CGRect, imageSize: CGSize, viewSize: CGSize) -> CGRect {
    guard imageSize.width > 0, imageSize.height > 0, viewSize.width > 0, viewSize.height > 0 else {
        return .zero
    }
    let scale = max(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
    let displayed = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    let offsetX = (viewSize.width - displayed.width) / 2
    let offsetY = (viewSize.height - displayed.height) / 2

    return CGRect(
        x: offsetX + normalizedRect.minX * displayed.width,
        y: offsetY + normalizedRect.minY * displayed.height,
        width: normalizedRect.width * displayed.width,
        height: normalizedRect.height * displayed.height
    )
}
