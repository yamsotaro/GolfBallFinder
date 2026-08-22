import SwiftUI

struct DetectionOverlay: View {
    let state: FinderState
    let imageSize: CGSize

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                switch state {
                case .candidate(let detection):
                    ring(for: detection, in: geometry.size, confirmed: false)
                case .found(let detection):
                    ring(for: detection, in: geometry.size, confirmed: true)
                default:
                    EmptyView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .allowsHitTesting(false)
    }

    @ViewBuilder
    private func ring(for detection: DetectionObservation, in viewSize: CGSize, confirmed: Bool) -> some View {
        let rect = aspectFillDisplayRect(
            normalizedRect: detection.normalizedRect,
            imageSize: imageSize,
            viewSize: viewSize
        )
        let diameter = max(72, max(rect.width, rect.height) * 2.2)
        let center = CGPoint(x: rect.midX, y: rect.midY)

        ZStack {
            Circle()
                .stroke(confirmed ? Color.green : Color.yellow, lineWidth: confirmed ? 8 : 5)
                .frame(width: diameter, height: diameter)
                .shadow(radius: 3)

            if confirmed {
                Image(systemName: "arrow.down.circle.fill")
                    .font(.system(size: 34, weight: .bold))
                    .foregroundStyle(.green)
                    .offset(y: -diameter / 2 - 28)
            }
        }
        .position(center)
        .animation(.easeOut(duration: 0.12), value: center.x)
        .animation(.easeOut(duration: 0.12), value: center.y)
    }
}
