import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraController()

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            DetectionOverlay(state: camera.finderState, imageSize: camera.sourceImageSize)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                statusPill
                Spacer()
                bottomControls
            }
            .padding(.top, 12)
            .padding(.bottom, 24)
            .padding(.horizontal, 18)
        }
        .background(.black)
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }

    @ViewBuilder
    private var statusPill: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 10, height: 10)
            Text(statusText)
                .font(.headline.monospacedDigit())
                .lineLimit(2)
            if camera.inferenceMs > 0 {
                Text(String(format: "%.0fms", camera.inferenceMs))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.black.opacity(0.70), in: Capsule())
    }

    private var bottomControls: some View {
        HStack(spacing: 14) {
            Button {
                camera.toggleZoom()
            } label: {
                Label(camera.zoomFactor < 1.5 ? "1x" : "2x", systemImage: "camera.metering.center.weighted")
            }
            .buttonStyle(ControlButtonStyle())

            Button {
                camera.resetSearch()
            } label: {
                Label("再探索", systemImage: "scope")
            }
            .buttonStyle(ControlButtonStyle())
        }
    }

    private var statusText: String {
        switch camera.finderState {
        case .loading:
            return "AI読み込み中"
        case .scanning:
            return "探索中"
        case .candidate:
            return "候補を確認中"
        case .found:
            return "ボール発見"
        case .error(let message):
            return message
        }
    }

    private var statusColor: Color {
        switch camera.finderState {
        case .found: return .green
        case .candidate: return .yellow
        case .error: return .red
        case .loading: return .orange
        case .scanning: return .cyan
        }
    }
}

private struct ControlButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundStyle(.white)
            .padding(.horizontal, 18)
            .padding(.vertical, 12)
            .background(.black.opacity(configuration.isPressed ? 0.85 : 0.65), in: Capsule())
    }
}
