import CoreGraphics
import SwiftUI

struct FieldDiagnosticsView: View {
    @ObservedObject private var camera: CameraController
    @ObservedObject private var diagnostics: FieldDiagnosticsStore
    @Environment(\.dismiss) private var dismiss
    @State private var occlusion: OcclusionBucket = .unknown
    @State private var sceneID = ""
    @State private var sceneType: FieldSceneType = .positive
    @State private var note = ""

    init(camera: CameraController) {
        _camera = ObservedObject(wrappedValue: camera)
        _diagnostics = ObservedObject(wrappedValue: camera.diagnostics)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Live metrics") {
                    metric("Inference latency", String(format: "%.1f ms", diagnostics.inferenceLatencyMs))
                    metric("Effective inference FPS", String(format: "%.1f", diagnostics.effectiveInferenceFPS))
                    metric("Target FPS", String(format: "%.0f", diagnostics.targetInferenceFPS))
                    metric("Thermal state", diagnostics.thermalState)
                    metric("Detection confidence", confidenceText)
                    metric("Scan mode", diagnostics.scanMode)
                    metric("Detection source", diagnostics.detectionSource ?? "—")
                    metric("Scene start → first candidate", firstCandidateText)
                    metric("Candidate → confirmed", confirmationText)
                    metric("Scene start → confirmed", sceneConfirmationText)
                    metric("BBox (normalized)", boundingBoxText)
                    metric(
                        "Frames received / admitted",
                        "\(diagnostics.frameStatistics.received) / \(diagnostics.frameStatistics.admitted)"
                    )
                    metric(
                        "Dropped busy / cadence",
                        "\(diagnostics.frameStatistics.droppedBusy) / \(diagnostics.frameStatistics.droppedCadence)"
                    )
                }

                Section("Color Assist (Experimental)") {
                    Toggle(
                        "Prioritize tiles with Color Assist",
                        isOn: Binding(
                            get: { camera.colorAssistEnabled },
                            set: { camera.setColorAssistEnabled($0) }
                        )
                    )

                    Picker(
                        "Priority filter",
                        selection: Binding(
                            get: { camera.colorAssistMode },
                            set: { camera.setColorAssistMode($0) }
                        )
                    ) {
                        ForEach(ColorAssistMode.allCases) { mode in
                            Text(mode.displayName).tag(mode)
                        }
                    }

                    metric("Effective", diagnostics.colorAssistEnabled ? "ON" : "OFF")
                    metric("Color processing", colorProcessingText)
                    metric("Tile scores [0...4]", tileScoresText)
                    metric("Selected order", tileOrderText)

                    Picker("Ball-containing tile", selection: ballTileSelection) {
                        Text("Not annotated").tag(-1)
                        ForEach(AppConfig.searchTiles.indices, id: \.self) { index in
                            Text("Tile \(index)").tag(index)
                        }
                    }
                    Text("Tile 0/1/2/3 = top-left/top-right/bottom-left/bottom-right; Tile 4 = center.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    metric("Annotated tile rank", ballTileRankText)

                    if let preview = camera.colorAssistPreview {
                        previewImage("Raw", image: preview.raw)
                        previewImage("Golf Contrast", image: preview.golfContrast)
                        previewImage("White-ball saliency", image: preview.saliency)
                    } else {
                        Text("Waiting for a low-resolution preview. Preview work pauses at serious/critical thermal state.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    Text("Experimental ordering only. YOLO always receives the original Raw RGB frame/crop.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Field feedback") {
                    TextField("Scene ID（例: courseA_pos_001）", text: $sceneID)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Picker("Scene type", selection: $sceneType) {
                        ForEach(FieldSceneType.allCases) { type in
                            Text(type.displayName).tag(type)
                        }
                    }

                    Picker("Ball visibility", selection: $occlusion) {
                        ForEach(OcclusionBucket.allCases) { bucket in
                            Text(bucket.displayName).tag(bucket)
                        }
                    }

                    TextField("メモ（キノコ、石、逆光など）", text: $note, axis: .vertical)
                        .lineLimit(2...4)

                    Button {
                        camera.beginFieldScene(id: sceneID, type: sceneType, occlusion: occlusion, note: note)
                    } label: {
                        Label("10秒シーン開始 / 再探索", systemImage: "timer")
                    }

                    Button {
                        camera.endFieldScene(note: note)
                    } label: {
                        Label("シーン終了", systemImage: "stop.circle")
                    }
                    .disabled(diagnostics.activeSceneID == nil)

                    Button(role: .destructive) {
                        camera.recordFalsePositive(occlusion: occlusion, note: note)
                        note = ""
                    } label: {
                        Label("これはボールではない", systemImage: "xmark.circle")
                    }
                    .disabled(!hasCandidate || diagnostics.isSavingEvidence)

                    Button {
                        camera.recordTruePositive(occlusion: occlusion, note: note)
                        note = ""
                    } label: {
                        Label("正しいボール発見", systemImage: "checkmark.circle")
                    }
                    .disabled(!isFound || diagnostics.isSavingEvidence)

                    Button {
                        camera.recordMiss(occlusion: occlusion, note: note)
                        note = ""
                    } label: {
                        Label("見逃しを記録", systemImage: "eye.slash")
                    }
                    .disabled(diagnostics.isSavingEvidence)

                    if diagnostics.isSavingEvidence {
                        HStack {
                            ProgressView()
                            Text("最新フレームとmetadataを保存中")
                        }
                    } else if let message = diagnostics.lastSaveMessage {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Local session") {
                    LabeledContent("Session", value: diagnostics.sessionID)
                    LabeledContent("Active scene", value: diagnostics.activeSceneID ?? "—")
                    if let logURL = diagnostics.logFileURL {
                        ShareLink(item: logURL) {
                            Label("events.jsonlを共有", systemImage: "square.and.arrow.up")
                        }
                    }
                    if let directory = diagnostics.sessionDirectoryURL {
                        Text("Files > このiPhone内 > Golf Ball Finder > FieldDiagnostics > \(directory.lastPathComponent)")
                            .font(.footnote.monospaced())
                            .textSelection(.enabled)
                    }
                    Text("画像・ログは端末内だけに保存され、自動アップロードされません。")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                if camera.cameraPermissionDenied {
                    Section("Camera permission") {
                        Button("設定を開く") { camera.openApplicationSettings() }
                    }
                }
            }
            .navigationTitle("Field Diagnostics")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("閉じる") { dismiss() }
                }
            }
            .onAppear { camera.setColorAssistPreviewRequested(true) }
            .onDisappear { camera.setColorAssistPreviewRequested(false) }
        }
    }

    private func metric(_ label: String, _ value: String) -> some View {
        LabeledContent(label) {
            Text(value).font(.body.monospacedDigit())
        }
    }

    private var hasCandidate: Bool {
        diagnostics.hasRecentDetection
    }

    private var isFound: Bool {
        diagnostics.hasRecentConfirmedDetection
    }

    private var confidenceText: String {
        diagnostics.detectionConfidence.map { String(format: "%.3f", $0) } ?? "—"
    }

    private var confirmationText: String {
        diagnostics.candidateToConfirmedMs.map { String(format: "%.0f ms", $0) } ?? "—"
    }

    private var sceneConfirmationText: String {
        diagnostics.sceneStartToConfirmedMs.map { String(format: "%.0f ms", $0) } ?? "—"
    }

    private var firstCandidateText: String {
        diagnostics.sceneStartToFirstCandidateMs.map { String(format: "%.0f ms", $0) } ?? "—"
    }

    private var colorProcessingText: String {
        diagnostics.colorProcessingLatencyMs.map { String(format: "%.2f ms", $0) } ?? "—"
    }

    private var tileScoresText: String {
        guard let scores = diagnostics.tileSaliencyScores else { return "—" }
        return scores.map { String(format: "%.3f", $0) }.joined(separator: ", ")
    }

    private var tileOrderText: String {
        diagnostics.selectedTileOrder.map(String.init).joined(separator: " → ")
    }

    private var ballTileRankText: String {
        diagnostics.ballContainingTileRank.map { "\($0) / \(AppConfig.searchTiles.count)" } ?? "—"
    }

    private var ballTileSelection: Binding<Int> {
        Binding(
            get: { diagnostics.ballContainingTileIndex ?? -1 },
            set: { camera.annotateBallContainingTile($0 >= 0 ? $0 : nil) }
        )
    }

    private func previewImage(_ title: String, image: CGImage) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption.weight(.semibold))
            Image(decorative: image, scale: 1, orientation: .up)
                .resizable()
                .scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .accessibilityLabel(title)
        }
    }

    private var boundingBoxText: String {
        guard let box = diagnostics.detectedBBox else { return "—" }
        return String(format: "x %.3f y %.3f w %.3f h %.3f", box.x, box.y, box.width, box.height)
    }
}
