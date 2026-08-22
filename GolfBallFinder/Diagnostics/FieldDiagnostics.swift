import Combine
import CoreImage
import Foundation
import ImageIO

enum FieldEventKind: String, Codable, Sendable {
    case sessionStart = "session_start"
    case sceneStart = "scene_start"
    case sceneEnd = "scene_end"
    case inferenceSample = "inference_sample"
    case confirmed
    case truePositive = "true_positive"
    case falsePositive = "false_positive"
    case miss
    case cameraEvent = "camera_event"
}

enum FieldSceneType: String, CaseIterable, Codable, Identifiable, Sendable {
    case positive
    case negative

    var id: String { rawValue }
    var displayName: String { self == .positive ? "ボールあり" : "ボールなし" }
}

enum OcclusionBucket: String, CaseIterable, Codable, Identifiable, Sendable {
    case unknown
    case visible100 = "visible_100"
    case visible75 = "visible_75"
    case visible50 = "visible_50"
    case visible25 = "visible_25"
    case visible10 = "visible_10"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .unknown: return "未指定"
        case .visible100: return "100%見える"
        case .visible75: return "約75%見える"
        case .visible50: return "約50%見える"
        case .visible25: return "約25%見える"
        case .visible10: return "約10%見える"
        }
    }
}

struct DiagnosticBoundingBox: Codable, Equatable, Sendable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double

    init(_ rect: CGRect) {
        x = Double(rect.minX)
        y = Double(rect.minY)
        width = Double(rect.width)
        height = Double(rect.height)
    }
}

struct FieldDiagnosticRecord: Codable, Equatable, Sendable {
    let schemaVersion: Int
    let eventID: String
    let sessionID: String
    let sceneID: String?
    let sceneType: FieldSceneType?
    let timestamp: Date
    let kind: FieldEventKind
    let detectionState: String
    let appVersion: String
    let buildNumber: String
    let modelCheckpointSHA256: String?
    let inferenceLatencyMs: Double?
    let effectiveInferenceFPS: Double?
    let targetInferenceFPS: Double?
    let thermalState: String
    let detectionConfidence: Float?
    let scanMode: String?
    let colorAssistRequested: Bool
    let colorAssistEnabled: Bool
    let colorAssistFilterMode: String
    let colorProcessingLatencyMs: Double?
    let tileSaliencyScores: [Double]?
    let selectedTileOrder: [Int]
    let ballContainingTileIndex: Int?
    let ballContainingTileRank: Int?
    let sceneStartToFirstCandidateMs: Double?
    let candidateToConfirmedMs: Double?
    let sceneStartToConfirmedMs: Double?
    let detectedBBox: DiagnosticBoundingBox?
    let zoomFactor: Double
    let cameraFramesReceived: Int?
    let inferenceFramesAdmitted: Int?
    let framesDroppedBusy: Int?
    let framesDroppedCadence: Int?
    let manualLabel: String?
    let occlusion: OcclusionBucket?
    let note: String?
    let evidenceFilename: String?
}

struct InferenceRateMeter: Sendable {
    private var timestamps: [TimeInterval] = []
    private let windowSeconds: TimeInterval

    init(windowSeconds: TimeInterval = 2.0) {
        self.windowSeconds = windowSeconds
    }

    mutating func record(at timestamp: TimeInterval) -> Double {
        timestamps.append(timestamp)
        let cutoff = timestamp - windowSeconds
        timestamps.removeAll { $0 < cutoff }
        guard timestamps.count >= 2, let first = timestamps.first, let last = timestamps.last, last > first else {
            return 0
        }
        return Double(timestamps.count - 1) / (last - first)
    }
}

private struct ModelIdentity: Decodable {
    let checkpointSHA256: String?

    enum CodingKeys: String, CodingKey {
        case checkpointSHA256 = "checkpoint_sha256"
    }
}

/// Retains exactly one camera-backed CIImage. Replacing it releases the previous buffer, avoiding
/// an image queue while still allowing an explicit field-feedback tap to preserve local evidence.
final class LatestFrameSnapshot: @unchecked Sendable {
    private let lock = NSLock()
    private var image: CIImage?

    func update(pixelBuffer: CVPixelBuffer) {
        lock.lock()
        image = CIImage(cvPixelBuffer: pixelBuffer)
        lock.unlock()
    }

    func current() -> CIImage? {
        lock.lock()
        defer { lock.unlock() }
        return image
    }
}

final class FieldDiagnosticsStore: ObservableObject {
    @Published private(set) var inferenceLatencyMs: Double = 0
    @Published private(set) var effectiveInferenceFPS: Double = 0
    @Published private(set) var targetInferenceFPS: Double = AppConfig.processedFPS
    @Published private(set) var thermalState = ProcessInfo.processInfo.thermalState.diagnosticName
    @Published private(set) var detectionConfidence: Float?
    @Published private(set) var hasRecentDetection = false
    @Published private(set) var hasRecentConfirmedDetection = false
    @Published private(set) var scanMode = ScanSource.fullFrame.rawValue
    @Published private(set) var colorAssistRequested = AppConfig.colorAssistDefaultEnabled
    @Published private(set) var colorAssistEnabled = false
    @Published private(set) var colorAssistFilterMode = ColorAssistMode.whiteBallSaliency.rawValue
    @Published private(set) var colorProcessingLatencyMs: Double?
    @Published private(set) var tileSaliencyScores: [Double]?
    @Published private(set) var selectedTileOrder = Array(AppConfig.searchTiles.indices)
    @Published private(set) var ballContainingTileIndex: Int?
    @Published private(set) var ballContainingTileRank: Int?
    @Published private(set) var sceneStartToFirstCandidateMs: Double?
    @Published private(set) var candidateToConfirmedMs: Double?
    @Published private(set) var sceneStartToConfirmedMs: Double?
    @Published private(set) var detectedBBox: DiagnosticBoundingBox?
    @Published private(set) var frameStatistics = InferenceFrameStatistics(
        received: 0,
        admitted: 0,
        droppedBusy: 0,
        droppedCadence: 0
    )
    @Published private(set) var sessionID: String
    @Published private(set) var activeSceneID: String?
    @Published private(set) var activeSceneType: FieldSceneType?
    @Published private(set) var sessionDirectoryURL: URL?
    @Published private(set) var logFileURL: URL?
    @Published private(set) var isSavingEvidence = false
    @Published private(set) var lastSaveMessage: String?

    private let ioQueue = DispatchQueue(label: "GolfBallFinder.field-diagnostics", qos: .utility)
    private let encoder: JSONEncoder
    private let modelCheckpointSHA256: String?
    private let appVersion: String
    private let buildNumber: String
    private var rateMeter = InferenceRateMeter()
    private var lastSampleLoggedAt: TimeInterval?
    private var candidateBeganAt: TimeInterval?
    private var sceneBeganAt: TimeInterval?
    private var recentSceneStartToFirstCandidateMs: Double?
    private var previousWasFound = false
    private var latestObservation: DetectionObservation?
    private var recentFeedbackObservation: DetectionObservation?
    private var recentFeedbackAt: TimeInterval?
    private var recentFeedbackState = "scanning"
    private var recentCandidateToConfirmedMs: Double?
    private var recentSceneStartToConfirmedMs: Double?
    private var latestDetectionState = "loading"
    private var latestZoom: CGFloat = 1

    init() {
        sessionID = FieldDiagnosticsStore.makeSessionID()
        appVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        buildNumber = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
        modelCheckpointSHA256 = FieldDiagnosticsStore.loadModelIdentity()?.checkpointSHA256
        encoder = JSONEncoder()
        let timestampFormatter = ISO8601DateFormatter()
        timestampFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(timestampFormatter.string(from: date))
        }
        encoder.outputFormatting = [.sortedKeys]
        createSessionAndWriteStart()
    }

    func recordInference(
        latencyMs: Double,
        source: ScanSource,
        state: FinderState,
        thermal: ProcessInfo.ThermalState,
        targetFPS: Double,
        zoom: CGFloat,
        statistics: InferenceFrameStatistics,
        colorAssistRequested: Bool,
        colorAssistEnabled: Bool,
        colorAssistMode: ColorAssistMode,
        colorProcessingLatencyMs: Double?,
        tileSaliencyScores: [Double]?,
        selectedTileOrder: [Int],
        now: Date = Date()
    ) {
        let time = now.timeIntervalSinceReferenceDate
        inferenceLatencyMs = latencyMs
        effectiveInferenceFPS = rateMeter.record(at: time)
        targetInferenceFPS = targetFPS
        thermalState = thermal.diagnosticName
        scanMode = source.rawValue
        frameStatistics = statistics
        latestZoom = zoom
        self.colorAssistRequested = colorAssistRequested
        self.colorAssistEnabled = colorAssistEnabled
        colorAssistFilterMode = colorAssistMode.rawValue
        self.colorProcessingLatencyMs = colorProcessingLatencyMs
        self.tileSaliencyScores = tileSaliencyScores
        self.selectedTileOrder = selectedTileOrder
        updateBallContainingTileRank()

        let observation: DetectionObservation?
        switch state {
        case .candidate(let item), .found(let item): observation = item
        default: observation = nil
        }
        latestObservation = observation
        latestDetectionState = state.diagnosticName
        detectionConfidence = observation?.confidence
        detectedBBox = observation.map { DiagnosticBoundingBox($0.normalizedRect) }
        if let observation {
            recentFeedbackObservation = observation
            recentFeedbackAt = time
            recentFeedbackState = state.diagnosticName
            hasRecentDetection = true
            if case .found = state { hasRecentConfirmedDetection = true }
        } else if let recentFeedbackAt, time - recentFeedbackAt > AppConfig.feedbackCaptureGraceSeconds {
            recentFeedbackObservation = nil
            self.recentFeedbackAt = nil
            hasRecentDetection = false
            hasRecentConfirmedDetection = false
            recentCandidateToConfirmedMs = nil
            recentSceneStartToConfirmedMs = nil
            recentSceneStartToFirstCandidateMs = nil
        }

        switch state {
        case .candidate:
            if candidateBeganAt == nil { candidateBeganAt = time }
            recordFirstCandidateIfNeeded(at: time)
            previousWasFound = false
        case .found:
            if candidateBeganAt == nil { candidateBeganAt = time }
            recordFirstCandidateIfNeeded(at: time)
            let duration = max(0, (time - (candidateBeganAt ?? time)) * 1_000)
            candidateToConfirmedMs = duration
            if let sceneBeganAt {
                sceneStartToConfirmedMs = max(0, (time - sceneBeganAt) * 1_000)
            }
            recentCandidateToConfirmedMs = duration
            recentSceneStartToConfirmedMs = sceneStartToConfirmedMs
            if !previousWasFound {
                append(record(
                    kind: .confirmed,
                    now: now,
                    sceneStartToFirstCandidateMs: sceneStartToFirstCandidateMs,
                    candidateToConfirmedMs: duration,
                    sceneStartToConfirmedMs: sceneStartToConfirmedMs
                ))
            }
            previousWasFound = true
        case .scanning, .loading, .paused, .error:
            candidateBeganAt = nil
            candidateToConfirmedMs = nil
            previousWasFound = false
        }

        if lastSampleLoggedAt.map({ time - $0 >= AppConfig.diagnosticsSampleInterval }) ?? true {
            lastSampleLoggedAt = time
            append(record(kind: .inferenceSample, now: now, candidateToConfirmedMs: candidateToConfirmedMs))
        }
    }

    @discardableResult
    func beginScene(id: String, type: FieldSceneType, occlusion: OcclusionBucket, note: String) -> Bool {
        let cleanedID = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanedID.isEmpty else {
            lastSaveMessage = "Scene IDを入力してください"
            return false
        }
        if activeSceneID != nil {
            append(record(kind: .sceneEnd, now: Date(), note: "superseded_by_new_scene"))
        }
        activeSceneID = cleanedID
        activeSceneType = type
        sceneBeganAt = Date().timeIntervalSinceReferenceDate
        sceneStartToConfirmedMs = nil
        sceneStartToFirstCandidateMs = nil
        candidateBeganAt = nil
        previousWasFound = false
        latestObservation = nil
        recentFeedbackObservation = nil
        recentFeedbackAt = nil
        detectionConfidence = nil
        detectedBBox = nil
        hasRecentDetection = false
        hasRecentConfirmedDetection = false
        recentCandidateToConfirmedMs = nil
        recentSceneStartToConfirmedMs = nil
        recentSceneStartToFirstCandidateMs = nil
        latestDetectionState = "scanning"
        lastSaveMessage = "Scene \(cleanedID) を開始しました"
        append(record(
            kind: .sceneStart,
            now: Date(),
            occlusion: occlusion,
            note: note.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        ))
        return true
    }

    @discardableResult
    func endScene(note: String) -> Bool {
        guard let activeSceneID else {
            lastSaveMessage = "開始中のSceneがありません"
            return false
        }
        append(record(
            kind: .sceneEnd,
            now: Date(),
            note: note.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        ))
        lastSaveMessage = "Scene \(activeSceneID) を終了しました"
        self.activeSceneID = nil
        activeSceneType = nil
        sceneBeganAt = nil
        sceneStartToConfirmedMs = nil
        sceneStartToFirstCandidateMs = nil
        candidateBeganAt = nil
        previousWasFound = false
        latestObservation = nil
        recentFeedbackObservation = nil
        recentFeedbackAt = nil
        detectionConfidence = nil
        detectedBBox = nil
        hasRecentDetection = false
        hasRecentConfirmedDetection = false
        recentCandidateToConfirmedMs = nil
        recentSceneStartToConfirmedMs = nil
        recentSceneStartToFirstCandidateMs = nil
        latestDetectionState = "scanning"
        return true
    }

    func annotateBallContainingTile(_ tileIndex: Int?) {
        if let tileIndex, AppConfig.searchTiles.indices.contains(tileIndex) {
            ballContainingTileIndex = tileIndex
        } else {
            ballContainingTileIndex = nil
        }
        updateBallContainingTileRank()
    }

    func recordCameraEvent(_ note: String, thermal: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState) {
        thermalState = thermal.diagnosticName
        append(record(kind: .cameraEvent, now: Date(), note: note))
    }

    func recordManualEvent(
        kind: FieldEventKind,
        image: CIImage?,
        occlusion: OcclusionBucket,
        note: String
    ) {
        guard kind == .truePositive || kind == .falsePositive || kind == .miss else { return }
        guard !isSavingEvidence else {
            lastSaveMessage = "前の記録を保存中です"
            return
        }

        isSavingEvidence = true
        lastSaveMessage = nil
        let now = Date()
        let eventID = UUID().uuidString.lowercased()
        let directory = sessionDirectoryURL
        let baseRecord = record(
            kind: kind,
            now: now,
            eventID: eventID,
            sceneStartToFirstCandidateMs: kind == .miss
                ? sceneStartToFirstCandidateMs
                : recentSceneStartToFirstCandidateMs,
            candidateToConfirmedMs: kind == .miss ? candidateToConfirmedMs : recentCandidateToConfirmedMs,
            sceneStartToConfirmedMs: kind == .miss ? sceneStartToConfirmedMs : recentSceneStartToConfirmedMs,
            manualLabel: manualLabel(for: kind),
            occlusion: occlusion,
            note: note.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty,
            observation: kind == .miss ? latestObservation : recentFeedbackObservation,
            detectionState: kind == .miss ? latestDetectionState : recentFeedbackState
        )
        if kind == .truePositive || kind == .falsePositive {
            recentFeedbackObservation = nil
            recentFeedbackAt = nil
            hasRecentDetection = false
            hasRecentConfirmedDetection = false
            recentCandidateToConfirmedMs = nil
            recentSceneStartToConfirmedMs = nil
            recentSceneStartToFirstCandidateMs = nil
        }

        ioQueue.async { [weak self] in
            guard let self else { return }
            var evidenceFilename: String?
            var saveError: Error?
            do {
                if let image, let directory {
                    let evidenceDirectory = directory.appendingPathComponent("evidence", isDirectory: true)
                    try FileManager.default.createDirectory(
                        at: evidenceDirectory,
                        withIntermediateDirectories: true
                    )
                    let filename = "\(kind.rawValue)_\(eventID).jpg"
                    let target = evidenceDirectory.appendingPathComponent(filename)
                    let context = CIContext(options: [.cacheIntermediates: false])
                    guard let data = context.jpegRepresentation(
                        of: image,
                        colorSpace: CGColorSpaceCreateDeviceRGB(),
                        options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.90]
                    ) else {
                        throw DiagnosticsError.cannotEncodeEvidence
                    }
                    try data.write(to: target, options: .atomic)
                    try FileManager.default.setAttributes(
                        [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
                        ofItemAtPath: target.path
                    )
                    evidenceFilename = "evidence/\(filename)"
                }
                let completed = FieldDiagnosticRecord(
                    schemaVersion: baseRecord.schemaVersion,
                    eventID: baseRecord.eventID,
                    sessionID: baseRecord.sessionID,
                    sceneID: baseRecord.sceneID,
                    sceneType: baseRecord.sceneType,
                    timestamp: baseRecord.timestamp,
                    kind: baseRecord.kind,
                    detectionState: baseRecord.detectionState,
                    appVersion: baseRecord.appVersion,
                    buildNumber: baseRecord.buildNumber,
                    modelCheckpointSHA256: baseRecord.modelCheckpointSHA256,
                    inferenceLatencyMs: baseRecord.inferenceLatencyMs,
                    effectiveInferenceFPS: baseRecord.effectiveInferenceFPS,
                    targetInferenceFPS: baseRecord.targetInferenceFPS,
                    thermalState: baseRecord.thermalState,
                    detectionConfidence: baseRecord.detectionConfidence,
                    scanMode: baseRecord.scanMode,
                    colorAssistRequested: baseRecord.colorAssistRequested,
                    colorAssistEnabled: baseRecord.colorAssistEnabled,
                    colorAssistFilterMode: baseRecord.colorAssistFilterMode,
                    colorProcessingLatencyMs: baseRecord.colorProcessingLatencyMs,
                    tileSaliencyScores: baseRecord.tileSaliencyScores,
                    selectedTileOrder: baseRecord.selectedTileOrder,
                    ballContainingTileIndex: baseRecord.ballContainingTileIndex,
                    ballContainingTileRank: baseRecord.ballContainingTileRank,
                    sceneStartToFirstCandidateMs: baseRecord.sceneStartToFirstCandidateMs,
                    candidateToConfirmedMs: baseRecord.candidateToConfirmedMs,
                    sceneStartToConfirmedMs: baseRecord.sceneStartToConfirmedMs,
                    detectedBBox: baseRecord.detectedBBox,
                    zoomFactor: baseRecord.zoomFactor,
                    cameraFramesReceived: baseRecord.cameraFramesReceived,
                    inferenceFramesAdmitted: baseRecord.inferenceFramesAdmitted,
                    framesDroppedBusy: baseRecord.framesDroppedBusy,
                    framesDroppedCadence: baseRecord.framesDroppedCadence,
                    manualLabel: baseRecord.manualLabel,
                    occlusion: baseRecord.occlusion,
                    note: baseRecord.note,
                    evidenceFilename: evidenceFilename
                )
                try self.appendSynchronously(completed)
            } catch {
                saveError = error
                if let data = try? self.encoder.encode(baseRecord) {
                    try? self.appendDataSynchronously(data)
                }
            }

            DispatchQueue.main.async {
                self.isSavingEvidence = false
                if let saveError {
                    self.lastSaveMessage = "metadataのみ保存しました: \(saveError.localizedDescription)"
                } else if evidenceFilename != nil {
                    self.lastSaveMessage = "画像とmetadataを保存しました"
                } else {
                    self.lastSaveMessage = "metadataを保存しました（フレームなし）"
                }
            }
        }
    }

    private func record(
        kind: FieldEventKind,
        now: Date,
        eventID: String = UUID().uuidString.lowercased(),
        sceneStartToFirstCandidateMs: Double? = nil,
        candidateToConfirmedMs: Double? = nil,
        sceneStartToConfirmedMs: Double? = nil,
        manualLabel: String? = nil,
        occlusion: OcclusionBucket? = nil,
        note: String? = nil,
        observation: DetectionObservation? = nil,
        detectionState: String? = nil
    ) -> FieldDiagnosticRecord {
        let capturedObservation = observation ?? latestObservation
        return FieldDiagnosticRecord(
            schemaVersion: 2,
            eventID: eventID,
            sessionID: sessionID,
            sceneID: activeSceneID,
            sceneType: activeSceneType,
            timestamp: now,
            kind: kind,
            detectionState: detectionState ?? latestDetectionState,
            appVersion: appVersion,
            buildNumber: buildNumber,
            modelCheckpointSHA256: modelCheckpointSHA256,
            inferenceLatencyMs: inferenceLatencyMs > 0 ? inferenceLatencyMs : nil,
            effectiveInferenceFPS: effectiveInferenceFPS > 0 ? effectiveInferenceFPS : nil,
            targetInferenceFPS: targetInferenceFPS,
            thermalState: thermalState,
            detectionConfidence: capturedObservation?.confidence,
            scanMode: scanMode,
            colorAssistRequested: colorAssistRequested,
            colorAssistEnabled: colorAssistEnabled,
            colorAssistFilterMode: colorAssistFilterMode,
            colorProcessingLatencyMs: colorProcessingLatencyMs,
            tileSaliencyScores: tileSaliencyScores,
            selectedTileOrder: selectedTileOrder,
            ballContainingTileIndex: ballContainingTileIndex,
            ballContainingTileRank: ballContainingTileRank,
            sceneStartToFirstCandidateMs: sceneStartToFirstCandidateMs ?? self.sceneStartToFirstCandidateMs,
            candidateToConfirmedMs: candidateToConfirmedMs,
            sceneStartToConfirmedMs: sceneStartToConfirmedMs,
            detectedBBox: capturedObservation.map { DiagnosticBoundingBox($0.normalizedRect) },
            zoomFactor: Double(latestZoom),
            cameraFramesReceived: frameStatistics.received,
            inferenceFramesAdmitted: frameStatistics.admitted,
            framesDroppedBusy: frameStatistics.droppedBusy,
            framesDroppedCadence: frameStatistics.droppedCadence,
            manualLabel: manualLabel,
            occlusion: occlusion,
            note: note,
            evidenceFilename: nil
        )
    }

    private func manualLabel(for kind: FieldEventKind) -> String {
        switch kind {
        case .truePositive: return "golf_ball_confirmed"
        case .falsePositive: return "not_golf_ball"
        case .miss: return "missed_golf_ball"
        default: return "unknown"
        }
    }

    private func recordFirstCandidateIfNeeded(at time: TimeInterval) {
        guard sceneStartToFirstCandidateMs == nil, let sceneBeganAt else { return }
        let duration = max(0, (time - sceneBeganAt) * 1_000)
        sceneStartToFirstCandidateMs = duration
        recentSceneStartToFirstCandidateMs = duration
    }

    private func updateBallContainingTileRank() {
        guard let ballContainingTileIndex,
              let position = selectedTileOrder.firstIndex(of: ballContainingTileIndex) else {
            ballContainingTileRank = nil
            return
        }
        ballContainingTileRank = position + 1
    }

    private func createSessionAndWriteStart() {
        do {
            let documents = try FileManager.default.url(
                for: .documentDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            let directory = documents
                .appendingPathComponent("FieldDiagnostics", isDirectory: true)
                .appendingPathComponent(sessionID, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let log = directory.appendingPathComponent("events.jsonl")
            if !FileManager.default.fileExists(atPath: log.path) {
                try Data().write(to: log, options: .atomic)
            }
            try FileManager.default.setAttributes(
                [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
                ofItemAtPath: directory.path
            )
            sessionDirectoryURL = directory
            logFileURL = log
            append(record(kind: .sessionStart, now: Date(), note: "local_only"))
        } catch {
            lastSaveMessage = "診断セッションを作成できません: \(error.localizedDescription)"
        }
    }

    private func append(_ item: FieldDiagnosticRecord) {
        ioQueue.async { [weak self] in
            try? self?.appendSynchronously(item)
        }
    }

    private func appendSynchronously(_ item: FieldDiagnosticRecord) throws {
        try appendDataSynchronously(encoder.encode(item))
    }

    private func appendDataSynchronously(_ data: Data) throws {
        guard let logFileURL else { throw DiagnosticsError.logUnavailable }
        let handle = try FileHandle(forWritingTo: logFileURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        handle.write(data)
        handle.write(Data([0x0A]))
    }

    private static func loadModelIdentity() -> ModelIdentity? {
        guard let url = Bundle.main.url(forResource: "ModelManifest", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ModelIdentity.self, from: data)
    }

    private static func makeSessionID() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd'T'HHmmss'Z'"
        return "\(formatter.string(from: Date()))_\(UUID().uuidString.prefix(8).lowercased())"
    }
}

private enum DiagnosticsError: LocalizedError {
    case logUnavailable
    case cannotEncodeEvidence

    var errorDescription: String? {
        switch self {
        case .logUnavailable: return "診断ログが利用できません"
        case .cannotEncodeEvidence: return "最新フレームをJPEGへ変換できません"
        }
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
