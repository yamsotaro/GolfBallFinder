import AVFoundation
import CoreImage
import Foundation
import SwiftUI
import UIKit

final class CameraController: NSObject, ObservableObject {
    @Published private(set) var finderState: FinderState = .loading
    @Published private(set) var sourceImageSize: CGSize = .zero
    @Published private(set) var inferenceMs: Double = 0
    @Published private(set) var currentSource: ScanSource = .fullFrame
    @Published private(set) var zoomFactor: CGFloat = 1.0
    @Published private(set) var cameraPermissionDenied = false
    @Published private(set) var colorAssistEnabled = AppConfig.colorAssistDefaultEnabled
    @Published private(set) var colorAssistEffective = false
    @Published private(set) var colorAssistMode: ColorAssistMode = .whiteBallSaliency
    @Published private(set) var colorAssistPreview: ColorAssistPreview?

    let session = AVCaptureSession()
    let diagnostics = FieldDiagnosticsStore()

    private let sessionQueue = DispatchQueue(label: "GolfBallFinder.camera.session")
    private let videoQueue = DispatchQueue(label: "GolfBallFinder.camera.frames", qos: .userInteractive)
    private let inferenceQueue = DispatchQueue(label: "GolfBallFinder.inference", qos: .userInitiated)
    private let detector = GolfBallDetector()
    private let colorAssistEngine = ColorAssistEngine()
    private let latestFrame = LatestFrameSnapshot()

    // Accessed only from inferenceQueue.
    private var scheduler = ScanScheduler()
    private var stabilizer = DetectionStabilizer()
    private var colorAssistRequested = AppConfig.colorAssistDefaultEnabled
    private var selectedColorAssistMode: ColorAssistMode = .whiteBallSaliency
    private var colorAssistPreviewRequested = false
    private var lastColorAssistPreviewAt: TimeInterval?
    private var latestColorAssistScores: [Double]?
    private var latestSelectedTileOrder = Array(AppConfig.searchTiles.indices)

    // Accessed only from videoQueue. The gate intentionally has no pending-frame storage.
    private var frameGate = InferenceFrameGate()
    private var searchEpoch = 0
    private var inferenceSuspended = false

    // Accessed only from sessionQueue after configuration.
    private var captureDevice: AVCaptureDevice?
    private var configured = false

    // Accessed on the main queue through SwiftUI and main-queue notification observers.
    private var wantsSessionRunning = false
    private var permissionRequestInFlight = false
    private var notificationTokens: [NSObjectProtocol] = []

    override init() {
        super.init()
        installObservers()
        detector.load { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                switch result {
                case .success:
                    if case .loading = self.finderState {
                        self.finderState = .scanning
                    }
                case .failure(let error):
                    self.finderState = .error(
                        "GolfBallモデルを読み込めません。GolfBall.mlpackageをResourcesへ追加してください。\n\(error.localizedDescription)"
                    )
                }
            }
        }
    }

    deinit {
        for token in notificationTokens {
            NotificationCenter.default.removeObserver(token)
        }
    }

    func start() {
        wantsSessionRunning = true
        guard !permissionRequestInFlight else { return }
        permissionRequestInFlight = true
        Task { [weak self] in
            guard let self else { return }
            let granted = await self.requestCameraPermission()
            guard granted else {
                DispatchQueue.main.async {
                    self.permissionRequestInFlight = false
                    self.cameraPermissionDenied = true
                    self.finderState = .error(
                        "カメラ権限が必要です。設定 > プライバシーとセキュリティ > カメラで許可してください。"
                    )
                    self.diagnostics.recordCameraEvent("camera_permission_denied")
                }
                return
            }
            DispatchQueue.main.async {
                self.permissionRequestInFlight = false
                self.cameraPermissionDenied = false
                if self.wantsSessionRunning {
                    self.configureAndStartSession()
                }
            }
        }
    }

    func stop() {
        wantsSessionRunning = false
        invalidateSearch(publishScanning: false, suspendInference: true)
        stopSession()
    }

    func openApplicationSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }

    func toggleZoom() {
        let target = zoomFactor < 1.5 ? CGFloat(2.0) : CGFloat(1.0)
        setZoom(target)
    }

    func setZoom(_ requested: CGFloat) {
        sessionQueue.async { [weak self] in
            guard let self, let device = self.captureDevice else { return }
            do {
                try device.lockForConfiguration()
                let value = min(
                    max(requested, device.minAvailableVideoZoomFactor),
                    device.maxAvailableVideoZoomFactor
                )
                device.videoZoomFactor = value
                device.unlockForConfiguration()
                DispatchQueue.main.async {
                    self.zoomFactor = value
                    self.diagnostics.recordCameraEvent("zoom_changed_to_\(String(format: "%.2f", value))x")
                }
            } catch {
                DispatchQueue.main.async {
                    self.finderState = .error("ズーム変更に失敗しました: \(error.localizedDescription)")
                    self.diagnostics.recordCameraEvent("zoom_change_failed: \(error.localizedDescription)")
                }
            }
        }
    }

    func resetSearch() {
        invalidateSearch(publishScanning: true)
    }

    func setColorAssistEnabled(_ enabled: Bool) {
        colorAssistEnabled = enabled
        if !enabled {
            colorAssistEffective = false
        }
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            self.colorAssistRequested = enabled
            if !enabled {
                _ = self.scheduler.updateTilePriorities(scores: nil, enabled: false)
                self.latestColorAssistScores = nil
                self.latestSelectedTileOrder = Array(AppConfig.searchTiles.indices)
            }
        }
    }

    func setColorAssistMode(_ mode: ColorAssistMode) {
        colorAssistMode = mode
        inferenceQueue.async { [weak self] in
            self?.selectedColorAssistMode = mode
        }
    }

    func setColorAssistPreviewRequested(_ requested: Bool) {
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            self.colorAssistPreviewRequested = requested
            if !requested {
                self.lastColorAssistPreviewAt = nil
                DispatchQueue.main.async {
                    self.colorAssistPreview = nil
                }
            }
        }
    }

    func annotateBallContainingTile(_ tileIndex: Int?) {
        diagnostics.annotateBallContainingTile(tileIndex)
    }

    private func invalidateSearch(publishScanning: Bool, suspendInference: Bool = false) {
        videoQueue.async { [weak self] in
            guard let self else { return }
            self.searchEpoch += 1
            if suspendInference { self.inferenceSuspended = true }
            self.inferenceQueue.async { [weak self] in
                guard let self else { return }
                self.scheduler.reset()
                self.stabilizer.reset()
                DispatchQueue.main.async {
                    self.diagnostics.resetSearchState()
                    if publishScanning {
                        if self.detector.isLoaded { self.finderState = .scanning }
                    }
                }
            }
        }
    }

    func recordFalsePositive(occlusion: OcclusionBucket, note: String) {
        diagnostics.recordManualEvent(
            kind: .falsePositive,
            image: latestFrame.current(),
            occlusion: occlusion,
            note: note
        )
        resetSearch()
    }

    func recordTruePositive(occlusion: OcclusionBucket, note: String) {
        diagnostics.recordManualEvent(
            kind: .truePositive,
            image: latestFrame.current(),
            occlusion: occlusion,
            note: note
        )
        resetSearch()
    }

    func recordMiss(occlusion: OcclusionBucket, note: String) {
        diagnostics.recordManualEvent(
            kind: .miss,
            image: latestFrame.current(),
            occlusion: occlusion,
            note: note
        )
    }

    func beginFieldScene(id: String, type: FieldSceneType, occlusion: OcclusionBucket, note: String) {
        if diagnostics.beginScene(id: id, type: type, occlusion: occlusion, note: note) {
            resetSearch()
        }
    }

    func endFieldScene(note: String) {
        if diagnostics.endScene(note: note) {
            resetSearch()
        }
    }

    private func requestCameraPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .video)
        default:
            return false
        }
    }

    private func configureAndStartSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.configured {
                do {
                    try self.configureSession()
                    self.configured = true
                } catch {
                    DispatchQueue.main.async {
                        self.finderState = .error("カメラ初期化に失敗しました: \(error.localizedDescription)")
                        self.diagnostics.recordCameraEvent("camera_configuration_failed: \(error.localizedDescription)")
                    }
                    return
                }
            }
            if !self.session.isRunning {
                self.session.startRunning()
            }
            self.videoQueue.async { [weak self] in
                self?.inferenceSuspended = false
            }
            DispatchQueue.main.async {
                self.diagnostics.recordCameraEvent("camera_session_started")
                if self.detector.isLoaded { self.finderState = .scanning }
            }
        }
    }

    private func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    private func configureSession() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1280x720

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CameraError.noBackCamera
        }
        captureDevice = device

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.cannotAddInput }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { throw CameraError.cannotAddOutput }
        session.addOutput(output)

        if let connection = output.connection(with: .video) {
            configurePortraitVideoConnection(connection)
        }
    }

    private func installObservers() {
        let center = NotificationCenter.default
        notificationTokens.append(center.addObserver(
            forName: AVCaptureSession.wasInterruptedNotification,
            object: session,
            queue: .main
        ) { [weak self] notification in
            self?.sessionWasInterrupted(notification)
        })
        notificationTokens.append(center.addObserver(
            forName: AVCaptureSession.interruptionEndedNotification,
            object: session,
            queue: .main
        ) { [weak self] _ in
            self?.sessionInterruptionEnded()
        })
        notificationTokens.append(center.addObserver(
            forName: AVCaptureSession.runtimeErrorNotification,
            object: session,
            queue: .main
        ) { [weak self] notification in
            self?.sessionRuntimeError(notification)
        })
        notificationTokens.append(center.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.applicationDidEnterBackground()
        })
        notificationTokens.append(center.addObserver(
            forName: UIApplication.willEnterForegroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.applicationWillEnterForeground()
        })
        notificationTokens.append(center.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: ProcessInfo.processInfo,
            queue: .main
        ) { [weak self] _ in
            let state = ProcessInfo.processInfo.thermalState
            self?.diagnostics.recordCameraEvent("thermal_state_changed_to_\(state.diagnosticName)", thermal: state)
        })
    }

    private func applicationDidEnterBackground() {
        guard wantsSessionRunning else { return }
        stopSession()
        invalidateSearch(publishScanning: false, suspendInference: true)
        finderState = .paused("バックグラウンドのためカメラを停止中")
        diagnostics.recordCameraEvent("application_entered_background")
    }

    private func applicationWillEnterForeground() {
        guard wantsSessionRunning else { return }
        diagnostics.recordCameraEvent("application_entered_foreground")
        start()
    }

    private func sessionWasInterrupted(_ notification: Notification) {
        let reasonValue = (notification.userInfo?[AVCaptureSessionInterruptionReasonKey] as? NSNumber)?.intValue
        let message = reasonValue.map { "カメラが一時停止しました（reason=\($0)）" } ?? "カメラが一時停止しました"
        finderState = .paused(message)
        invalidateSearch(publishScanning: false, suspendInference: true)
        diagnostics.recordCameraEvent("camera_interrupted_reason_\(reasonValue.map(String.init) ?? "unknown")")
    }

    private func sessionInterruptionEnded() {
        diagnostics.recordCameraEvent("camera_interruption_ended")
        guard wantsSessionRunning else { return }
        if detector.isLoaded { finderState = .scanning }
        configureAndStartSession()
    }

    private func sessionRuntimeError(_ notification: Notification) {
        let error = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
        diagnostics.recordCameraEvent("camera_runtime_error_\(error?.code ?? -1): \(error?.localizedDescription ?? "unknown")")
        if error?.code == AVError.Code.mediaServicesWereReset.rawValue, wantsSessionRunning {
            invalidateSearch(publishScanning: false, suspendInference: true)
            configureAndStartSession()
        } else {
            invalidateSearch(publishScanning: false, suspendInference: true)
            finderState = .error("カメラ実行エラー: \(error?.localizedDescription ?? "不明なエラー")")
        }
    }

    /// Called only on videoQueue by AVCaptureVideoDataOutput.
    private func scheduleInference(pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        let thermal = ProcessInfo.processInfo.thermalState
        let policy = ThermalInferencePolicy.current(for: thermal)
        guard !inferenceSuspended else { return }
        guard frameGate.admit(
            timestamp: timestamp,
            targetFPS: policy.processedFPS,
            detectorReady: detector.isLoaded
        ) else { return }

        let epoch = searchEpoch
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let imageSize = image.extent.size

        inferenceQueue.async { [weak self] in
            guard let self else { return }
            let colorAssistRequested = self.colorAssistRequested
            let selectedColorAssistMode = self.selectedColorAssistMode
            let isROILocked = self.scheduler.isROILocked
            let thermalAllowsColorAssist = thermal.allowsColorAssist
            let colorAssistEffective = colorAssistRequested && thermalAllowsColorAssist
            let shouldCreatePreview = self.colorAssistPreviewRequested
                && thermalAllowsColorAssist
                && (self.lastColorAssistPreviewAt.map {
                    timestamp < $0 || timestamp - $0 >= AppConfig.colorAssistPreviewInterval
                } ?? true)
            let shouldAnalyzeColor = thermalAllowsColorAssist
                && !isROILocked
                && (colorAssistEffective || shouldCreatePreview)

            let colorAnalysis: ColorAssistAnalysis?
            if shouldAnalyzeColor {
                colorAnalysis = self.colorAssistEngine.analyze(
                    image: image,
                    mode: selectedColorAssistMode,
                    tileRegions: AppConfig.searchTiles,
                    includePreview: shouldCreatePreview
                )
                if shouldCreatePreview, colorAnalysis?.preview != nil {
                    self.lastColorAssistPreviewAt = timestamp
                }
            } else {
                colorAnalysis = nil
            }

            let scoreValues = colorAnalysis?.tileScores.map(\.score)
            let selectedTileOrder: [Int]
            if isROILocked {
                selectedTileOrder = self.scheduler.currentTileOrder
            } else {
                selectedTileOrder = self.scheduler.updateTilePriorities(
                    scores: scoreValues,
                    enabled: colorAssistEffective && colorAnalysis != nil
                )
                self.latestSelectedTileOrder = selectedTileOrder
                self.latestColorAssistScores = scoreValues
            }
            let colorAssistOperational = colorAssistEffective
                && (isROILocked || colorAnalysis != nil)
            let diagnosticTileScores = scoreValues ?? self.latestColorAssistScores
            let diagnosticTileOrder = isROILocked
                ? self.latestSelectedTileOrder
                : selectedTileOrder
            let region = self.scheduler.nextRegion()
            let regionRect = region.normalizedRect
            // Baseline invariant: YOLO always receives the original raw camera image or its raw
            // crop. Color Assist outputs are used only by updateTilePriorities above.
            let modelImage = regionRect == CGRect(x: 0, y: 0, width: 1, height: 1)
                ? image
                : image.cropped(normalizedTopLeft: regionRect)

            let prediction = self.detector.predict(
                image: modelImage,
                cropRegion: regionRect,
                fullFrameImageSize: imageSize,
                source: region.source,
                timestamp: timestamp
            )

            let best = prediction.observations.first {
                $0.confidence >= AppConfig.candidateMinConfidence
            }
            if let best {
                if region.source == .roi {
                    self.scheduler.updateLockedROI(with: best)
                } else {
                    self.scheduler.lock(on: best)
                }
            } else if region.source == .roi {
                self.scheduler.updateLockedROI(with: nil)
            }

            let stabilized = self.stabilizer.update(candidates: prediction.observations)

            self.videoQueue.async { [weak self] in
                guard let self else { return }
                self.frameGate.complete()
                let statistics = self.frameGate.statistics
                guard epoch == self.searchEpoch else { return }

                DispatchQueue.main.async {
                    self.sourceImageSize = imageSize
                    self.inferenceMs = prediction.inferenceMs
                    self.currentSource = region.source
                    self.finderState = stabilized.state
                    self.colorAssistEffective = colorAssistOperational
                    if let preview = colorAnalysis?.preview {
                        self.colorAssistPreview = preview
                    }
                    self.diagnostics.recordInference(
                        latencyMs: prediction.inferenceMs,
                        source: region.source,
                        state: stabilized.state,
                        thermal: thermal,
                        targetFPS: policy.processedFPS,
                        zoom: self.zoomFactor,
                        statistics: statistics,
                        colorAssistRequested: colorAssistRequested,
                        colorAssistEnabled: colorAssistOperational,
                        colorAssistMode: selectedColorAssistMode,
                        colorProcessingLatencyMs: colorAnalysis?.processingLatencyMs,
                        tileSaliencyScores: diagnosticTileScores,
                        selectedTileOrder: diagnosticTileOrder,
                        candidateTrackStartedAt: stabilized.candidateTrackStartedAt,
                        confirmationLatencyMs: stabilized.confirmationLatencyMs
                    )
                    if stabilized.shouldTriggerFeedback {
                        FeedbackManager.shared.ballFound()
                    }
                }
            }
        }
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        latestFrame.update(pixelBuffer: pixelBuffer)
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        scheduleInference(pixelBuffer: pixelBuffer, timestamp: timestamp)
    }
}

func configurePortraitVideoConnection(_ connection: AVCaptureConnection) {
    if connection.isVideoRotationAngleSupported(90) {
        connection.videoRotationAngle = 90
    }
}

enum CameraError: LocalizedError {
    case noBackCamera
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .noBackCamera: return "背面カメラが見つかりません"
        case .cannotAddInput: return "カメラ入力をセッションへ追加できません"
        case .cannotAddOutput: return "映像出力をセッションへ追加できません"
        }
    }
}
