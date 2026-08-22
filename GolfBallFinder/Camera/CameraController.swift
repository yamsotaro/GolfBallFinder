import AVFoundation
import CoreImage
import Foundation
import SwiftUI

final class CameraController: NSObject, ObservableObject {
    @Published private(set) var finderState: FinderState = .loading
    @Published private(set) var sourceImageSize: CGSize = .zero
    @Published private(set) var inferenceMs: Double = 0
    @Published private(set) var currentSource: ScanSource = .fullFrame
    @Published private(set) var zoomFactor: CGFloat = 1.0

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "GolfBallFinder.camera.session")
    private let videoQueue = DispatchQueue(label: "GolfBallFinder.camera.frames", qos: .userInteractive)
    private let inferenceQueue = DispatchQueue(label: "GolfBallFinder.inference", qos: .userInitiated)
    private let detector = GolfBallDetector()

    // Accessed only from inferenceQueue.
    private var scheduler = ScanScheduler()
    private var stabilizer = DetectionStabilizer()

    // Accessed only from videoQueue.
    private var lastProcessedTime: TimeInterval = 0
    private var inferenceBusy = false

    // Accessed only from sessionQueue after configuration.
    private var captureDevice: AVCaptureDevice?
    private var configured = false

    override init() {
        super.init()
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
                        "GolfBallモデルを読み込めません。GolfBall.mlpackageをResourcesに追加してください。\n\n\(error.localizedDescription)"
                    )
                }
            }
        }
    }

    func start() {
        Task { [weak self] in
            guard let self else { return }
            let granted = await self.requestCameraPermission()
            guard granted else {
                DispatchQueue.main.async {
                    self.finderState = .error("カメラ権限が必要です。設定 > プライバシーとセキュリティ > カメラで許可してください。")
                }
                return
            }
            self.configureAndStartSession()
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
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
                let value = min(max(requested, device.minAvailableVideoZoomFactor), device.maxAvailableVideoZoomFactor)
                device.videoZoomFactor = value
                device.unlockForConfiguration()
                DispatchQueue.main.async { self.zoomFactor = value }
            } catch {
                DispatchQueue.main.async {
                    self.finderState = .error("ズーム変更に失敗しました: \(error.localizedDescription)")
                }
            }
        }
    }

    func resetSearch() {
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            self.scheduler = ScanScheduler()
            self.stabilizer.reset()
            DispatchQueue.main.async {
                if self.detector.isLoaded { self.finderState = .scanning }
            }
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
                    }
                    return
                }
            }
            guard !self.session.isRunning else { return }
            self.session.startRunning()
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

        if let connection = output.connection(with: .video), connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
    }

    /// Called only on videoQueue by AVCaptureVideoDataOutput.
    private func scheduleInference(pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        let minInterval = 1.0 / AppConfig.processedFPS
        guard timestamp - lastProcessedTime >= minInterval else { return }
        guard !inferenceBusy, detector.isLoaded else { return }
        lastProcessedTime = timestamp
        inferenceBusy = true

        // CIImage retains the CVPixelBuffer backing store; no CPU base-address lock is required.
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let imageSize = image.extent.size

        inferenceQueue.async { [weak self] in
            guard let self else { return }
            let region = self.scheduler.nextRegion()
            let regionRect = region.normalizedRect
            let modelImage = regionRect == CGRect(x: 0, y: 0, width: 1, height: 1)
                ? image
                : image.cropped(normalizedTopLeft: regionRect)

            let prediction = self.detector.predict(
                image: modelImage,
                cropRegion: regionRect,
                source: region.source,
                timestamp: timestamp
            )

            // Do not spend the ROI budget on detections too weak to enter temporal confirmation.
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

            DispatchQueue.main.async {
                self.sourceImageSize = imageSize
                self.inferenceMs = prediction.inferenceMs
                self.currentSource = region.source
                self.finderState = stabilized.state
                if stabilized.shouldTriggerFeedback {
                    FeedbackManager.shared.ballFound()
                }
            }

            self.videoQueue.async { [weak self] in
                self?.inferenceBusy = false
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
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        scheduleInference(pixelBuffer: pixelBuffer, timestamp: timestamp)
    }
}

enum CameraError: LocalizedError {
    case noBackCamera
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .noBackCamera: return "背面カメラが見つかりません"
        case .cannotAddInput: return "カメラ入力をセッションに追加できません"
        case .cannotAddOutput: return "映像出力をセッションに追加できません"
        }
    }
}
