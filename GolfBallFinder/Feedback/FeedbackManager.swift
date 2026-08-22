import AudioToolbox
import UIKit

final class FeedbackManager {
    static let shared = FeedbackManager()
    private let generator = UINotificationFeedbackGenerator()

    private init() {
        generator.prepare()
    }

    func ballFound() {
        generator.notificationOccurred(.success)
        AudioServicesPlaySystemSound(1104)
        generator.prepare()
    }
}
