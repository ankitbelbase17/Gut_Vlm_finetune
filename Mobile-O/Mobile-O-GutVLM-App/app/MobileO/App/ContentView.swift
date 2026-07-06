import SwiftUI
import MLXLMCommon
import MLXVLM
import Tokenizers

/// Root view of the GutVLM app.
///
/// Loads a single shared FastVLM container (vision encoder + LLM only - no
/// generation components) and presents two tabs backed by the same model:
///   - VQA: single-turn visual question answering
///   - Hallucination: two-turn detection + correction
struct ContentView: View {

    /// Directory containing downloaded model weights. Passed from the download gate.
    let modelDirectory: URL

    @State private var model = GutVLMModel()
    @State private var loaded = false
    @State private var loadError: String?

    var body: some View {
        Group {
            if loaded {
                TabView {
                    VQAView(model: model)
                        .tabItem { Label("VQA", systemImage: "questionmark.bubble") }
                    HallucinationView(model: model)
                        .tabItem { Label("Hallucination", systemImage: "checkmark.shield") }
                }
            } else if let loadError {
                ContentUnavailableView {
                    Label("Failed to load model", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(loadError)
                }
            } else {
                LoadingOverlay()
            }
        }
        .task { await loadModel() }
    }

    // MARK: - Model loading

    /// Loads the shared FastVLM container once (understanding path only).
    private func loadModel() async {
        guard !loaded else { return }

        do {
            let llmDirectory = modelDirectory.appendingPathComponent("llm")

            // Copy bundled preprocessor_config.json into the downloaded llm/ dir if missing.
            let preprocDest = llmDirectory.appendingPathComponent("preprocessor_config.json")
            if !FileManager.default.fileExists(atPath: preprocDest.path),
               let bundled = Bundle.main.url(forResource: "preprocessor_config", withExtension: "json") {
                try? FileManager.default.copyItem(at: bundled, to: preprocDest)
            }

            let config = ModelConfiguration(directory: llmDirectory)

            // Point FastVLM to the downloaded model directory for config + vision encoder.
            FastVLM.customModelDirectory = llmDirectory
            FastVLM.register(modelFactory: VLMModelFactory.shared)

            let container = try await VLMModelFactory.shared.loadContainer(configuration: config) { progress in
                Task { @MainActor in print("Loading FastVLM: \(Int(progress.fractionCompleted * 100))%") }
            }

            // Warm up the LLM to remove cold-start latency.
            try await container.perform { context in
                if let fastVLM = context.model as? FastVLM {
                    fastVLM.warmup()
                }
            }

            model.attach(container: container)
            loaded = true

        } catch {
            print("ContentView ERROR: Failed to load model: \(error)")
            loadError = error.localizedDescription
        }
    }
}

#Preview {
    ContentView(modelDirectory: URL(fileURLWithPath: Bundle.main.resourcePath ?? "/tmp"))
}
