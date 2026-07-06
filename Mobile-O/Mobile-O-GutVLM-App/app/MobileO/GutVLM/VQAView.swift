import SwiftUI
import PhotosUI

/// Visual Question Answering tab: pick an endoscopy image, ask a question,
/// get a single-turn answer from the finetuned model.
struct VQAView: View {
    let model: GutVLMModel

    @State private var pickerItem: PhotosPickerItem?
    @State private var image: PlatformImage?
    @State private var question = "Is there a polyp visible in this endoscopy image?"
    @State private var answer = ""
    @State private var isRunning = false
    @State private var errorMessage: String?

    private var canRun: Bool { image != nil && model.isReady }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    ImagePickerCard(image: $image, pickerItem: $pickerItem)

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Question")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)
                        TextField("Ask about the image...", text: $question, axis: .vertical)
                            .lineLimit(1...4)
                            .padding(12)
                            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
                    }

                    RunButton(title: "Ask", systemImage: "questionmark.bubble.fill",
                              isRunning: isRunning, isEnabled: canRun, action: run)

                    if let errorMessage {
                        Text(errorMessage)
                            .font(.callout)
                            .foregroundStyle(.red)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    if !answer.isEmpty {
                        ResultCard(title: "Answer", systemImage: "text.bubble.fill") {
                            Text(answer).textSelection(.enabled)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Visual QA")
            .background(Color(.systemGroupedBackground))
            .onChange(of: pickerItem) { _, newItem in
                Task {
                    image = await GutVLMImage.load(from: newItem)
                    answer = ""
                    errorMessage = nil
                }
            }
        }
    }

    private func run() {
        guard let image else { return }
        let q = question
        isRunning = true
        answer = ""
        errorMessage = nil
        Task {
            do {
                answer = try await model.runVQA(image: image, question: q)
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
        }
    }
}
