import SwiftUI
import PhotosUI

/// Hallucination-aware tab: pick an endoscopy image + paste an AI-generated
/// caption. The model (1) tags each sentence as hallucinated / non-hallucinated
/// and (2) produces a corrected caption. Mirrors the two-turn Gut-VLM format.
struct HallucinationView: View {
    let model: GutVLMModel

    @State private var pickerItem: PhotosPickerItem?
    @State private var image: PlatformImage?
    @State private var caption = ""
    @State private var result: HallucinationResult?
    @State private var isRunning = false
    @State private var errorMessage: String?

    private var canRun: Bool {
        image != nil && model.isReady &&
        !caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    ImagePickerCard(image: $image, pickerItem: $pickerItem)

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Caption to check")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)
                        TextField("Paste an AI-generated report/caption...", text: $caption, axis: .vertical)
                            .lineLimit(3...8)
                            .padding(12)
                            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
                    }

                    RunButton(title: "Detect & Correct", systemImage: "checkmark.shield.fill",
                              isRunning: isRunning, isEnabled: canRun, action: run)

                    if let errorMessage {
                        Text(errorMessage)
                            .font(.callout)
                            .foregroundStyle(.red)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    if let result {
                        ResultCard(title: "Sentence analysis", systemImage: "list.bullet.rectangle.fill") {
                            DetectionView(detection: result.detection)
                        }
                        ResultCard(title: "Corrected caption", systemImage: "wand.and.stars") {
                            Text(result.correction.isEmpty ? "(no correction returned)" : result.correction)
                                .textSelection(.enabled)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Hallucination")
            .background(Color(.systemGroupedBackground))
            .onChange(of: pickerItem) { _, newItem in
                Task {
                    image = await GutVLMImage.load(from: newItem)
                    result = nil
                    errorMessage = nil
                }
            }
        }
    }

    private func run() {
        guard let image else { return }
        let cap = caption
        isRunning = true
        result = nil
        errorMessage = nil
        Task {
            do {
                result = try await model.runHallucination(image: image, caption: cap)
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
        }
    }
}

/// Renders the per-sentence detection block, colour-coding each line by its tag.
private struct DetectionView: View {
    let detection: String

    private struct Line: Identifiable {
        let id = UUID()
        let text: String
        let isHallucinated: Bool
        let isTagged: Bool
    }

    private var lines: [Line] {
        detection
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { raw in
                let s = String(raw)
                // Check the negative tag first: "<non-hallucinated>" does not
                // contain the substring "<hallucinated>".
                if s.contains("<non-hallucinated>") {
                    return Line(text: s, isHallucinated: false, isTagged: true)
                } else if s.contains("<hallucinated>") {
                    return Line(text: s, isHallucinated: true, isTagged: true)
                } else {
                    return Line(text: s, isHallucinated: false, isTagged: false)
                }
            }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(lines) { line in
                HStack(alignment: .top, spacing: 8) {
                    if line.isTagged {
                        Image(systemName: line.isHallucinated ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                            .foregroundStyle(line.isHallucinated ? .red : .green)
                    }
                    Text(cleanLine(line.text))
                        .textSelection(.enabled)
                        .foregroundStyle(line.isTagged ? (line.isHallucinated ? .red : .primary) : .primary)
                }
            }
        }
    }

    /// Drop the trailing tag markers for display (the icon already conveys them).
    private func cleanLine(_ s: String) -> String {
        s.replacingOccurrences(of: "<non-hallucinated>", with: "")
         .replacingOccurrences(of: "<hallucinated>", with: "")
         .trimmingCharacters(in: .whitespaces)
    }
}
