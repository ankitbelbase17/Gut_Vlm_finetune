import Foundation
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM
import Tokenizers
import UIKit

// MARK: - Prompt templates

/// Exact prompt strings used during finetuning on Clariden (mirror of the Python
/// `inference.py`). Reproducing these byte-for-byte on-device is important: the
/// model is a small 0.5B network finetuned narrowly, so it is sensitive to the
/// system framing and turn delimiters it saw in training.
enum GutVLMPrompts {

    /// Hallucination-detection system framing (identical to inference.py SYSTEM_PREFIX).
    static let systemPrefix = """
    A chat between a user and an artificial intelligence assistant expert in \
    Gastrointestinal endoscopic images. The assistant is tasked with detecting \
    hallucinated sentences in a given caption. Hallucination occurs when the \
    caption is incorrect, misleading, or non-existent information that is not \
    grounded in the input image or context. Hallucinations include factual \
    errors, misidentification of anatomy, false detection of abnormalities, \
    incorrect reasoning, and nonexistent instruments or conditions like \
    bleeding, infection, or inflammation.\n\n
    """

    /// Single-turn VQA. Note: no system prompt, and the user turn is closed with
    /// `<|im_start|>assistant` (not `<|im_end|>`) - exactly how training built it.
    static func vqa(question: String) -> String {
        "<|im_start|>user\n<image>\n\(question)<|im_start|>assistant\n"
    }

    /// Hallucination detection - turn 1 (tag each caption sentence).
    static func hallucinationDetect(caption: String) -> String {
        "<|im_start|>user\n\(systemPrefix)<image>\nCaption: \(caption)\n\n"
        + "Can you detect which sentences are hallucinated in the given caption?"
        + "<|im_start|>assistant\n"
    }

    /// Hallucination correction - turn 2 (replays turn 1, asks for a fix).
    /// The image appears only once, in the first user turn - matches training.
    static func hallucinationCorrect(caption: String, detection: String) -> String {
        "<|im_start|>user\n\(systemPrefix)<image>\nCaption: \(caption)\n\n"
        + "Can you detect which sentences are hallucinated in the given caption?"
        + "<|im_start|>assistant\n\(detection)<|im_end|>\n"
        + "<|im_start|>user\n"
        + "Can you please correct any hallucinated sentences and generate a modified response?"
        + "<|im_start|>assistant\n"
    }

    /// Strip any residual special tokens / whitespace from a decoded generation.
    static func cleanup(_ s: String) -> String {
        s.replacingOccurrences(of: "<|im_end|>", with: "")
         .replacingOccurrences(of: "<|im_start|>", with: "")
         .replacingOccurrences(of: "<|endoftext|>", with: "")
         .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Remove the "Modified caption:" prefix the model was trained to emit.
    static func stripModifiedPrefix(_ s: String) -> String {
        let marker = "modified caption:"
        if s.lowercased().hasPrefix(marker) {
            return String(s.dropFirst(marker.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return s
    }
}

// MARK: - Result types

struct HallucinationResult {
    let detection: String   // per-sentence tag block
    let correction: String  // corrected caption (prefix stripped)
}

// MARK: - Model

/// Runs the two GutVLM understanding tasks against the shared FastVLM container:
///   - VQA (single-turn)
///   - Hallucination detection + correction (two sequential turns)
///
/// Generation is greedy (temperature 0) to match the deterministic Clariden
/// `do_sample=False` inference and to keep clinical outputs reproducible.
@Observable
@MainActor
final class GutVLMModel {

    private(set) var isReady = false
    /// Human-readable status/error for the UI.
    var status = "Loading model..."

    private var container: ModelContainer?

    init() {}

    func attach(container: ModelContainer) {
        self.container = container
        self.isReady = true
        self.status = "Ready"
    }

    // MARK: Public task API

    /// Single-turn visual question answering.
    nonisolated func runVQA(image: PlatformImage, question: String) async throws -> String {
        let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
        let prompt = GutVLMPrompts.vqa(question: q.isEmpty ? "What do you see in this endoscopy image?" : q)
        let raw = try await generate(rawPrompt: prompt, image: image, maxTokens: 256)
        return GutVLMPrompts.cleanup(raw)
    }

    /// Two-turn hallucination detection then correction.
    nonisolated func runHallucination(image: PlatformImage, caption: String) async throws -> HallucinationResult {
        let cap = caption.trimmingCharacters(in: .whitespacesAndNewlines)

        // Turn 1: detect
        let detectPrompt = GutVLMPrompts.hallucinationDetect(caption: cap)
        let detection = GutVLMPrompts.cleanup(try await generate(rawPrompt: detectPrompt, image: image, maxTokens: 512))

        // Turn 2: correct (replays turn 1)
        let correctPrompt = GutVLMPrompts.hallucinationCorrect(caption: cap, detection: detection)
        var correction = GutVLMPrompts.cleanup(try await generate(rawPrompt: correctPrompt, image: image, maxTokens: 512))
        correction = GutVLMPrompts.stripModifiedPrefix(correction)

        return HallucinationResult(detection: detection, correction: correction)
    }

    // MARK: Core generation

    /// Run one greedy generation over a fully-formatted raw prompt + image.
    /// The raw prompt contains a single `<image>` placeholder; FastVLMProcessor
    /// (patched) passes it through verbatim, only expanding the image tokens.
    nonisolated private func generate(
        rawPrompt: String,
        image: PlatformImage,
        maxTokens: Int
    ) async throws -> String {
        let container = await self.container
        guard let container else {
            throw NSError(domain: "GutVLMModel", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "GutVLMModel", code: -2,
                          userInfo: [NSLocalizedDescriptionKey: "Could not read the selected image"])
        }

        let userInput = UserInput(
            prompt: .text(rawPrompt),
            images: [.ciImage(CIImage(cgImage: cgImage))]
        )

        var output = ""
        try await container.perform { context in
            guard let processor = context.processor as? UserInputProcessor else {
                throw NSError(domain: "GutVLMModel", code: -3,
                              userInfo: [NSLocalizedDescriptionKey: "Invalid processor type"])
            }

            let prepared = try await processor.prepare(input: userInput)

            let result = try MLXLMCommon.generate(
                input: prepared,
                parameters: GenerateParameters(temperature: 0),   // greedy == do_sample=False
                context: context
            ) { tokens in
                if Task.isCancelled { return .stop }
                return tokens.count >= maxTokens ? .stop : .more
            }

            output = result.output
        }
        return output
    }
}
