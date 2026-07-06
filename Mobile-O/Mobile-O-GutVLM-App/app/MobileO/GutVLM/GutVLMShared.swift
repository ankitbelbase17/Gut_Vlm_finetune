import SwiftUI
import PhotosUI
import UIKit

// MARK: - Image loading

enum GutVLMImage {
    /// Load a picked photo into a `PlatformImage`, downsampled so its longest
    /// edge is at most `maxDimension` points (keeps memory / preprocessing sane).
    static func load(from item: PhotosPickerItem?, maxDimension: CGFloat = 2048) async -> PlatformImage? {
        guard let item,
              let data = try? await item.loadTransferable(type: Data.self),
              let image = UIImage(data: data) else { return nil }
        return downsample(image, maxDimension: maxDimension)
    }

    static func downsample(_ image: UIImage, maxDimension: CGFloat) -> UIImage {
        let size = image.size
        guard max(size.width, size.height) > maxDimension else { return image }
        let scale = maxDimension / max(size.width, size.height)
        let newSize = CGSize(width: size.width * scale, height: size.height * scale)
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        image.draw(in: CGRect(origin: .zero, size: newSize))
        return UIGraphicsGetImageFromCurrentImageContext() ?? image
    }
}

// MARK: - Image picker card

/// A reusable "tap to pick / preview" endoscopy image card used by both tabs.
struct ImagePickerCard: View {
    @Binding var image: PlatformImage?
    @Binding var pickerItem: PhotosPickerItem?

    var body: some View {
        PhotosPicker(selection: $pickerItem, matching: .images) {
            ZStack {
                RoundedRectangle(cornerRadius: 16)
                    .fill(Color(.secondarySystemBackground))
                    .frame(height: 240)

                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 240)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "photo.badge.plus")
                            .font(.system(size: 40))
                            .foregroundStyle(.secondary)
                        Text("Tap to select an endoscopy image")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Result card

/// A titled card that shows a block of result text.
struct ResultCard<Content: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            content
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
    }
}

// MARK: - Run button

struct RunButton: View {
    let title: String
    let systemImage: String
    let isRunning: Bool
    let isEnabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                if isRunning {
                    ProgressView().tint(.white)
                } else {
                    Image(systemName: systemImage)
                }
                Text(isRunning ? "Working..." : title)
                    .font(.headline)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
        }
        .buttonStyle(.borderedProminent)
        .disabled(!isEnabled || isRunning)
    }
}
