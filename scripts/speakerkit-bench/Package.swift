// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "speakerkit-bench",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.12.0"),
    ],
    targets: [
        .executableTarget(
            name: "speakerkit-bench",
            dependencies: [
                .product(name: "SpeakerKit", package: "WhisperKit"),
                .product(name: "WhisperKit", package: "WhisperKit"),
            ],
            path: "Sources"
        ),
    ]
)
