// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "fluidaudio-bench",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "/Users/praveen/code/fluid-audio"),
    ],
    targets: [
        .executableTarget(
            name: "fluidaudio-bench",
            dependencies: [
                .product(name: "FluidAudio", package: "fluid-audio"),
            ],
            path: "Sources"
        ),
    ]
)
