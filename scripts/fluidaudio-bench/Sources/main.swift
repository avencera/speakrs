import FluidAudio
import Foundation

@available(macOS 14.0, *)
enum Bench {
    static func run() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard !args.isEmpty else {
            FileHandle.standardError.write(
                Data("Usage: fluidaudio-bench <file1.wav> [file2.wav ...]\n".utf8))
            Foundation.exit(1)
        }

        // Validate all files exist before loading models
        for path in args {
            guard FileManager.default.fileExists(atPath: path) else {
                FileHandle.standardError.write(Data("File not found: \(path)\n".utf8))
                Foundation.exit(1)
            }
        }

        // Load models once
        let modelLoadStart = Date()
        let config = OfflineDiarizerConfig()
        let manager = OfflineDiarizerManager(config: config)
        try await manager.prepareModels()
        let modelLoadTime = Date().timeIntervalSince(modelLoadStart)
        let modelLoadStr = String(format: "%.3f", modelLoadTime)
        FileHandle.standardError.write(Data("models loaded: \(modelLoadStr)s\n".utf8))

        let totalStart = Date()
        let totalFiles = args.count
        var cumulative = 0.0

        for (i, path) in args.enumerated() {
            let url = URL(fileURLWithPath: path)
            let stem = url.deletingPathExtension().lastPathComponent

            let fileStart = Date()
            let result = try await manager.process(url)
            let elapsed = Date().timeIntervalSince(fileStart)
            cumulative += elapsed

            let avg = cumulative / Double(i + 1)
            let remaining = Double(totalFiles - i - 1) * avg
            let eta = formatEta(remaining)
            FileHandle.standardError.write(
                Data("  [\(i + 1)/\(totalFiles)] \(stem): \(String(format: "%.1f", elapsed))s (ETA \(eta))\n".utf8))

            // RTTM output to stdout
            for seg in result.segments {
                let start = String(format: "%.3f", seg.startTimeSeconds)
                let duration = String(format: "%.3f", seg.durationSeconds)
                print("SPEAKER \(stem) 1 \(start) \(duration) <NA> <NA> \(seg.speakerId) <NA> <NA>")
            }
        }

        let totalElapsed = Date().timeIntervalSince(totalStart)
        let totalStr = String(format: "%.3f", totalElapsed)
        FileHandle.standardError.write(Data("total: \(totalStr)s\n".utf8))
    }
}

@available(macOS 14.0, *)
func formatEta(_ seconds: Double) -> String {
    if seconds < 60 {
        return String(format: "%.0fs", seconds)
    }
    let mins = Int(seconds / 60)
    let secs = Int(seconds.truncatingRemainder(dividingBy: 60).rounded())
    return String(format: "%dm %02ds", mins, secs)
}

if #available(macOS 14.0, *) {
    do {
        try await Bench.run()
    } catch {
        FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
        Foundation.exit(1)
    }
} else {
    FileHandle.standardError.write(Data("error: macOS 14+ required\n".utf8))
    Foundation.exit(1)
}
