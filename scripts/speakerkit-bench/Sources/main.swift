import Foundation
import SpeakerKit
import WhisperKit

@available(macOS 14.0, *)
enum Bench {
    static func run() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        guard !args.isEmpty else {
            FileHandle.standardError.write(
                Data("Usage: speakerkit-bench <file1.wav> [file2.wav ...]\n".utf8))
            Foundation.exit(1)
        }

        for path in args {
            guard FileManager.default.fileExists(atPath: path) else {
                FileHandle.standardError.write(Data("File not found: \(path)\n".utf8))
                Foundation.exit(1)
            }
        }

        // Load models once
        let modelLoadStart = Date()
        let speakerKit = try await SpeakerKit(PyannoteConfig())
        let modelLoadTime = Date().timeIntervalSince(modelLoadStart)
        let modelLoadStr = String(format: "%.3f", modelLoadTime)
        FileHandle.standardError.write(Data("models loaded: \(modelLoadStr)s\n".utf8))

        let totalStart = Date()
        let totalFiles = args.count
        var cumulative = 0.0
        let profileFileTiming = ProcessInfo.processInfo.environment["SPEAKERKIT_PROFILE_FILE_TIMING"] != nil

        for (i, path) in args.enumerated() {
            let url = URL(fileURLWithPath: path)
            let stem = url.deletingPathExtension().lastPathComponent

            let fileStart = Date()
            let loadStart = Date()
            let audioArray = try AudioProcessor.loadAudioAsFloatArray(fromPath: path)
            let loadTime = Date().timeIntervalSince(loadStart)
            // match whisperkit-cli diarize defaults (useExclusiveReconciliation=false)
            let options = PyannoteDiarizationOptions(useExclusiveReconciliation: false)
            let diarizeStart = Date()
            let result = try await speakerKit.diarize(audioArray: audioArray, options: options)
            let diarizeTime = Date().timeIntervalSince(diarizeStart)
            let elapsed = Date().timeIntervalSince(fileStart)
            cumulative += elapsed

            let avg = cumulative / Double(i + 1)
            let remaining = Double(totalFiles - i - 1) * avg
            let eta = formatEta(remaining)
            let timeFmt = DateFormatter()
            timeFmt.dateFormat = "HH:mm:ss"
            let now = timeFmt.string(from: Date())
            FileHandle.standardError.write(
                Data("  [\(i + 1)/\(totalFiles)] \(stem): \(String(format: "%.1f", elapsed))s (ETA \(eta)) [\(now)]\n".utf8))
            if profileFileTiming {
                let fullPipelineMs = result.timings.map { Int($0.fullPipeline.rounded()) } ?? 0
                let segmenterMs = result.timings.map { Int($0.segmenterTime.rounded()) } ?? 0
                let embedderMs = result.timings.map { Int($0.embedderTime.rounded()) } ?? 0
                let clusteringMs = result.timings.map { Int($0.clusteringTime.rounded()) } ?? 0
                let chunkCount = result.timings.map { $0.numberOfChunks } ?? 0
                let embedderWorkers = result.timings.map { $0.numberOfEmbedderWorkers } ?? 0
                let line = String(
                    format: "    file timing %@: load_ms=%d diarize_ms=%d full_pipeline_ms=%d segmenter_ms=%d embedder_ms=%d clustering_ms=%d chunks=%d embedder_workers=%d\n",
                    stem,
                    Int((loadTime * 1000).rounded()),
                    Int((diarizeTime * 1000).rounded()),
                    fullPipelineMs,
                    segmenterMs,
                    embedderMs,
                    clusteringMs,
                    chunkCount,
                    embedderWorkers
                )
                FileHandle.standardError.write(Data(line.utf8))
            }

            // RTTM output to stdout
            let rttmLines = SpeakerKit.generateRTTM(from: result, fileName: stem)
            for line in rttmLines {
                print(line)
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
