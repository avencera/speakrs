# Examples

These examples cover the basic pyannote-style workflows:

- run diarization on one WAV file
- iterate speaker turns
- compute speaker airtime
- reconcile transcript chunks with diarization output

All examples expect:

- a models directory containing `segmentation-3.0.onnx`, `wespeaker-voxceleb-resnet34.onnx`, and the PLDA `.npy` files
- a mono 16kHz 16-bit PCM WAV input

Populate `fixtures/models` with:

```bash
just export-models
```

## Run diarization and print RTTM

```bash
cargo run --example diarize_wav -- fixtures/models fixtures/test.wav
```

Optional file id:

```bash
cargo run --example diarize_wav -- fixtures/models fixtures/test.wav meeting-42
```

## Print speaker turns

This matches pyannote's `for turn, speaker in output.speaker_diarization` pattern.

```bash
cargo run --example print_turns -- fixtures/models fixtures/test.wav
```

Output:

```text
start   end     speaker
4.705   46.690  SPEAKER_00
...
```

## Compute speaker airtime

Useful if you want per-speaker totals.

```bash
cargo run --example speaker_airtime -- fixtures/models fixtures/test.wav
```

Output:

```text
speaker     total_seconds
SPEAKER_00  341.245
...
```

## Assign speakers to transcript chunks

This example uses exclusive diarization to assign one speaker to each transcript chunk.

Transcript input is a tab-separated file with:

```text
start_seconds<TAB>end_seconds<TAB>text
```

Example transcript:

```text
0.000	3.200	Welcome everyone
3.200	7.800	Let's get started
```

Run it with:

```bash
cargo run --example assign_transcript_speakers -- fixtures/models fixtures/test.wav transcript.tsv
```

Output:

```text
start   end     speaker     text
0.000   3.200   SPEAKER_00  Welcome everyone
3.200   7.800   SPEAKER_01  Let's get started
```
