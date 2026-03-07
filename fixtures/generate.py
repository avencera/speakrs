# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyannote.audio>=3.3",
#     "torch",
#     "numpy",
#     "soundfile",
#     "huggingface-hub",
#     "onnx",
#     "onnxruntime",
# ]
# ///
"""Generate golden test fixtures for speakrs parity testing.

Usage:
    uv run fixtures/generate.py

Requires HF_TOKEN env var with access to pyannote models.
Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch


def generate_model_free_fixtures(output: Path):
    """Generate fixtures that don't need any models"""
    from pyannote.audio.utils.powerset import Powerset

    print("=== Generating model-free fixtures ===")

    # powerset mapping matrices
    for nc in range(2, 5):
        for ms in range(1, nc + 1):
            ps = Powerset(nc, ms)
            np.save(output / f"powerset_mapping_{nc}_{ms}.npy", ps.mapping.numpy())

    # powerset decode test with synthetic logits
    ps3 = Powerset(num_classes=3, max_set_size=2)
    rng = np.random.RandomState(42)
    logits = rng.randn(1, 589, 7).astype(np.float32)
    np.save(output / "powerset_input_logits.npy", logits)
    np.save(
        output / "powerset_hard_output.npy",
        ps3.to_multilabel(torch.from_numpy(logits), soft=False).numpy(),
    )
    np.save(
        output / "powerset_soft_output.npy",
        ps3.to_multilabel(torch.from_numpy(logits), soft=True).numpy(),
    )

    # cosine similarity
    vecs = rng.randn(10, 256).astype(np.float32)
    vecs_norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    cos_sim = vecs_norm @ vecs_norm.T
    np.save(output / "cosine_sim_input.npy", vecs)
    np.save(output / "cosine_sim_expected.npy", cos_sim)

    print(f"  Model-free fixtures saved to {output}")


def get_test_audio(output: Path) -> Path:
    """Get a real speech test WAV for pipeline testing"""
    import soundfile as sf
    import urllib.request

    wav_path = output / "test.wav"
    if wav_path.exists():
        wav_path.unlink()

    import subprocess
    import tempfile

    sr = 16000

    with tempfile.TemporaryDirectory() as tmpdir:
        # generate two different speakers using macOS voices
        aiff1 = os.path.join(tmpdir, "s1.aiff")
        aiff2 = os.path.join(tmpdir, "s2.aiff")

        text1 = (
            "Hello, welcome to our meeting today. "
            "I would like to discuss the quarterly results and our plans for next year. "
            "The revenue has grown by fifteen percent compared to last quarter."
        )
        text2 = (
            "Thank you for the introduction. "
            "I agree that the results are very promising. "
            "Let me share some details about the engineering progress we have made."
        )

        subprocess.run(
            ["say", "-v", "Samantha", "-o", aiff1, "--rate", "180", text1],
            check=True,
        )
        subprocess.run(
            ["say", "-v", "Daniel", "-o", aiff2, "--rate", "170", text2],
            check=True,
        )

        # convert to 16kHz mono wav using afconvert
        wav1 = os.path.join(tmpdir, "s1.wav")
        wav2 = os.path.join(tmpdir, "s2.wav")
        for src, dst in [(aiff1, wav1), (aiff2, wav2)]:
            subprocess.run(
                [
                    "afconvert",
                    "-f", "WAVE",
                    "-d", "LEI16@16000",
                    "-c", "1",
                    src,
                    dst,
                ],
                check=True,
            )

        audio1, _ = sf.read(wav1, dtype="float32")
        audio2, _ = sf.read(wav2, dtype="float32")

    # arrange: speaker 0 talks, then speaker 1, then both overlap, then speaker 0 again
    gap = np.zeros(int(0.5 * sr), dtype=np.float32)  # 0.5s silence gap

    overlap_len = min(len(audio1) // 4, len(audio2) // 4, 3 * sr)

    # build timeline:
    # [0..len1]: speaker 0 only
    # [len1..len1+gap]: silence
    # [len1+gap..len1+gap+len2]: speaker 1 only
    # [len1+gap+len2..+overlap]: both speakers overlapping
    # [+overlap..+gap+tail]: speaker 0 final segment

    total = len(audio1) + len(gap) + len(audio2) + overlap_len + len(gap) + len(audio1) // 2
    audio = np.zeros(total, dtype=np.float32)

    pos = 0
    # speaker 0 first segment
    audio[pos : pos + len(audio1)] += audio1
    pos += len(audio1) + len(gap)

    # speaker 1 segment
    audio[pos : pos + len(audio2)] += audio2
    pos += len(audio2)

    # overlap section: both speakers
    s0_overlap = audio1[: overlap_len]
    s1_overlap = audio2[: overlap_len]
    audio[pos : pos + overlap_len] += s0_overlap * 0.7
    audio[pos : pos + overlap_len] += s1_overlap * 0.7
    pos += overlap_len + len(gap)

    # speaker 0 final segment
    tail = audio1[: len(audio1) // 2]
    end = min(pos + len(tail), total)
    audio[pos : end] += tail[: end - pos]

    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(wav_path), audio, sr)
    print(f"  Generated TTS test audio ({len(audio)/sr:.1f}s): {wav_path}")
    return wav_path


def tts_to_wav(voice: str, text: str, tmpdir: str, name: str, rate: int = 180) -> str:
    """Generate speech with macOS TTS and convert to 16kHz mono WAV"""
    import subprocess

    aiff = os.path.join(tmpdir, f"{name}.aiff")
    wav = os.path.join(tmpdir, f"{name}.wav")

    subprocess.run(
        ["say", "-v", voice, "-o", aiff, "--rate", str(rate), text],
        check=True,
    )
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav],
        check=True,
    )
    return wav


def generate_three_speaker_audio(output: Path) -> Path:
    """Three speakers taking turns with some overlap"""
    import soundfile as sf
    import tempfile

    sr = 16000
    wav_path = output / "test_3speakers.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        wav1 = tts_to_wav("Samantha", (
            "Good morning everyone. Let us begin the standup. "
            "Yesterday I worked on the new authentication module. "
            "Today I plan to finish the integration tests."
        ), tmpdir, "s1")

        wav2 = tts_to_wav("Daniel", (
            "Thanks for the update. On my end, I have been debugging "
            "the deployment pipeline. The issue was a misconfigured "
            "environment variable in the staging cluster."
        ), tmpdir, "s2", rate=170)

        wav3 = tts_to_wav("Fred", (
            "I have a quick question about the authentication changes. "
            "Will they affect the existing API keys? "
            "We need to make sure nothing breaks for current users."
        ), tmpdir, "s3", rate=175)

        audio1, _ = sf.read(wav1, dtype="float32")
        audio2, _ = sf.read(wav2, dtype="float32")
        audio3, _ = sf.read(wav3, dtype="float32")

    gap = np.zeros(int(0.3 * sr), dtype=np.float32)
    overlap_len = min(len(audio1) // 6, len(audio3) // 6, sr)

    # timeline: s1 → gap → s2 → gap → s3 → s1+s3 overlap → gap → s2 again
    parts = [
        audio1, gap, audio2, gap, audio3,
    ]

    # overlap section between s1 and s3
    s1_tail = audio1[:overlap_len] * 0.7
    s3_head = audio3[:overlap_len] * 0.7
    overlap = s1_tail + s3_head
    parts.extend([overlap, gap, audio2[:len(audio2) // 2]])

    audio = np.concatenate(parts)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(wav_path), audio, sr)
    print(f"  Generated 3-speaker audio ({len(audio)/sr:.1f}s): {wav_path}")
    return wav_path


def generate_single_speaker_audio(output: Path) -> Path:
    """Single speaker monologue — edge case"""
    import soundfile as sf
    import tempfile

    sr = 16000
    wav_path = output / "test_single_speaker.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        wav = tts_to_wav("Samantha", (
            "Welcome to this lecture on distributed systems. "
            "Today we will cover consensus algorithms, "
            "including Paxos and Raft. "
            "These algorithms are fundamental to building "
            "reliable distributed databases and services. "
            "Let us start with the basics of fault tolerance."
        ), tmpdir, "mono", rate=160)

        audio, _ = sf.read(wav, dtype="float32")

    sf.write(str(wav_path), audio, sr)
    print(f"  Generated single-speaker audio ({len(audio)/sr:.1f}s): {wav_path}")
    return wav_path


def generate_short_clip_audio(output: Path) -> Path:
    """Very short clip (~5s) to test partial window handling"""
    import soundfile as sf
    import tempfile

    sr = 16000
    wav_path = output / "test_short.wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        wav1 = tts_to_wav("Samantha", "Hello, how are you?", tmpdir, "s1")
        wav2 = tts_to_wav("Daniel", "I am fine, thanks.", tmpdir, "s2", rate=170)

        audio1, _ = sf.read(wav1, dtype="float32")
        audio2, _ = sf.read(wav2, dtype="float32")

    gap = np.zeros(int(0.2 * sr), dtype=np.float32)
    audio = np.concatenate([audio1, gap, audio2])
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(wav_path), audio, sr)
    print(f"  Generated short clip ({len(audio)/sr:.1f}s): {wav_path}")
    return wav_path


def run_pipeline_on_audio(pipeline, wav_path: Path, output: Path, prefix: str):
    """Run the diarization pipeline on an audio file and save fixtures"""
    from pyannote.audio.pipelines.utils.hook import ArtifactHook

    file = {"audio": str(wav_path)}
    with ArtifactHook() as hook:
        result = pipeline(file, hook=hook)

    artifacts = file.get("artifact", {})
    print(f"  [{prefix}] Artifact keys: {list(artifacts.keys())}")

    for step_name, artifact in artifacts.items():
        if hasattr(artifact, "data"):
            np.save(output / f"{prefix}_{step_name}_data.npy", artifact.data)
            if hasattr(artifact, "sliding_window"):
                sw = artifact.sliding_window
                with open(output / f"{prefix}_{step_name}_window.json", "w") as f:
                    json.dump(
                        {"start": sw.start, "duration": sw.duration, "step": sw.step},
                        f,
                        indent=2,
                    )
        elif isinstance(artifact, np.ndarray):
            np.save(output / f"{prefix}_{step_name}.npy", artifact)
        elif isinstance(artifact, torch.Tensor):
            np.save(output / f"{prefix}_{step_name}.npy", artifact.numpy(force=True))

    # save RTTM
    annotation = result
    if hasattr(result, "speaker_diarization"):
        annotation = result.speaker_diarization
    elif not hasattr(result, "itertracks"):
        for attr in ("annotation", "diarization", "output"):
            if hasattr(result, attr):
                annotation = getattr(result, attr)
                break

    rttm_path = output / f"{prefix}_expected.rttm"
    with open(rttm_path, "w") as f:
        if hasattr(annotation, "itertracks"):
            for seg, _, speaker in annotation.itertracks(yield_label=True):
                f.write(
                    f"SPEAKER {prefix} 1 {seg.start:.6f} {seg.duration:.6f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )
        else:
            f.write(str(annotation) if annotation else str(result))

    print(f"  [{prefix}] RTTM saved to {rttm_path}")


def export_onnx_models(output: Path, hf_token: str):
    """Download ONNX models from HuggingFace Hub"""
    from huggingface_hub import hf_hub_download

    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)

    # segmentation model
    try:
        seg_onnx = hf_hub_download(
            repo_id="pyannote/segmentation-3.0",
            filename="onnx/model.onnx",
            token=hf_token,
        )
        shutil.copy2(seg_onnx, models_dir / "segmentation-3.0.onnx")
        print(f"  Segmentation ONNX: {models_dir / 'segmentation-3.0.onnx'}")
    except Exception as e:
        print(f"  Could not download segmentation ONNX: {e}")

    # embedding model (WeSpeaker)
    try:
        emb_onnx = hf_hub_download(
            repo_id="pyannote/wespeaker-vox-ceb-cnceleb-resnet34",
            filename="onnx/model.onnx",
            token=hf_token,
        )
        shutil.copy2(emb_onnx, models_dir / "wespeaker-voxceleb-resnet34.onnx")
        print(f"  Embedding ONNX: {models_dir / 'wespeaker-voxceleb-resnet34.onnx'}")
    except Exception as e:
        print(f"  Could not download embedding ONNX: {e}")


def export_plda_params(pipeline, output: Path):
    """Extract and save PLDA parameters from the clustering component"""
    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)

    # walk the pipeline to find PLDA params
    clustering = None
    for attr in ("_clustering", "clustering", "klustering"):
        if hasattr(pipeline, attr):
            clustering = getattr(pipeline, attr)
            break

    if clustering is None:
        print("  Could not find clustering component on pipeline")
        print(f"  Pipeline attributes: {[a for a in dir(pipeline) if not a.startswith('__')]}")
        return

    print(f"  Clustering component: {type(clustering).__name__}")
    print(f"  Clustering attributes: {[a for a in dir(clustering) if not a.startswith('__')]}")

    # look for PLDA parameters in various locations
    plda = None
    for obj in (clustering, pipeline):
        for attr in ("plda", "_plda", "plda_model"):
            if hasattr(obj, attr):
                plda = getattr(obj, attr)
                break
        if plda is not None:
            break

    if plda is not None:
        print(f"  PLDA object: {type(plda).__name__}")
        print(f"  PLDA attributes: {[a for a in dir(plda) if not a.startswith('__')]}")

        for param_name, file_name in [("mu", "plda_mu.npy"), ("phi", "plda_phi.npy")]:
            val = None
            for attr in (param_name, f"_{param_name}", f"plda_{param_name}"):
                if hasattr(plda, attr):
                    val = getattr(plda, attr)
                    break

            if val is not None:
                if isinstance(val, torch.Tensor):
                    val = val.numpy(force=True)
                elif not isinstance(val, np.ndarray):
                    val = np.array(val)
                np.save(models_dir / file_name, val)
                print(f"  Saved {file_name}: shape={val.shape}")
            else:
                print(f"  Could not find PLDA {param_name}")
    else:
        print("  Could not find PLDA model in clustering component")

        # try to extract from pipeline params directly
        params = pipeline.parameters(instantiated=True)
        for k, v in params.items():
            if "plda" in k.lower():
                print(f"  Found PLDA-related param: {k} = {v}")


def generate_pipeline_fixtures(output: Path, hf_token: str):
    """Generate fixtures from running the full pipeline"""
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ArtifactHook

    print("=== Generating pipeline fixtures ===")

    wav_path = get_test_audio(output)

    # try community-1 first, fall back to 3.1
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
        print("  Using community-1 pipeline (VBx + PLDA)")
    except Exception:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        print("  Falling back to 3.1 pipeline (AHC)")

    # force CPU
    pipeline.to(torch.device("cpu"))

    # export ONNX models and PLDA params
    export_onnx_models(output, hf_token)
    export_plda_params(pipeline, output)

    # run pipeline on main 2-speaker test audio
    run_pipeline_on_audio(pipeline, wav_path, output, "pipeline")

    # generate and run on additional audio scenarios
    scenarios = [
        ("3speakers", generate_three_speaker_audio),
        ("single_speaker", generate_single_speaker_audio),
        ("short", generate_short_clip_audio),
    ]

    for name, gen_fn in scenarios:
        try:
            scenario_wav = gen_fn(output)
            run_pipeline_on_audio(pipeline, scenario_wav, output, name)
        except Exception as e:
            print(f"  [{name}] Failed: {e}")

    # save pipeline params
    params = pipeline.parameters(instantiated=True)
    with open(output / "pipeline_params.json", "w") as f:
        json.dump({k: str(v) for k, v in params.items()}, f, indent=2)

    print(f"  Pipeline fixtures saved to {output}")


def main():
    output = Path(__file__).parent
    output.mkdir(parents=True, exist_ok=True)

    generate_model_free_fixtures(output)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        generate_pipeline_fixtures(output, hf_token)
    else:
        print("\n  No HF_TOKEN set. Skipping pipeline fixtures.")
        print("  Set HF_TOKEN env var for full pipeline fixture generation.")

    print(f"\n=== Done. Fixtures in {output} ===")


if __name__ == "__main__":
    main()
