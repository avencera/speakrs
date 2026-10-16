"""Integration tests for the speakrs Python package."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import speakrs
from conftest import SHORT_WAV, TEST_WAV, read_wav, rust_reference


def test_cpu_rttm_matches_rust_pipeline(cpu_models, cpu_pipeline):
    expected, expected_exclusive = rust_reference("cpu", cpu_models.model_dir, TEST_WAV, "test")
    assert expected.count("SPEAKER") > 1

    from_file = cpu_pipeline.diarize_file(TEST_WAV)
    assert from_file.file_id == "test"
    assert from_file.rttm() == expected
    assert from_file.rttm(exclusive=True) == expected_exclusive
    assert from_file.mode == speakrs.ExecutionMode.CPU
    assert from_file.duration == pytest.approx(26.0154375)
    assert from_file.speakers == sorted({segment.speaker for segment in from_file.segments})
    assert from_file.timing is not None and from_file.timing.pipeline_ms > 0
    # the fixture is already 16 kHz, so nothing was resampled
    assert from_file.timing.audio_resample_ms == 0.0

    samples = read_wav(TEST_WAV)
    assert cpu_pipeline.diarize_samples(samples, file_id="test").rttm() == expected
    # a strided view cannot be borrowed in place and is copied instead
    doubled = np.repeat(samples, 2)[::2]
    assert cpu_pipeline.diarize_samples(doubled, file_id="test").rttm() == expected
    assert cpu_pipeline.diarize_samples(samples.tolist(), file_id="test").rttm() == expected


def test_samples_input_validation(cpu_pipeline):
    samples = read_wav(SHORT_WAV)

    with pytest.raises(TypeError, match="float32"):
        cpu_pipeline.diarize_samples(samples.astype(np.float64))
    # big-endian float32 is byte-swapped on copy, not read as native floats
    expected = cpu_pipeline.diarize_samples(samples).rttm()
    assert "SPEAKER" in expected
    assert cpu_pipeline.diarize_samples(samples.astype(">f4")).rttm() == expected
    with pytest.raises(ValueError, match="1-D"):
        cpu_pipeline.diarize_samples(samples.reshape(1, -1))
    with pytest.raises(TypeError):
        cpu_pipeline.diarize_samples("not audio")


def test_batch_matches_single_files_and_reports_progress(cpu_pipeline):
    singles = [cpu_pipeline.diarize_file(path) for path in (TEST_WAV, SHORT_WAV)]
    events = []

    batch = cpu_pipeline.diarize_files([TEST_WAV, SHORT_WAV], progress=events.append)

    assert [result.file_id for result in batch] == ["test", "test_short"]
    assert [result.rttm() for result in batch] == [result.rttm() for result in singles]
    assert all(result.timing is None for result in batch)
    assert [(event.kind, event.name, event.index) for event in events] == [
        ("batch_file", "test", 0),
        ("batch_file", "test_short", 1),
        ("completed", None, None),
    ]

    from_async = asyncio.run(cpu_pipeline.diarize_files_async([SHORT_WAV], file_ids=["x"]))
    assert from_async[0].rttm() == singles[1].rttm(file_id="x")


def test_queue_returns_results_then_stops(cpu_models):
    reference = speakrs.Pipeline.from_prepared(cpu_models).diarize_file(SHORT_WAV)

    with speakrs.Pipeline.from_prepared(cpu_models).into_queue(capacity=4) as queue:
        file_job = queue.push_file("from-file", SHORT_WAV)
        samples_job = queue.push_samples("from-samples", read_wav(SHORT_WAV))

    results = sorted(queue, key=lambda result: result.job_id)
    assert [result.job_id for result in results] == [file_job, samples_job]
    for result in results:
        assert result.ok and result.error is None
        assert result.unwrap().rttm(file_id="x") == reference.rttm(file_id="x")
        assert result.unwrap().duration == pytest.approx(reference.duration)

    with pytest.raises(speakrs.SpeakrsError) as closed:
        queue.push_file("late", SHORT_WAV)
    assert closed.value.code == "queue_closed"


def test_cancelled_async_call_stops_before_returning(cpu_pipeline):
    samples = read_wav(TEST_WAV)
    events = []
    caller_token = speakrs.CancelToken()

    async def time_out():
        call = cpu_pipeline.diarize_samples_async(
            samples, cancel_token=caller_token, progress=lambda event: events.append(event.kind)
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(call, timeout=0.05)
        # the worker reads `samples`, so the timeout may only arrive once it stopped
        return list(events)

    seen_at_timeout = asyncio.run(time_out())
    # a later call waits for the pipeline, so any worker still running finishes first
    cpu_pipeline.diarize_file(SHORT_WAV)
    assert events == seen_at_timeout
    assert "completed" not in events
    assert not caller_token.cancelled

def test_cancelled_recv_async_leaves_results_for_later_receivers(cpu_models):
    async def scenario(queue):
        # nothing is pushed yet, so this wait times out and is cancelled
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(queue.recv_async(), timeout=0.05)
        job = queue.push_file("after-timeout", SHORT_WAV)
        queue.close()
        result = await asyncio.wait_for(queue.recv_async(), timeout=120)
        assert result.job_id == job and result.ok
        assert "SPEAKER" in result.unwrap().rttm()
        with pytest.raises(speakrs.SpeakrsError) as closed:
            await asyncio.wait_for(queue.recv_async(), timeout=10)
        assert closed.value.code == "queue_closed"

    with speakrs.Pipeline.from_prepared(cpu_models).into_queue(capacity=4) as queue:
        asyncio.run(scenario(queue))

def test_try_recv_never_waits_for_a_blocked_receiver(cpu_models):
    with speakrs.Pipeline.from_prepared(cpu_models).into_queue(capacity=4) as queue:
        errors = []

        def wait_on_idle_queue():
            try:
                queue.recv()
            except speakrs.SpeakrsError as error:
                errors.append(error.code)

        receiver = threading.Thread(target=wait_on_idle_queue)
        receiver.start()
        time.sleep(0.2)
        # the blocked recv holds the receiver in 100 ms slices; pausing between polls,
        # as recv_async does, lands inside a slice, where waiting would stall the loop
        slowest = 0.0
        for _ in range(10):
            time.sleep(0.01)
            started = time.monotonic()
            assert queue.try_recv() is None
            slowest = max(slowest, time.monotonic() - started)
        assert slowest < 0.05

        queue.close()
        receiver.join(timeout=10)
    assert errors == ["queue_closed"]

def test_errors_carry_stable_categories(cpu_models, cpu_pipeline, tmp_path):
    with pytest.raises(speakrs.SpeakrsError) as missing:
        speakrs.prepare("cpu", model_dir=tmp_path)
    assert missing.value.category == "model_manifest"

    with pytest.raises(speakrs.SpeakrsError) as mode:
        speakrs.Pipeline.from_prepared(cpu_models, "migraphx")
    assert mode.value.category == "configuration"

    with pytest.raises(speakrs.SpeakrsError) as decode:
        cpu_pipeline.diarize_file(tmp_path / "missing.wav")
    assert decode.value.category == "audio_decode"

    with pytest.raises(speakrs.SpeakrsError) as config:
        speakrs.VbxConfig(max_iters=0)
    assert config.value.category == "configuration"
    for epsilon in (-0.1, float("nan"), float("inf")):
        with pytest.raises(speakrs.SpeakrsError, match="smoothing_epsilon") as smoothing:
            speakrs.PipelineConfig(smoothing_epsilon=epsilon)
        assert smoothing.value.category == "configuration"

    token = speakrs.CancelToken()
    token.cancel()
    with pytest.raises(speakrs.SpeakrsError) as cancelled:
        cpu_pipeline.diarize_files([SHORT_WAV], cancel_token=token)
    assert cancelled.value.category == "cancelled"

    manifest = speakrs.generate_manifest(cpu_models.model_dir, "cpu")
    restored = speakrs.ModelManifest.from_json(manifest.to_json())
    assert speakrs.prepare("cpu", model_dir=cpu_models.model_dir, manifest=restored).revision == manifest.revision


def test_misuse_fails_fast_and_keeps_the_pipeline_usable(cpu_models):
    pipeline = speakrs.Pipeline.from_prepared(cpu_models)
    expected = pipeline.diarize_file(SHORT_WAV).rttm()
    assert "SPEAKER" in expected

    with pytest.raises(speakrs.SpeakrsError) as capacity:
        pipeline.into_queue(capacity=0)
    assert capacity.value.category == "configuration"

    # a callback calling back into the pipeline would wait on the lock its caller holds
    with pytest.raises(speakrs.SpeakrsError, match="progress callback") as reentrant:
        pipeline.diarize_file(SHORT_WAV, progress=lambda _: pipeline.pipeline_config)
    assert reentrant.value.category == "configuration"
    with pytest.raises(speakrs.SpeakrsError, match="progress callback"):
        pipeline.diarize_file(SHORT_WAV, progress=lambda _: pipeline.into_queue())

    events = []

    def fail_on_first_file(event):
        events.append(event.kind)
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        pipeline.diarize_files([TEST_WAV, SHORT_WAV, TEST_WAV], progress=fail_on_first_file)
    # the raise cancels the batch, so later files never start
    assert events == ["batch_file"]

    # a cancel raised by the event that opens a stage stops that stage, too
    events.clear()
    with pytest.raises(RuntimeError, match="stop"):
        pipeline.diarize_samples(read_wav(SHORT_WAV), progress=fail_on_first_file)
    assert events == ["running_pipeline"]

    # the raise cancels only its own call, never a token the caller passed in
    token = speakrs.CancelToken()
    with pytest.raises(RuntimeError, match="stop"):
        pipeline.diarize_file(SHORT_WAV, cancel_token=token, progress=fail_on_first_file)
    assert not token.cancelled
    assert pipeline.diarize_file(SHORT_WAV, cancel_token=token).rttm() == expected

    assert pipeline.diarize_file(SHORT_WAV).rttm() == expected


def test_pipeline_config_changes_results(cpu_pipeline):
    default = cpu_pipeline.diarize_file(TEST_WAV)
    merged = cpu_pipeline.diarize_file(
        TEST_WAV, pipeline_config=speakrs.PipelineConfig(merge_gap=5.0)
    )

    assert speakrs.PipelineConfig.for_mode("cpu") == cpu_pipeline.pipeline_config
    assert len(merged.segments) < len(default.segments)


def test_cli_prints_rttm_and_clean_errors(cpu_models, cpu_pipeline, tmp_path):
    speakrs_cli = shutil.which("speakrs", path=str(sys.prefix + "/bin"))
    assert speakrs_cli is not None
    run = subprocess.run(
        [speakrs_cli, "diarize", str(SHORT_WAV), "--model-dir", str(cpu_models.model_dir)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout == cpu_pipeline.diarize_file(SHORT_WAV).rttm()

    failed = subprocess.run(
        [speakrs_cli, "diarize", str(SHORT_WAV), "--model-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 1
    assert failed.stdout == ""
    assert failed.stderr.startswith("error: model file")
    assert "Traceback" not in failed.stderr


@pytest.mark.coreml
@pytest.mark.skipif("CoreML" not in speakrs.available_modes(), reason="build without CoreML")
@pytest.mark.parametrize("mode", ["coreml", "coreml-fast"])
def test_coreml_rttm_matches_rust_pipeline(mode):
    prepared = speakrs.prepare(mode)
    expected, expected_exclusive = rust_reference(mode, prepared.model_dir, TEST_WAV, "test")

    result = speakrs.Pipeline.from_prepared(prepared).diarize_file(TEST_WAV)

    assert result.mode == {"coreml": "CoreML", "coreml-fast": "CoreMLFast"}[mode]
    assert expected.count("SPEAKER") > 1
    assert result.rttm() == expected
    assert result.rttm(exclusive=True) == expected_exclusive


def _has_nvidia_gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    return smi is not None and subprocess.run([smi, "-L"], capture_output=True).returncode == 0


@pytest.mark.cuda
@pytest.mark.skipif("CUDA" not in speakrs.available_modes(), reason="build without CUDA")
@pytest.mark.skipif(not _has_nvidia_gpu(), reason="no NVIDIA GPU")
@pytest.mark.parametrize("mode", ["cuda", "cuda-fast"])
def test_cuda_rttm_matches_rust_pipeline(mode, cpu_pipeline):
    prepared = speakrs.prepare(mode)
    expected, _ = rust_reference(mode, prepared.model_dir, TEST_WAV, "test")

    pipeline = speakrs.Pipeline.from_prepared(prepared)
    result = pipeline.diarize_file(TEST_WAV)
    batch = pipeline.diarize_files([TEST_WAV, SHORT_WAV])

    assert result.mode.lower() == mode.replace("-", "")
    assert result.rttm() == expected
    assert batch[0].rttm() == expected
    # the GPU finds the same speakers as the CPU on this clean two-speaker file
    assert result.speakers == cpu_pipeline.diarize_file(TEST_WAV).speakers


@pytest.mark.skipif("CUDA" not in speakrs.available_modes(), reason="build without CUDA")
@pytest.mark.skipif(_has_nvidia_gpu(), reason="checks the behavior without a GPU")
def test_cuda_build_without_gpu_runs_cpu_and_fails_cuda_cleanly(cpu_models, cpu_pipeline):
    # the CPU and CUDA modes load the same files, so the fixtures serve both
    assert speakrs.required_model_files("cuda") == speakrs.required_model_files("cpu")
    assert len(cpu_pipeline.diarize_file(SHORT_WAV).segments) > 0

    with pytest.raises(speakrs.SpeakrsError) as error:
        speakrs.Pipeline.from_prepared(cpu_models, "cuda")
    assert error.value.category == "model_load"
