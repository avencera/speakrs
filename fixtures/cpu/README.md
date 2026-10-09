# CPU model references

`segmentation-test-short.npy` contains the complete `[589, 7]` output from
the original ONNX segmentation model for `fixtures/test_short.wav`.
`segmentation-test-short.json` records the pinned model, input and reference
hashes, CPU runtime and input preparation.

The reference was captured with ONNX Runtime, not the native Rust model.
Normal CPU tests compare all logits within `2e-3` and require exact decoded
activity. The existing CI jobs already download the pinned native weights;
they need no extra runtime or external acceptance captures for this test.

Do not regenerate this reference from native Rust output. A deliberate model
or input update must include a new independent reference and its provenance.
