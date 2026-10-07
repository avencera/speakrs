"""Fixed collection inventories and fault applicability before production approval."""

SEGDENSE = {
    "conv1": "sincnet.conv1",
    "conv2": "sincnet.conv2",
    "linear0": "linear0",
    "linear1": "linear1",
    "classifier": "linear2",
    "embedding": "resnet.seg_1",
}
WIDECONV = (
    "resnet.conv1",
    *(
        f"resnet.layer{stage}.{block}.conv{conv}"
        for stage, count in ((3, 6), (4, 3))
        for block in range(count)
        for conv in (1, 2)
    ),
    *(f"resnet.layer{stage}.0.shortcut.0" for stage in (2, 3, 4)),
)
LAYERS = {
    "wideconv": WIDECONV,
    **{f"segdense-{name}": (layer,) for name, layer in SEGDENSE.items()},
}
AREAS = {target: "segdense" for target in LAYERS if target != "wideconv"}
AREAS["wideconv"] = "wideconv"
EMBEDDING_TARGETS = frozenset(("resnet", "wideconv", "segdense-embedding"))
DRAW_ELEMENTS = {
    "sincnet.conv1": 60 * 5321,
    "sincnet.conv2": 60 * 1769,
    "linear0": 589 * 128,
    "linear1": 589 * 128,
    "linear2": 589 * 7,
    "resnet.seg_1": 3 * 256,
    "resnet.conv1": 32 * 80 * 998,
    "resnet.layer2.0.shortcut.0": 64 * 40 * 499,
    "resnet.layer3.0.shortcut.0": 128 * 20 * 250,
    "resnet.layer4.0.shortcut.0": 256 * 10 * 125,
    **{
        layer: 128 * 20 * 250 if layer.startswith("resnet.layer3.") else 256 * 10 * 125
        for layer in WIDECONV
        if ".conv" in layer and layer != "resnet.conv1"
    },
}
STANDARD_MUTANTS = (
    "Precision",
    "Shape",
    "Fallback",
    "FirstUseFallback",
    "FirstUseFallbackEager",
    "FirstUseFallbackCaptured",
    "FirstUseFallbackReplay",
    "Tail",
    "Atomic",
    "StageSlow",
    "StageAccuracy",
    "Slow",
    "PhaseCheat",
    "Unscoped",
    "Unlisted",
    "Lookup",
    "UninitShared",
)
MUTANT_APPLICABILITY = {
    target: {
        **{fault: "applicable" for fault in STANDARD_MUTANTS},
        "StageTail": "requires_accepted_production_plan",
        "StageTailControl": "requires_accepted_production_plan",
        "WrongLayout": "out_of_scope_no_padded_layout",
    }
    for target in LAYERS
}
