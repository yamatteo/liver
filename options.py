from pathlib import Path

defaults = dict(
    # sources=Path("/sources"),
    sources=Path("../inputs"),
    # outputs=Path("/outputs"),
    outputs=Path("../outputs"),
    # nifti_bin=Path("/usr/local/bin/"),
    nifti_bin=Path("~/niftireg/nifti_install/bin/"),
    # saved_models=Path("/models"),
    saved_models=Path("./saved_models"),

    epochs=1000,
    batch_size=4,
    validation_percent=0.1,
    learning_rate=1e-4,
    adabelief_eps=1e-8,
    adabelief_b1=0.9,
    adabelief_b2=0.999,
    num_workers=8,
    wafer_size=5,
)
