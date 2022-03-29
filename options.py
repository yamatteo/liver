from pathlib import Path
from envs import envs

defaults = dict(
    **envs,
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
