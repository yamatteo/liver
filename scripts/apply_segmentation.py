import argparse
import os
from pathlib import Path

import dotenv
import nibabel
import numpy as np
import torch
import wandb
from adabelief_pytorch import AdaBelief

# from dataset import BufferDataset2 as BufferDataset
from models.multi_unet import UNet
from train import train_cycle
from utils import generators
from utils.path_explorer import discover, get_criterion

dotenv.load_dotenv()
data_path = Path(os.getenv("OUTPUTS"))
models_path = Path(os.getenv("SAVED_MODELS"))

opts = argparse.Namespace(
    batch_size=20,
    buffer_size=400,
    epochs=400,
    learning_rate=2e-4,
    channels=[4, 32, 64, 96, 128],
    resume=True,
    slice_shape=(128, 128, 8),
    train_to_valid_ratio=10,
)

model = UNet(
    channels=opts.channels
)

model.load_state_dict(torch.load(models_path / "current.pth", map_location=torch.device('cpu')))

case = Path(os.getenv("OUTPUTS")) / "Hum331"
scan = ScanBatch(torch.stack([
        torch.tensor(np.array(nibabel.load(
            case / f"registered_phase_{phase}.nii.gz"
        ).dataobj, dtype=np.int16))
        for phase in ["b", "a", "v", "t"]
    ]).float().unsqueeze(0))

good_z = [not bool(torch.any(torch.all(torch.all(scan[:, :, :, z] == 0, dim=1), dim=1))) for z in
              range(scan.size(3))]
a = 0
for z in range(scan.size(3)):
    if any(good_z[:z]):
        break
    else:
        a = z
b = scan.size(3)
for z in reversed(range(scan.size(3))):
    if any(good_z[z:]):
        break
    else:
        b = z

with torch.no_grad():
    model.eval()
    our_best_guess = model(scan[:, :, :, :, a:b])
print(our_best_guess)
#
# net = models.get_model(**dict(opts, model="segm.2")).to(device=device, dtype=torch.float32)
# net.eval()
#
# net882 = models.get_model(**dict(opts, model="segm882.7")).to(device=device, dtype=torch.float32)
# net882.eval()
#
# console.print("[bold orange3]Segmenting:[/bold orange3]")
# for case_path in discover(os.getenv("OUTPUTS"), get_criterion(registered=True)):
#     source_path = os.getenv("OUTPUTS") / case_path
#     target_path = os.getenv("OUTPUTS") / case_path
#     target_path_is_complete = (target_path / f"prediction.nii.gz").exists()
#     if os.getenv("OVERWRITE") or not target_path_is_complete:
#         target_path.mkdir(parents=True, exist_ok=True)
#         our_best_guess = predict_case(case=source_path, net882=net882, net=net, device=device)
#
#         affine = nibabel.load(target_path / f"registered_phase_v.nii.gz").affine
#         nibabel.save(
#             nibabel.Nifti1Image(
#                 our_best_guess.cpu().numpy(),
#                 affine=affine
#             ),
#             target_path / "prediction.nii.gz",
#         )
#
#     else:
#         console.print(f"[bold black]{case_path.name}.[/bold black] is already complete, skipping.")
