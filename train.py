from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from adabelief_pytorch import AdaBelief
from rich import print
from torch import Tensor
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

import models
import wandbreport as report
from nibabelio import GeneratorDataset, repeat, load_generated, slices, split_iter_trainables

env = SimpleNamespace()
env.backend = "wandb"
env.dataset_path = Path("/gpfswork/rech/otc/uiu95bi/dataset")
env.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env.models_path = Path("/gpfswork/rech/otc/uiu95bi/saved_models")
env.sources_path = Path("/gpfswork/rech/otc/uiu95bi/sources")

# wandb login 6e44b780601adbcb29c5322b462b5da80078222e

opt = SimpleNamespace(
    batch_size=4,
    buffer_size=120,
    channels=[4, 16, 32, 64, 128, 256],
    grad_accumulation_steps=5,
    n_samples=4,
    split=False,
    staging_size=10,
    warmup_size=400,
)

print(f"Run with options: {vars(opt)}")
model = models.Unet3d(**vars(opt))
model.set_momentum(0.5)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs.")
    model = nn.DataParallel(model)
model = model.to(env.device)

print("Using model:")
print(repr(model))

# train_cases, valid_cases = split_iter_trainables(env.sources_path)

def train_cases():
    while True:
        for item in (env.dataset_path / "train").glob("*.pt"):
            yield torch.load(item)

def valid_cases():
    for item in (env.dataset_path / "valid").glob("*.pt"):
        yield torch.load(item)

train_dataset = GeneratorDataset(
    train_cases(),
    buffer_size=opt.buffer_size,
    staging_size=opt.staging_size,
)

valid_dataset = GeneratorDataset(
    valid_cases(),
)

tdl = DataLoader(
    train_dataset,
    pin_memory=True,
    batch_size=opt.batch_size,
)

vdl = DataLoader(
    valid_dataset,
    pin_memory=True,
    batch_size=opt.batch_size,
    shuffle=True,
)

optimizer = AdaBelief(
    model.parameters(),
    lr=1e-3,
    eps=1e-8,
    betas=(0.9, 0.999),
    weight_decouple=False,
    rectify=False,
    print_change_log=False,
)

raw_loss = nn.CrossEntropyLoss(torch.tensor([1, 5, 20]), reduction="none").to(device=env.device)


def mean_loss(pred, segm):
    return torch.mean(raw_loss(pred, segm))


def loss_and_scores(pred, segm, keys):
    raw = raw_loss(pred, segm)
    loss = torch.mean(raw)
    batch_losses = torch.mean(raw, dim=[1, 2, 3])
    scores = {k.item(): batch_losses[i].item() for i, k in enumerate(keys)}
    return loss, scores


def split2(t: Tensor, dim: int):
    midpoint = t.size(dim) // 2
    t: list[Tensor] = t.unbind()
    t: list[list[Tensor]] = [[_t.narrow(dim - 1, 0, midpoint), _t.narrow(dim - 1, midpoint, midpoint)] for _t in t]
    return torch.stack([
        _t
        for tl in t
        for _t in tl
    ])


def unsplit2(t: Tensor, dim: int):
    t: list[Tensor] = t.unbind()
    t = [torch.cat([t[i], t[i + 1]], dim=dim - 1) for i in range(0, len(t), 2)]
    return torch.stack(t)


def forward(model, x):
    # print(f"DEBUG: forward(model, x{x.shape}) call")
    if opt.split:
        x = split2(x, dim=2)
        x = split2(x, dim=3)
    # print(f"DEBUG: after split model(x{x.shape})")
    x = model(x)
    if opt.split:
        x = unsplit2(x, dim=3)
        x = unsplit2(x, dim=2)
    return x


## Validation round

@torch.no_grad()
def validation_round(epoch: int, epochs: int):
    round_loss = 0
    samples = []
    model.eval()

    # print(f"Validation round ({epoch+1}/{epochs}).")
    for batched_data in vdl:
        scan = batched_data["scan"].to(device=env.device)
        segm = batched_data["segm"].to(device=env.device, dtype=torch.int64)
        batch_size = segm.size(0)

        pred = forward(model, scan)
        loss = mean_loss(
            pred,
            functional.one_hot(segm, 3).permute(0, 4, 1, 2, 3).to(dtype=torch.float32),
        )
        round_loss += loss.item() * batch_size

        if len(samples) < opt.n_samples:
            samples.append(report.sample(
                scan.detach().cpu().numpy(),
                torch.argmax(pred.detach(), dim=1).cpu().numpy(),
                segm.detach().cpu().numpy()
            ))

    scan_loss = round_loss / len(valid_dataset)
    return scan_loss, samples


"""## Training round"""


def training_round(epoch: int, epochs: int):
    round_loss = 0
    samples = []
    round_scores = {}
    model.train()

    # print(f"Training round ({epoch+1}/{epochs}).")
    for batch_index, batched_data in enumerate(tdl):
        scan = batched_data["scan"].to(device=env.device)
        segm = batched_data["segm"].to(device=env.device, dtype=torch.int64)
        keys = batched_data["keys"]
        batch_size = segm.size(0)

        pred = forward(model, scan)
        loss, scores = loss_and_scores(
            pred,
            functional.one_hot(segm, 3).permute(0, 4, 1, 2, 3).to(dtype=torch.float32),
            keys=keys
        )
        loss = loss / opt.grad_accumulation_steps  # because of gradient accumulation
        loss.backward()
        round_loss += loss.item() * batch_size * opt.grad_accumulation_steps
        round_scores.update(scores)
        if len(samples) < opt.n_samples:
            samples.append(report.sample(
                scan.detach().cpu().numpy(),
                torch.argmax(pred.detach(), dim=1).cpu().numpy(),
                segm.detach().cpu().numpy()
            ))

        if ((batch_index + 1) % opt.grad_accumulation_steps == 0) or (batch_index + 1 == len(tdl)):
            optimizer.step()
            optimizer.zero_grad()
    optimizer.zero_grad(set_to_none=True)

    scan_loss = round_loss / len(train_dataset)
    return scan_loss, round_scores, samples


## Train cycle

def train(epochs: int = 21):
    models_path = env.models_path
    for epoch in range(epochs):
        if epoch % 20 == 0:
            model.eval()
            # scan_loss = validation_round(epoch, epochs)
            scan_loss, samples = validation_round(epoch, epochs)
            print(
                f"Validation epoch {epoch + 1}/{epochs}. "
                f"Loss per scan: {scan_loss:.2e}"
                f"".ljust(50, ' ')
            )
            # report.append({"valid_epoch_loss": scan_loss})
            report.append({"valid_epoch_loss": scan_loss, "samples": samples})
            torch.save(model.state_dict(), models_path / "last_checkpoint.pth")
            torch.save(model.state_dict(), models_path / f"checkpoint{epoch:03}.pth")
        else:
            model.train()
            # scan_loss, scores = training_round(epoch, epochs)
            scan_loss, scores, samples = training_round(epoch, epochs)
            train_dataset.drop(scores)
            print(
                f"Training epoch {epoch + 1}/{epochs}. "
                f"Loss per scan: {scan_loss:.2e}"
                f"".ljust(50, ' ')
            )
            # report.append({"train_epoch_loss": scan_loss})
            report.append({"train_epoch_loss": scan_loss, "samples": samples})
        # gc.collect()


## Start

try:
    model.load_state_dict(torch.load(env.models_path / "last_checkpoint.pth", map_location=env.device))
    print("Loaded model from 'last_checkpoint.pth'.")
except Exception as err:
    print("Using random weights. Loading attempt produced the following exception:")
    print(err)



def evaluator(item):
    scan = item["scan"].to(device=env.device).unsqueeze(0)
    segm = item["segm"].to(device=env.device, dtype=torch.int64).unsqueeze(0)
    pred = model(scan)
    loss = mean_loss(
        pred,
        functional.one_hot(segm, 3).permute(0, 4, 1, 2, 3).to(dtype=torch.float32),
    )
    return loss.item()


run = report.init(config=vars(opt))
try:
    if opt.warmup_size:
        train_dataset.warmup(opt.warmup_size, evaluator)
    train(201)
except Exception as err:
    run.finish()
    raise err

# """# Evaluation
#
# ## Stage zero
# """
#
# from torch import Tensor
# from torch.nn.functional import l1_loss
#
# def score(input: np.array, target: np.array, index: int):
#     input = (input == index).astype(np.float32)
#     target = (target == index).astype(np.float32)
#     return float(1 - np.abs(input - target).sum() / (np.maximum(input, target).sum()+1))
#
# def apply(model, case_path: Path):
#     load_apply_case(case_path)
#     pred = np.empty(nio.loaded_case.scan.shape[1:], dtype=np.int64)  # same shape as scan, without channels
#     for z in pred.shape[2]:  # shape is [X, Y, Z]
#         scan=torch.tensor(nio.loaded_case.scan[..., z]).to(device=env.device)
#         pred[..., z] = torch.argmax(model(scan).detach(), dim=1).cpu().numpy()
#     return score(pred, nio.loaded_case.segm, 1), score(pred, nio.loaded_case.segm, 2)
#
# print("Evaluating model zero")
#
# model = models.UNet([4, 64]).to(env.device)
# model.load_state_dict(torch.load(env.models_path / "stage_zero.pth", map_location=env.device))
# model.eval()
#
# cases = len(train_cases())
# liver_score = 0
# tumor_score = 0
# for case_path in train_cases():
#     ls, ts = apply(model, case_path)
#     liver_score += ls
#     tumor_score += ts
# print(f"Average scores on traning dataset are: {liver_score / cases} -- {tumor_score / cases}")
#
# cases = len(valid_cases())
# liver_score = 0
# tumor_score = 0
# for case_path in valid_cases():
#     ls, ts = apply(model, case_path)
#     liver_score += ls
#     tumor_score += ts
# print(f"Average scores on validation dataset are: {liver_score / cases} -- {tumor_score / cases}")
#
# print("Evaluating model one")
#
# model = models.UNet([4, 64, 128]).to(env.device)
# model.load_state_dict(torch.load(env.models_path / "stage_one.pth", map_location=env.device))
# model.eval()
#
# cases = len(train_cases())
# liver_score = 0
# tumor_score = 0
# for case_path in train_cases():
#     ls, ts = apply(model, case_path)
#     liver_score += ls
#     tumor_score += ts
# print(f"Average scores on traning dataset are: {liver_score / cases} -- {tumor_score / cases}")
#
# cases = len(valid_cases())
# liver_score = 0
# tumor_score = 0
# for case_path in valid_cases():
#     ls, ts = apply(model, case_path)
#     liver_score += ls
#     tumor_score += ts
# print(f"Average scores on validation dataset are: {liver_score / cases} -- {tumor_score / cases}")
