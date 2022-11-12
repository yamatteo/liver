from pathlib import Path

import numpy as np
import torch
from adabelief_pytorch import AdaBelief
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
import models
import report
from rich import print
from slicing import slices

cuda0 = torch.device("cuda:0")
# cuda1 = torch.device("cuda:1")

debug = Path(".env").exists()
res = 64 if debug else 512
shrink_shape = (16, 16, 4)

def scan_shrink(scan: np.ndarray):
    shape = scan.shape
    scan = torch.tensor(scan, dtype=torch.float32).view([1, *shape])
    scan = functional.avg_pool3d(scan, kernel_size=shrink_shape)
    return scan.numpy()

def segm_shrink(segm: np.ndarray):
    shape = segm.shape
    segm = np.clip(segm, 0, 1)
    segm = torch.tensor(segm, dtype=torch.int64).view([1, *shape])
    segm = functional.one_hot(segm, num_classes=2).permute([0, 4, 1, 2, 3]).to(dtype=torch.float32)
    segm = functional.avg_pool3d(segm, kernel_size=shrink_shape)
    # segm = torch.argmax(segm, dim=1)
    return segm.numpy()

def train_epoch(model, losses, ds, epoch, optimizer, args):
    """Assuming model is single stream."""
    round_scores = dict()
    round_recall = 0
    assert args.batch_size == 1
    dl = DataLoader(ds, batch_size=args.batch_size)
    for data in dl:
        key = int(data["keys"][0])
        model(data)
        losses(data)
        loss = (data["cross"] - (data["recall"] - 1))/args.grad_accumulation_steps
        loss.backward()
        round_scores.update({key: loss.item()})
        round_recall += data["recall"].item()
        if (key + 1) % args.grad_accumulation_steps == 0 or key + 1 == len(ds):
            optimizer.step()
            optimizer.zero_grad()
    ds.drop(round_scores)
    mean_loss = sum(round_scores.values()) * args.grad_accumulation_steps / len(ds)
    print(
        f"Training epoch {epoch + 1}/{args.epochs}. "
        f"Loss per scan: {mean_loss:.2e}. "
        f"Mean recall: {round_recall/len(ds):.2f}"
    )
    report.append({f"loss": mean_loss})

@torch.no_grad()
def validation_round(model, ds, *, args):
    # first_device, last_device = streams[0].device, streams[-1].device
    scores = []
    samples = []
    dl = DataLoader(ds)
    for data in dl:
        name = data.get("name", ["no name"])[0]
        scan = data["scan"]
        segm = data["segm"]
        pred = []
        for (x, t) in slices(scan, segm, shape=args.slice_shape, pad_up_to=1):
            items = model({"scan": x})
            pred.append(items["pred"])
        pred = torch.argmax(torch.cat(pred, dim=-1)[..., :scan.shape[-1]], dim=1)
        segm = torch.as_tensor(segm, device=pred.device)
        intersection = torch.sum(pred * segm).item() + 0.1
        union = torch.sum(torch.clamp(pred + segm, 0, 1)).item() + 0.1
        scores.append({"name": name, "value": intersection/union})
        for _ in range(4):
            samples.append(report.sample(
                scan.cpu().numpy(),
                # scans[1].detach().cpu().numpy(),
                pred.cpu().numpy(),
                # torch.argmax(preds[1], dim=1).detach().cpu().numpy(),
                segm.cpu().numpy()
            ))
    print("Validation scores are:")
    for score in scores:
        name = score["name"]
        value = score["value"]
        print(f"{name:>12}:{100 * value:6.1f}% iou")
    mean_score = sum(item["value"] for item in scores) / len(scores)
    report.append({f"validation_score": mean_score, "samples":samples}, commit=False)

def train(model, losses, tds, vds, args):
    optimizer = AdaBelief(
            model.parameters(),
            lr=args.lr,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=False,
            rectify=False,
            print_change_log=False,
        )
    for epoch in range(args.epochs):
        if (epoch+1) % 20 == 0:
            validation_round(model, vds, args=args)
            model.save()
        else:
            train_epoch(model, losses, ds=tds, epoch=epoch, optimizer=optimizer, args=args)
# def train(model: models.Pipeline, train_dataset, valid_dataset, args):
#     epoch = 0
#     next_key = 0
#     # last_device = model.streams[-1].device
#     # loss_func = nn.CrossEntropyLoss(torch.tensor(args.class_weights[:args.finals])).to(device=last_device)
#     round_loss = 0
#     round_scores = {}
#     steps = [{} for _ in model.steps]
#
#     # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum)
#     optimizer = AdaBelief(
#         model.parameters(),
#         lr=args.lr,
#         eps=1e-8,
#         betas=(0.9, 0.999),
#         weight_decouple=False,
#         rectify=False,
#         print_change_log=False,
#     )
#
#     # validation_round(model, valid_dataset, args)
#     while epoch < args.epochs:
#
#         # Time shift
#         steps = [train_dataset[next_key], *steps[:-1]]
#         steps[0]["scan"] = torch.tensor(steps[0]["scan"]).unsqueeze(0).float()
#         steps[0]["segm"] = torch.tensor(steps[0]["segm"]).unsqueeze(0).clamp(0, 1).long()
#         model.pipeline_forward(steps)
#
#         active_key = steps[-1].get("keys", None)
#         if active_key is not None:
#             loss = torch.mean(steps[-1]["loss128"] + steps[-1]["loss256"]) / args.grad_accumulation_steps
#             loss.backward()
#             round_loss += loss.item()
#             round_scores.update({steps[-1]["keys"]: loss.item()})
#             if (active_key + 1) % args.grad_accumulation_steps == 0\
#                     or active_key + 1 == len(train_dataset):
#                 optimizer.step()
#                 optimizer.zero_grad()
#
#         # keys = [k, *keys[:-1]]
#         # inputs[0] = [data["scan"].unsqueeze(0), *inputs[0][:-1]]
#         # for j in range(1, len(streams) + 1):
#         #     inputs[j] = [None, *inputs[j][:-1]]
#         # if args.finals == 2:
#         #     targets = [data["segm"].unsqueeze(0).clamp(0, 1), *targets[:-1]]
#         # else:
#         #     targets = [data["segm"].unsqueeze(0), *targets[:-1]]
#
#         next_key += 1
#         if next_key >= len(train_dataset):
#             next_key = 0
#             if round_scores:
#                 train_dataset.drop(round_scores)
#         if active_key == len(train_dataset) - 1:
#             mean_loss = (
#                     sum(round_scores.values())
#                     * args.grad_accumulation_steps
#                     / len(train_dataset)
#             )
#             print(
#                 f"Training epoch {epoch+1}/{args.epochs}. "
#                 f"Loss per scan: {mean_loss:.2e}"
#             )
#             if (epoch + 1) % 20 == 0:
#                 next_key = 0
#                 steps = [{} for _ in model.streams_dict]
#                 validation_round(model, valid_dataset, args)
#                 model.save(epoch=epoch)
#             report.append({f"loss": mean_loss})
#
#             epoch += 1
#             round_loss = 0
#             round_scores = {}
#
#
# @torch.no_grad()
# def validation_round(model, valid_dataset, args):
#     # first_device, last_device = streams[0].device, streams[-1].device
#     scores = []
#     samples = []
#     for bundle in valid_dataset:
#         pred = []
#         for (scan, ) in slices(bundle["scan"], shape=(res, res, args.slice_height), pad_up_to=1):
#             input = dict(
#                 scan=torch.tensor(scan).unsqueeze(0).float(),
#                 segm=torch.zeros((1, res, res, args.slice_height)),
#             )
#             model.single_forward(input)
#             pred.append(torch.nn.Upsample(scale_factor=(4, 4, 1), mode='nearest')(torch.argmax(input["pred128"], dim=1, keepdim=True).float()).squeeze(1))
#         scan = torch.tensor(bundle["scan"]).unsqueeze(0)
#         pred = torch.cat(pred, dim=-1)[..., :scan.size(-1)].cpu()
#         segm = torch.tensor(bundle["segm"]).unsqueeze(0).clamp(0, 1)
#         intersection = torch.sum(pred * segm).item() + 1
#         union = torch.sum((pred + segm).clamp(0, 1)).item() + 1
#         scores.append({"name": bundle.get("name", "no name"), "value": intersection/union})
#         for _ in range(4):
#             samples.append(report.sample(
#                 scan.detach().cpu().numpy(),
#                 # scans[1].detach().cpu().numpy(),
#                 pred.detach().cpu().numpy(),
#                 # torch.argmax(preds[1], dim=1).detach().cpu().numpy(),
#                 segm.detach().cpu().numpy()
#             ))
#     print("Validation scores are:")
#     for score in scores:
#         name = score["name"]
#         value = score["value"]
#         print(f"{name:>12}:{100 * value:6.1f}% iou")
#     mean_score = sum(item["value"] for item in scores) / len(scores)
#     report.append({f"validation_score": mean_score, "samples":samples}, commit=False)


# def to(t, device):
#     if t is None:
#         return t
#     return t.to(device=device)
#
#
# def extraenum(it, extra):
#     for i, item in enumerate(it):
#         yield i, item
#     for j in range(extra):
#         yield len(it) + j, None
#
#
# def streamlined_train_epoch(args, model, ds, optimizer, loss_and_scores):
#     round_loss = 0
#     samples = []
#     round_scores = {}
#     model.train()
#     optimizer.zero_grad()
#
#     streams = len(model.streams)
#     keys = segms = [None, ] * streams
#     tensors = [[None, ] * streams, ] * (streams + 1)
#     # print(f"Training round ({epoch+1}/{epochs}).")
#     # keys = scans = segms = intrs = preds = [None, None]
#     # occ = lambda x: '0' if x is None else 'X'
#     # def pocc(*title):
#     #     print(*title)
#     #     print("\n".join([ ''.join(map(occ, row)) for row in (keys, scans, segms, intrs, preds)]))
#     for batch_index, data in extraenum(ds, extra=streams):
#         # pocc(f"Batch {batch_index}.", "No data." if data is None else "Has data.")
#         outputs = model(*[tensors[i][i] for i in range(streams)])
#         for i in range(streams):
#             tensors[i + 1][i] = outputs[i]
#
#         # intrs[0], preds[1] = model(to(scans[0], cuda0), to(intrs[1], cuda1))
#         # pocc("Processing...")
#
#         if tensors[-1][-1] is not None:
#             # if preds[1] is not None:
#             loss, scores = loss_and_scores(
#                 to(tensors[-1][-1], cuda0),
#                 # to(preds[1], cuda0),
#                 functional.one_hot(to(segms[-1], cuda0), args.finals).permute(0, 4, 1, 2, 3).to(dtype=torch.float32),
#                 keys=[keys[-1]]
#             )
#             loss = loss / args.grad_accumulation_steps  # because of gradient accumulation
#             loss.backward()
#             round_loss += loss.item() * segms[-1].size(
#                 0) * args.grad_accumulation_steps  # segms[1].size(0) is just one, I know
#             round_scores.update(scores)
#             if args.rank == 0 and len(samples) < args.n_samples:
#                 samples.append(report.sample(
#                     tensors[0][-1].detach().cpu().numpy(),
#                     # scans[1].detach().cpu().numpy(),
#                     torch.argmax(tensors[-1][-1], dim=1).detach().cpu().numpy(),
#                     # torch.argmax(preds[1], dim=1).detach().cpu().numpy(),
#                     segms[-1].detach().cpu().numpy()
#                 ))
#
#             if ((max(0, batch_index - streams) + 1) % args.grad_accumulation_steps == 0) or (
#                     batch_index + 1 == len(ds) + streams):
#                 optimizer.step()
#                 optimizer.zero_grad()
#
#         # Time shift
#         keys = [None if data is None else data["keys"], *keys[:-1]]
#         for j in range(streams + 1):
#             tensors[j] = [None if (data is None or j != 0) else data["scan"].unsqueeze(0), *tensors[j][:-1]]
#         # scans = [None if data is None else data["scan"].unsqueeze(0), scans[0]]
#         segms = [None if data is None else data["segm"].unsqueeze(0), *segms[:-1]]
#         # intrs = [None, intrs[0]]
#         if args.finals == 2 and segms[0] is not None:
#             segms[0] = segms[0].clamp(0, 1)
#         # print(
#         #     "\tLast batch." if batch_index + 1 == len(ds) + streams else f"\tBatch {batch_index}.",
#         #     "No data" if tensors[0][0] is None else f"Scan is {tensors[0][0].shape}."
#         # )
#
#     optimizer.zero_grad(set_to_none=True)
#
#     scan_loss = round_loss / len(ds)
#     return scan_loss, round_scores, samples
#
#
# def train(model, ds, dl, device, args):
#     if args.finals == 3:
#         raw_loss = nn.CrossEntropyLoss(torch.tensor([1, 5, 20]), reduction="none").to(device=device)
#     elif args.finals == 2:
#         raw_loss = nn.CrossEntropyLoss(torch.tensor([1, 5]), reduction="none").to(device=device)
#
#     def mean_loss(pred, segm):
#         return torch.mean(raw_loss(pred, segm))
#
#     def loss_and_scores(pred, segm, keys):
#         raw = raw_loss(pred, segm)
#         loss = torch.mean(raw)
#         batch_losses = torch.mean(raw, dim=[1, 2, 3])
#         scores = {k.item(): batch_losses[i].item() for i, k in enumerate(keys)}
#         return loss, scores
#
#     # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum)
#     optimizer = AdaBelief(
#         model.parameters(),
#         lr=args.lr,
#         eps=1e-8,
#         betas=(0.9, 0.999),
#         weight_decouple=False,
#         rectify=False,
#         print_change_log=False,
#     )
#
#     for epoch in range(1, args.epochs + 1):
#         scan_loss, scores, samples = train_epoch(
#             args=args,
#             device=device,
#             data_loader=dl,
#             ds_len=len(ds),
#             loss_and_scores=loss_and_scores,
#             model=model,
#             optimizer=optimizer,
#         )
#         ds.drop(scores)
#         print(
#             f"Training epoch {epoch}/{args.epochs}. "
#             f"Loss per scan: {scan_loss:.2e}"
#             f"".ljust(30, ' ')
#         )
#         if args.rank == 0:
#             report.append({f"loss": scan_loss, "samples": samples})
#             if epoch % 20 == 0:
#                 torch.save(model.module.state_dict(), args.models_path / "last_checkpoint.pth")
#         else:
#             report.append({f"loss": scan_loss})
#
#
# def train_epoch(args, model, device, data_loader, optimizer, loss_and_scores, ds_len):
#     round_loss = 0
#     samples = []
#     round_scores = {}
#     model.train()
#     optimizer.zero_grad()
#
#     # print(f"Training round ({epoch+1}/{epochs}).")
#     for batch_index, batched_data in enumerate(data_loader):
#         scan = batched_data["scan"].to(device=device)
#         segm = batched_data["segm"].to(device=device)
#         if args.finals == 2:
#             segm = segm.clamp(0, 1)
#         keys = batched_data["keys"]
#         batch_size = segm.size(0)
#
#         with model.join():
#             pred = model(scan)
#             loss, scores = loss_and_scores(
#                 pred,
#                 functional.one_hot(segm, args.finals).permute(0, 4, 1, 2, 3).to(dtype=torch.float32),
#                 keys=keys
#             )
#             loss = loss / args.grad_accumulation_steps  # because of gradient accumulation
#             loss.backward()
#         torch.cuda.synchronize(device=args.local_rank)
#         round_loss += loss.item() * batch_size * args.grad_accumulation_steps
#         round_scores.update(scores)
#         if args.rank == 0 and len(samples) < args.n_samples:
#             samples.append(report.sample(
#                 scan.detach().cpu().numpy(),
#                 torch.argmax(pred.detach(), dim=1).cpu().numpy(),
#                 segm.detach().cpu().numpy()
#             ))
#
#         if ((batch_index + 1) % args.grad_accumulation_steps == 0) or (batch_index + 1 == len(data_loader)):
#             optimizer.step()
#             optimizer.zero_grad()
#     optimizer.zero_grad(set_to_none=True)
#
#     scan_loss = round_loss / ds_len
#     return scan_loss, round_scores, samples
