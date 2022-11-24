import time

import torch
from adabelief_pytorch import AdaBelief
from rich import print
from torch.utils.data import DataLoader

import report
from dataset import GeneratorDataset
from models import Architecture
from slicing import slices


def train_epoch(model: Architecture, *, loss: Architecture, ds: GeneratorDataset, epoch, optimizer, args):
    """Assuming model is single stream."""
    round_scores = dict()
    optimizer.zero_grad()
    assert args.batch_size == 1
    dl = DataLoader(ds, batch_size=args.batch_size)
    for data in dl:
        key = int(data["keys"][0])
        model.forward(data)
        loss.forward(data)
        data["loss"].backward()
        round_scores.update({key: data["loss"].item()})
        if (key + 1) % args.grad_accumulation_steps == 0 or key + 1 == len(ds):
            optimizer.step()
            optimizer.zero_grad()
    ds.drop(round_scores)
    mean_loss = sum(round_scores.values()) * args.grad_accumulation_steps / len(ds)
    print(
        f"Training epoch {epoch + 1}/{args.epochs}. "
        f"Loss: {mean_loss:.2e}. "
    )
    report.append({"loss": mean_loss})


@torch.no_grad()
def validation_round(model: Architecture, *, metrics: Architecture, ds: GeneratorDataset, epoch=0, args):
    scores = []
    samples = {}
    dl = DataLoader(ds)
    for data in dl:
        name = data.get("name", ["no name"])[0]  # 0 as in 'the name of the first (unique) case in the batch'
        scan = data["scan"]
        segm = data["segm"]
        pred = []
        for (x, t) in slices(scan, segm, shape=args.slice_shape, pad_up_to=1):
            items = model.forward({"scan": x})
            pred.append(items["pred"])
        pred = torch.cat(pred, dim=-1)[..., :scan.shape[-1]]
        _metrics = metrics.forward(dict(data, pred=pred))
        scores.append(dict(_metrics, name=name))
        # segm = torch.as_tensor(segm, device=pred.device).clamp(0, 1)
        # recall = losses(dict(pred=pred, segm=segm)).get("recall").item()
        # intersection = torch.sum(pred * segm).item() + 0.1
        # union = torch.sum(torch.clamp(pred + segm, 0, 1)).item() + 0.1
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        samples[f"samples: {name}"] = report.samples(
            scan,
            pred,
            segm,
        )[0]  # 0 as in 'the samples from the first (unique) case in the batch'
    print("Validation scores are:")
    for score in scores:
        name = score["name"]
        jaccard = score["jaccard"]
        recall = score["recall"]
        print(f"{name:>12}:{100 * jaccard:6.1f}% jaccard --- {100 * recall:6.1f}% recall")
    total_time = time.time() - args.start_time
    mean_time = total_time / max(1, epoch)
    print(f"Mean time: {mean_time:.0f}s per training epoch.")
    mean_jaccard = sum(item["jaccard"] for item in scores) / len(scores)
    mean_recall = sum(item["recall"] for item in scores) / len(scores)
    report.append(dict(samples, validation_score=mean_jaccard, validation_recall=mean_recall), commit=False)


def train(model: Architecture, *, loss: Architecture, metrics: Architecture, tds: GeneratorDataset,
          vds: GeneratorDataset, args):
    optimizer = AdaBelief(
        model.stream.parameters(),
        lr=args.lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=False,
        print_change_log=False,
    )
    model.stream.eval()
    validation_round(model, metrics=metrics, ds=vds, epoch=0, args=args)
    for epoch in range(args.epochs):
        model.stream.train()
        train_epoch(model, loss=loss, ds=tds, epoch=epoch, optimizer=optimizer, args=args)
        if (epoch + 1) % 20 == 0:
            model.stream.eval()
            validation_round(model, metrics=metrics, ds=vds, epoch=epoch + 1, args=args)
            model.save()
            tds.buffer_size += args.buffer_increment
            args.grad_accumulation_steps = (tds.buffer_size + 3) // 4
    report.append({})
