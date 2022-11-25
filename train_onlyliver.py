import functools
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import rich
from adabelief_pytorch import AdaBelief
from rich import print
from torch.nn import *
from torch.utils.data import DataLoader

import dataset
import nibabelio
import path_explorer as px
import report
from models import *
from slicing import slices

debug = not torch.cuda.is_available()
rich.reconfigure(width=180)


def main():
    args = SimpleNamespace(
        batch_size=1,
        buffer_increment=1 if debug else 2,
        buffer_size=10 if debug else 40,
        debug=debug,
        device=torch.device("cpu") if debug else torch.device("cuda:0"),
        epochs=40 if debug else 200,
        finals=2,
        fold_shape=(4, 4, 2) if debug else (16, 16, 8),
        grad_accumulation_steps=10,
        group_id=f"OnlyLiver{int(time.time()) // 120:06X}",
        lr=0.002,
        models_path=Path("../saved_models") if debug else Path('/gpfswork/rech/otc/uiu95bi/saved_models'),
        n_samples=4,
        rebuild=False,
        norm_momentum=0.9,
        repetitions=1 if debug else 4,
        slice_shape=(64, 64, 8) if debug else (512, 512, 32),
        sources_path=Path('/gpfswork/rech/otc/uiu95bi/sources'),
        turnover_ratio=0.05,
    )
    print(f"Run with options: {vars(args)}")

    for i in range(args.repetitions):
        args.id = args.group_id + "-" + str(i)

        lt = Architecture.rebuild(args.models_path / "livertumor.pt")
        try:
            model = Architecture.rebuild(args.rebuild)
            print(f"Model rebuilt from {args.rebuild}")
        except:
            print("Model is not rebuilt; a new one is initialized.")
            model = Architecture(
                Sequential(
                    Cat(),
                    Separate(
                        Identity(),
                        Sequential(
                            AvgPool3d(kernel_size=(2, 2, 1)),
                            ConvBlock([6, 16, 16], actv="LeakyReLU"),
                            FoldNorm3d(folded_shape=args.fold_shape, num_features=16, momentum=0.9),
                            SkipConnection(
                                MaxPool3d(kernel_size=(2, 2, 1)),
                                ConvBlock([16, 32, 32], actv="LeakyReLU"),
                                FoldNorm3d(folded_shape=args.fold_shape, num_features=32, momentum=0.9),
                                SkipConnection(
                                    MaxPool3d(kernel_size=(2, 2, 2)),
                                    ConvBlock([32, 64, 64], actv="LeakyReLU"),
                                    FoldNorm3d(folded_shape=args.fold_shape, num_features=64, momentum=0.9),
                                    ConvBlock([64, 32, 32], actv="LeakyReLU", norm="InstanceNorm3d"),
                                    Dropout3d(),
                                    Upsample(scale_factor=(2, 2, 2), mode='nearest'),
                                ),
                                ConvBlock([64, 32, 16], actv="LeakyReLU", norm="InstanceNorm3d"),
                                Dropout3d(),
                                Upsample(scale_factor=(2, 2, 1), mode='nearest'),
                            ),
                            ConvBlock([32, 16, 16], actv="LeakyReLU", norm="InstanceNorm3d"),
                            Dropout3d(),
                            Upsample(scale_factor=(2, 2, 1), mode='trilinear'),
                        ),
                        Sequential(
                            AvgPool3d(kernel_size=(4, 4, 1)),
                            ConvBlock([6, 16, 16], actv="LeakyReLU"),
                            FoldNorm3d(folded_shape=args.fold_shape, num_features=16, momentum=0.9),
                            SkipConnection(
                                MaxPool3d(kernel_size=(2, 2, 2)),
                                ConvBlock([16, 32, 32], actv="LeakyReLU"),
                                FoldNorm3d(folded_shape=args.fold_shape, num_features=32, momentum=0.9),
                                SkipConnection(
                                    MaxPool3d(kernel_size=(2, 2, 2)),
                                    ConvBlock([32, 64, 64], actv="LeakyReLU"),
                                    FoldNorm3d(folded_shape=args.fold_shape, num_features=64, momentum=0.9),
                                    ConvBlock([64, 32, 32], actv="LeakyReLU", norm="InstanceNorm3d"),
                                    Dropout3d(),
                                    Upsample(scale_factor=(2, 2, 2), mode='nearest'),
                                ),
                                ConvBlock([64, 32, 16], actv="LeakyReLU", norm="InstanceNorm3d"),
                                Dropout3d(),
                                Upsample(scale_factor=(2, 2, 2), mode='nearest'),
                            ),
                            ConvBlock([32, 16, 16], actv="LeakyReLU", norm="InstanceNorm3d"),
                            Dropout3d(),
                            Upsample(scale_factor=(4, 4, 1), mode='trilinear'),
                        ),
                    ),
                    Cat(),
                    ConvBlock([38, 32, 16, 2], kernel_size=(1, 1, 1), actv="LeakyReLU"),
                ),
                inputs=[("scan", torch.float32), "pred"],
                outputs=["ol_pred"],
                cuda_rank=1,
                storage=args.models_path / (args.id + ".pt"),
            )

        metrics = Architecture(
            Sequential(
                Parallel(
                    Argmax(),
                    MaskOf(index=1)
                ),
                Separate(
                    Jaccard(index=1),
                    Precision(index=1)
                )
            ),
            inputs=["ol_pred", ("segm", torch.int64)],
            outputs=["jaccard", "precision"],
        )

        loss = Architecture(
            Sequential(
                Parallel(
                    Identity(),
                    MaskOf(index=1),
                ),
                Separate(
                    CrossEntropyLoss(),
                    SoftPrecision(),
                ),
                Expression("args[0] + (1 - args[1])")
            ),
            inputs=["ol_pred", ("segm", torch.int64)],
            outputs=["loss"],
            cuda_rank=1,
        )

        lt.to_device()
        lt.stream.requires_grad_(False)
        model.to_device()
        loss.to_device()

        print("Using model:", repr(model))
        args.arch = model.stream.summary

        train_cases, valid_cases = px.split_trainables(args.sources_path, shuffle=True, offset=i)

        queue = dataset.queue_generator(list(train_cases), 5)
        train_dataset = dataset.GeneratorDataset(
            dataset.debug_slice_gen(None, args.slice_shape) if debug else dataset.train_slice_gen(queue, args),
            buffer_size=args.buffer_size,
            turnover_ratio=args.turnover_ratio,
        )

        if debug:
            shape = args.slice_shape
            shape = (shape[0], shape[1], 4 * shape[2] + random.randint(1, 6))
            valid_dataset = dataset.GeneratorDataset(
                dataset.debug_slice_gen(None, shape),
                buffer_size=5,
                turnover_size=1
            )
        else:
            valid_dataset = dataset.GeneratorDataset(
                ({"case_path": case_path} for case_path in valid_cases),
                post_func=functools.partial(nibabelio.load, segm=True, clip=(-300, 400))
            )

        print(f"Training dataset has {len(train_dataset)} elements in buffer.")
        print(
            f"Each element has a scan of shape {train_dataset[0]['scan'].shape} "
            f"and a segm of shape {train_dataset[0]['segm'].shape}"
        )
        print(f"They are slices taken from a population of {len(list(train_cases))} cases.")
        print(f"Validation dataset has {len(valid_dataset)} elements.")

        run = report.init(config=vars(args), id=args.id, group=args.group_id, mute=debug)
        args.start_time = time.time()
        try:
            train(model, lt=lt, loss=loss, metrics=metrics, tds=train_dataset, vds=valid_dataset, args=args)
        finally:
            del train_dataset, valid_dataset
            run.finish()
        try:
            queue.send(None)
            queue.send(True)
        except Exception as err:
            print(err)


def train(model: Architecture, *, lt: Architecture, loss: Architecture, metrics: Architecture,
          tds: dataset.GeneratorDataset,
          vds: dataset.GeneratorDataset, args):
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
    validation_round(model, lt=lt, metrics=metrics, ds=vds, epoch=0, args=args)
    for epoch in range(args.epochs):
        model.stream.train()
        train_epoch(model, lt=lt, loss=loss, ds=tds, epoch=epoch, optimizer=optimizer, args=args)
        if (epoch + 1) % 20 == 0:
            model.stream.eval()
            validation_round(model, lt=lt, metrics=metrics, ds=vds, epoch=epoch + 1, args=args)
            model.save()
            tds.buffer_size += args.buffer_increment
            args.grad_accumulation_steps = (tds.buffer_size + 3) // 4
    report.append({})


@torch.no_grad()
def validation_round(model: Architecture, *, lt: Architecture, metrics: Architecture, ds: dataset.GeneratorDataset,
                     epoch=0, args):
    scores = []
    samples = {}
    dl = DataLoader(ds)
    for data in dl:
        name = data.get("name", ["no name"])[0]  # 0 as in 'the name of the first (unique) case in the batch'
        scan = data["scan"]
        segm = data["segm"]
        pred = []
        for (x, t) in slices(scan, segm, shape=args.slice_shape, pad_up_to=1):
            items = lt.forward({"scan": x})
            pred.append(items["pred"])
        pred = torch.cat(pred, dim=-1)[..., :scan.shape[-1]]
        ol_pred = []
        for (x, p) in slices(scan, pred, shape=args.slice_shape, pad_up_to=1):
            items = model.forward({"scan": x, "pred": p})
            ol_pred.append(items["ol_pred"])
        ol_pred = torch.cat(ol_pred, dim=-1)[..., :scan.shape[-1]]
        _metrics = metrics.forward(dict(data, ol_pred=ol_pred))
        scores.append(dict(_metrics, name=name))
        # segm = torch.as_tensor(segm, device=pred.device).clamp(0, 1)
        # recall = losses(dict(pred=pred, segm=segm)).get("recall").item()
        # intersection = torch.sum(pred * segm).item() + 0.1
        # union = torch.sum(torch.clamp(pred + segm, 0, 1)).item() + 0.1
        ol_pred = torch.argmax(ol_pred, dim=1).cpu().numpy()
        samples[f"samples: {name}"] = report.samples(
            scan,
            ol_pred,
            segm,
            only_liver=True
        )[0]  # 0 as in 'the samples from the first (unique) case in the batch'
    print("Validation scores are:")
    for score in scores:
        name = score["name"]
        jaccard = score["jaccard"]
        precision = score["precision"]
        print(f"{name:>12}:{100 * jaccard:6.1f}% jaccard --- {100 * precision:6.1f}% precision")
    total_time = time.time() - args.start_time
    mean_time = total_time / max(1, epoch)
    print(f"Mean time: {mean_time:.0f}s per training epoch.")
    mean_jaccard = sum(item["jaccard"] for item in scores) / len(scores)
    mean_precision = sum(item["precision"] for item in scores) / len(scores)
    report.append(dict(samples, validation_score=mean_jaccard, validation_recall=mean_precision), commit=False)


def train_epoch(model: Architecture, *, lt: Architecture, loss: Architecture, ds: dataset.GeneratorDataset, epoch,
                optimizer, args):
    """Assuming model is single stream."""
    round_scores = dict()
    optimizer.zero_grad()
    assert args.batch_size == 1
    dl = DataLoader(ds, batch_size=args.batch_size)
    for data in dl:
        key = int(data["keys"][0])
        lt.forward(data)
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


if __name__ == "__main__":
    main()
