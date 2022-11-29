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
import slicing
from models import *

debug = not torch.cuda.is_available()
rich.reconfigure(width=180)

def main():
    args = SimpleNamespace(
        batch_size=1,
        buffer_increment=1 if debug else 2,
        buffer_size=10 if debug else 40,
        clip=(-300, 400),
        debug=debug,
        device=torch.device("cpu") if debug else torch.device("cuda:0"),
        epochs=40 if debug else 200,
        finals=2,
        fold_shape=(4, 4, 2) if debug else (16, 16, 8),
        grad_accumulation_steps=10,
        group_id=f"LiverTumor{int(time.time()) // 120:06X}",
        lr=0.002,
        models_path=Path("../saved_models") if debug else Path('/gpfswork/rech/otc/uiu95bi/saved_models'),
        n_samples=4,
        rebuild=False,
        norm_momentum=0.1,
        repetitions=1 if debug else 1,
        slice_shape=(64, 64, 8) if debug else (512, 512, 32),
        sources_path=Path('/gpfswork/rech/otc/uiu95bi/sources'),
        turnover_ratio=0.05,
        zw_height=8 if debug else 32,
    )
    print(f"Run with options: {vars(args)}")

    for i in range(args.repetitions):
        args.id = args.group_id + "-" + str(i)
        args.norm_momentum = 0.2

        if args.rebuild:
            model = Architecture.rebuild(args.rebuild)
        else:
            model = Architecture(
                Sequential(
                    Separate(
                        EncDecConnection("avg", "up_n", (2, 2, 1), cat=True)(
                            ConvBlock([4, 16, 16], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum),
                            EncDecConnection("max", "up_n", (2, 2, 1))(
                                ConvBlock([16, 32, 32], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum),
                                EncDecConnection("max", "up_n", (2, 2, 2))(
                                    ConvBlock([32, 64, 64], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum),
                                    EncDecConnection("max", "up_n", (2, 2, 2))(
                                        ConvBlock([64, 64, 64], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum)
                                    ),
                                    ConvBlock([128, 64, 32], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum, drop=Dropout3d)
                                ),
                                ConvBlock([64, 32, 16], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum, drop=Dropout3d)
                            ),
                            ConvBlock([32, 32, 16], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum, drop=Dropout3d)
                        ),
                        EncDecConnection("avg", "up_n", (4, 4, 1), cat=False)(
                            ConvBlock([4, 16, 16], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum),
                            EncDecConnection("max", "up_n", (2, 2, 2))(
                                ConvBlock([16, 32, 32], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum),
                                EncDecConnection("max", "up_n", (2, 2, 2))(
                                    ConvBlock([32, 64, 64], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum),
                                    EncDecConnection("max", "up_n", (2, 2, 2))(
                                        ConvBlock([64, 64, 64], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum)
                                    ),
                                    ConvBlock([128, 64, 32], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum, drop=Dropout3d)
                                ),
                                ConvBlock([64, 32, 16], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum, drop=Dropout3d)
                            ),
                            ConvBlock([32, 32, 16], actv=LeakyReLU, norm=IRNorm3d, momentum=args.norm_momentum, drop=Dropout3d)
                        ),
                    ),
                    Cat(),
                    ConvBlock([4+16+16, 32, 16, 2], kernel_size=(1, 1, 1), actv=LeakyReLU),
                ),
                inputs=[("scan", torch.float32)],
                outputs=["pred"],
                cuda_rank=0,
                storage=args.models_path / (args.id + ".pt"),
            )

        metrics = Architecture(
            Sequential(
                Parallel(
                    Argmax(),
                    Clamp(0, 1)
                ),
                Separate(
                    Jaccard(index=1),
                    Recall(index=1)
                )
            ),
            inputs=["pred", ("segm", torch.int64)],
            outputs=["jaccard", "recall"],
        )

        loss = Architecture(
            Sequential(
                Parallel(
                    Identity(),
                    Clamp(0, 1),
                ),
                Separate(
                    CrossEntropyLoss(),
                    SoftRecall(),
                ),
                Expression("args[0] + (1 - args[1])")
            ),
            inputs=["pred", ("segm", torch.int64)],
            outputs=["loss"],
            cuda_rank=0,
        )

        model.to_device()
        loss.to_device()

        print("Using model:", repr(model))
        args.arch = model.stream.summary

        train_cases, valid_cases = px.split_trainables(args.sources_path, shuffle=True, offset=i)

        queue = dataset.queue_generator(list(train_cases), length=8, clip=args.clip)
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
                post_func=functools.partial(nibabelio.load, segm=True, clip=args.clip)
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
            train(model, loss=loss, metrics=metrics, tds=train_dataset, vds=valid_dataset, args=args)
            # if not debug:
            #     global_dataset = dataset.GeneratorDataset(
            #         ({"case_path": args.sources_path / case_path} for case_path in sorted(px.iter_trainable(args.sources_path))),
            #         post_func=functools.partial(nibabelio.load, segm=True, clip=args.clip)
            #     )
            #     validation_round(model, metrics=metrics, ds=global_dataset, epoch=args.epochs, args=args)
        finally:
            del train_dataset, valid_dataset
            run.finish()
        try:
            queue.send(None)
            queue.send(True)
        except Exception as err:
            print(err)


def train_epoch(model: Architecture, *, loss: Architecture, ds: dataset.GeneratorDataset, epoch, optimizer, args):
    """Assuming model is single stream."""
    round_scores = dict()
    optimizer.zero_grad()
    assert args.batch_size == 1
    dl = DataLoader(ds, batch_size=args.batch_size)
    for data in dl:
        key = int(data["keys"][0])
        model.forward(data)
        loss.forward(data)
        data["loss"] /= args.grad_accumulation_steps
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
def validation_round(model: Architecture, *, metrics: Architecture, ds: dataset.GeneratorDataset, epoch=0, args):
    scores = []
    samples = {}
    dl = DataLoader(ds)
    for data in dl:
        name = data.get("name", ["no name"])[0]  # 0 as in 'the name of the first (unique) case in the batch'
        scan = data["scan"]
        segm = data["segm"]
        apply(model, data, args.slice_shape, args=args)
        pred = data["pred"]
        _metrics = metrics.forward(dict(data, pred=pred))
        zw_start = data["zw_start"]
        # print(f"Case {name}. Window [..., {zw_start}:{zw_start+args.zw_height}]")
        # print(
        #     f"Positives: {(segm[..., zw_start:zw_start+args.zw_height] > 0).sum().item()}.",
        #     F"Total: {(1+(segm > 0).sum()).item()}."
        # )
        window_coverage = (segm[..., zw_start:zw_start+args.zw_height] > 0).sum() / (1+(segm > 0).sum())
        scores.append(dict(_metrics, name=name, zw_start=zw_start, coverage=window_coverage))
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
        zw_start, coverage = score["zw_start"], score["coverage"]
        print(
            f"{name:>12}:"
            f"{100 * jaccard:6.1f}% jaccard"
            f"{100 * recall:6.1f}% recall"
            f"{100 * coverage:6.1f}% coverage starting at {zw_start}"
        )
    total_time = time.time() - args.start_time
    mean_time = total_time / max(1, epoch)
    print(f"Mean time: {mean_time:.0f}s per training epoch.")
    mean_jaccard = sum(item["jaccard"] for item in scores) / len(scores)
    mean_recall = sum(item["recall"] for item in scores) / len(scores)
    report.append(dict(samples, validation_score=mean_jaccard, validation_recall=mean_recall), commit=False)


def train(model: Architecture, *, loss: Architecture, metrics: Architecture, tds: dataset.GeneratorDataset,
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
    validation_round(model, metrics=metrics, ds=vds, epoch=0, args=args)
    for epoch in range(args.epochs):
        model.stream.train()
        train_epoch(model, loss=loss, ds=tds, epoch=epoch, optimizer=optimizer, args=args)
        if (epoch + 1) % 20 == 0:
            model.stream.eval()
            validation_round(model, metrics=metrics, ds=vds, epoch=epoch + 1, args=args)
            model.save()
            args.norm_momentum = 0.5 * (0.05 + args.norm_momentum)
            model.stream.apply(set_norm(args.norm_momentum))
            tds.buffer_size += args.buffer_increment
            args.grad_accumulation_steps = (tds.buffer_size + 3) // 4
    report.append({})


def apply(model, items: dict, slice_shape: tuple[int, ...], args) -> dict:
    input_names = tuple(input__.name for input__ in model.inputs)
    inputs = tuple(input__(items).cpu().numpy() for input__ in model.inputs)
    height = inputs[0].shape[-1]
    assert all(input.shape[-1] == height for input in inputs)
    for starts, *_inputs in slicing.slices(inputs, shape=slice_shape, mode="overlapping", indices=True):
        piece_items = model.forward({
            name: input__(array)
            for name, input__, array in zip(input_names, model.inputs, _inputs)
        })
        try:
            for output in model.outputs:
                items[output] = torch.cat([
                    items[output][..., :starts[-1]],
                    piece_items[output]
                ], dim=-1)
        except KeyError:
            for output in model.outputs:
                items[output] = piece_items[output]
    profile = (torch.argmax(items["pred"], dim=1) == 1).sum(dim=(0, 1, 2))
    # profile = torch.sum(items["pred"], dim=(0, 2, 3))[1]
    items["zw_start"] = max(range(len(profile)), key=lambda i: (profile[i:i+args.zw_height]).sum().item())

    return items

def set_norm(norm):
    def _set_norm(module):
        if isinstance(module, (BatchNorm3d, InstanceNorm3d, IRNorm3d)):
            module.momentum = norm
    return _set_norm


if __name__=="__main__":
    main()