import functools
import random
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import rich
from adabelief_pytorch import AdaBelief
from rich import print
from torch.utils.data import DataLoader

import dataset
import nibabelio
import path_explorer as px
import report
import slicing
from models import *
from models import BatchNorm3d

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
        epochs=40 if debug else 400,
        finals=2,
        fold_shape=(4, 4, 2) if debug else (16, 16, 8),
        grad_accumulation_steps=10,
        group_id=f"LiverTumor{int(time.time()) // 120:06X}",
        lr=0.002,
        models_path=Path("../saved_models") if debug else Path('/gpfswork/rech/otc/uiu95bi/saved_models'),
        n_samples=4,
        rebuild="LiverTumorD45A4C-0.pt" if debug else "LiverTumorD45477-0.pt",
        norm_momentum=0.1,
        repetitions=1 if debug else 4,
        slice_shape=(64, 64, 8) if debug else (512, 512, 32),
        sources_path=Path("../tmpdir") if debug else Path('/gpfswork/rech/otc/uiu95bi/sources'),
        turnover_ratio=0.05,
        xw_height=16 if debug else 256,
        yw_height=16 if debug else 256,
        zw_height=8 if debug else 32,
    )
    print(f"Run with options: {vars(args)}")


    prepare_dataset(args)
    for i in range(args.repetitions):
        args.id = args.group_id + "-" + str(i)
        model = Architecture(
            Sequential(
                ConvBlock([4, 16, 16], actv=LeakyReLU, norm=BatchNorm3d),
                MaxPool3d(kernel_size=(2, 2, 1)),  # 16 128 128 32
                ConvBlock([16, 32, 32], actv=LeakyReLU, norm=BatchNorm3d),
                MaxPool3d(kernel_size=(2, 2, 1)),  # 32 64 64 32
                ConvBlock([32, 64, 64], actv=LeakyReLU, norm=BatchNorm3d),
                MaxPool3d(kernel_size=(2, 2, 2)),  # 64 32 32 16
                ConvBlock([64, 128, 128], actv=LeakyReLU, norm=BatchNorm3d),
                MaxPool3d(kernel_size=(2, 2, 2)),  # 128 16 16 8
                ConvBlock([128, 256, 256], actv=LeakyReLU, norm=BatchNorm3d),
                MaxPool3d(kernel_size=(2, 2, 2)),  # 256 8 8 4 = 1024 64 = 65536
                Flatten(),
                Linear(2**16, 2**12),
                LeakyReLU(),
                Linear(2**12, 2**8),
                LeakyReLU(),
                Linear(2**8, 2**4),
                LeakyReLU(),
                Linear(2**4, 2),
            ),
            inputs=[("wscan", torch.float32)],
            outputs=["mvi"],
            cuda_rank=0,
            storage=args.models_path / (args.id + ".pt"),
        )

        # metrics = Architecture(
        #     Sequential(
        #         Parallel(
        #             Argmax(),
        #             Clamp(0, 1)
        #         ),
        #         Separate(
        #             Jaccard(index=1),
        #             Recall(index=1)
        #         )
        #     ),
        #     inputs=["mvi", ("target", torch.int64)],
        #     outputs=["cross"],
        # )

        loss = Architecture(
            CrossEntropyLoss(),
            inputs=["mvi", ("target", torch.int64)],
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
            dataset.debug_slice_gen(None, args.slice_shape) if debug else dataset.classifier_gen(queue, premodel, args),
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
            train(model, loss=loss, metrics=loss, tds=train_dataset, vds=valid_dataset, args=args)
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

def prepare_dataset(args):
    hercoles = pd.read_csv("hercoles.csv")
    premodel = Architecture.rebuild(args.models_path / args.rebuild)
    for case_path in px.iter_registered(args.sources_path):
        bundle = nibabelio.load(args.sources_path / case_path, segm=False, clip=args.clip)
        bundle = apply(premodel, bundle, args.slice_shape, args)
        x, y, z = bundle["xw_start"], bundle["yw_start"], bundle["zw_start"]
        xl, yl, zl = 256, 256, 32
        print(
            dict(
                name= bundle["name"],
                wscan=bundle["scan"][:, x:x + xl, y:y + yl, z:z + zl],
                mvi=torch.tensor(hercoles.loc[hercoles['ID_Paziente'] == bundle["name"]]["MVI"], dtype=torch.int64)
            )
        )
    assert False


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
        xw_start = data["xw_start"]
        yw_start = data["yw_start"]
        zw_start = data["zw_start"]
        # print(f"Case {name}. Window [..., {zw_start}:{zw_start+args.zw_height}]")
        # print(
        #     f"Positives: {(segm[..., zw_start:zw_start+args.zw_height] > 0).sum().item()}.",
        #     F"Total: {(1+(segm > 0).sum()).item()}."
        # )
        liver_coverage = (1 + (segm[...,
                               xw_start:xw_start + args.xw_height,
                               yw_start:yw_start + args.yw_height,
                               zw_start:zw_start + args.zw_height
                               ] == 1).sum()) / (1 + (segm == 1).sum())
        tumor_coverage = (1 + (segm[...,
                               xw_start:xw_start + args.xw_height,
                               yw_start:yw_start + args.yw_height,
                               zw_start:zw_start + args.zw_height
                               ] == 2).sum()) / (1 + (segm == 2).sum())
        scores.append(dict(_metrics, name=name, xw_start=xw_start, yw_start=yw_start, zw_start=zw_start,
                           liver_coverage=liver_coverage, tumor_coverage=tumor_coverage))
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
        coverage = score["liver_coverage"]
        print(
            f"{name:>12}:"
            f"{100 * jaccard:6.1f}% jaccard"
            f"{100 * recall:6.1f}% recall"
            f"{100 * score['liver_coverage']:6.1f}% - {100 * score['tumor_coverage']:.1f}% coverage "
            f"starting at {(score['xw_start'], score['yw_start'], score['zw_start'])}"
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
    model.stream.apply(set_norm(args.norm_momentum))
    for epoch in range(args.epochs):
        model.stream.train()
        train_epoch(model, loss=loss, ds=tds, epoch=epoch, optimizer=optimizer, args=args)
        if (epoch + 1) % 20 == 0:
            model.stream.eval()
            validation_round(model, metrics=metrics, ds=vds, epoch=epoch + 1, args=args)
            model.save()
            # args.norm_momentum = 0.4 * 0.01 + 0.6 * args.norm_momentum
            # model.stream.apply(set_norm(args.norm_momentum))
            tds.buffer_size += args.buffer_increment
            args.grad_accumulation_steps = (tds.buffer_size + 3) // 4
    report.append({})


def apply(model, items: dict, slice_shape: tuple[int, ...], args) -> dict:
    input_names = tuple(input__.name for input__ in model.inputs)
    inputs = tuple(input__(items).cpu().numpy() for input__ in model.inputs)
    height = inputs[0].shape[-1]
    assert all(input.shape[-1] == height for input in inputs)
    for starts, *_inputs in slicing.slices(inputs, shape=slice_shape, mode="overlapping", indices=True):
        print(starts)
        piece_items = model.forward({
            name: input__(array.reshape([1, *array.shape]))
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
    items["zw_start"] = z = max(range(len(profile)), key=lambda i: (profile[i:i + args.zw_height]).sum().item())
    profile = (torch.argmax(items["pred"], dim=1)[..., z:z + args.zw_height] == 1).sum(dim=(0, 1, 3))
    items["yw_start"] = y = max(range(len(profile) - args.yw_height),
                                key=lambda i: (profile[i:i + args.yw_height]).sum().item())
    profile = (torch.argmax(items["pred"], dim=1)[..., y:y + args.yw_height, z:z + args.zw_height] == 1).sum(
        dim=(0, 2, 3))
    items["xw_start"] = x = max(range(len(profile) - args.xw_height),
                                key=lambda i: (profile[i:i + args.xw_height]).sum().item())

    return items


def set_norm(norm):
    def _set_norm(module):
        if isinstance(module, (BatchNorm3d, InstanceNorm3d, IRNorm3d)):
            module.momentum = norm

    return _set_norm


if __name__ == "__main__":
    main()
