import functools
import random
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.utils.data
import rich
from adabelief_pytorch import AdaBelief
from rich import print
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import dataset
import nibabelio
import path_explorer as px
import report
import slicing
from models import *
from models import BatchNorm3d
import lovely_tensors
debug = not torch.cuda.is_available()
rich.reconfigure(width=180)
lovely_tensors.monkey_patch()

def main():
    args = SimpleNamespace(
        batch_size=1,
        # buffer_increment=1 if debug else 2,
        # buffer_size=10 if debug else 40,
        clip=(-300, 400),
        debug=debug,
        device=torch.device("cpu") if debug else torch.device("cuda:0"),
        epochs=40 if debug else 100,
        # finals=2,
        # fold_shape=(4, 4, 2) if debug else (16, 16, 8),
        grad_accumulation_steps=10,
        group_id=f"Classifier{int(time.time()) // 120:06X}",
        lr=0.002,
        models_path=Path('/gpfswork/rech/otc/uiu95bi/saved_models'),
        # n_samples=4,
        rebuild="LiverTumorD45477-0.pt",
        norm_momentum=0.1,
        repetitions=1 if debug else 4,
        slice_shape=(512, 512, 16),
        sources_path=Path('/gpfswork/rech/otc/uiu95bi/sources'),
        preds_path=Path('/gpfswork/rech/otc/uiu95bi/preds'),
        # turnover_ratio=0.05,
        xw_height=256,
        yw_height=256,
        zw_height=32,
    )
    print(f"Run with options: {vars(args)}")

    # prepare_dataset(args)
    for i in range(args.repetitions):
        args.id = args.group_id + "-" + str(i)
        mvi_model = Architecture(
            Sequential(
                Separate(
                    EncDecConnection("avg", "up_n", (2, 2, 1), cat=True)(
                        ConvBlock([4, 16, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum),
                        EncDecConnection("max", "up_n", (2, 2, 1))(
                            ConvBlock([16, 32, 32], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum),
                            EncDecConnection("max", "up_n", (2, 2, 2))(
                                ConvBlock([32, 64, 64], actv=LeakyReLU, norm=BatchNorm3d,
                                            momentum=args.norm_momentum),
                                EncDecConnection("max", "up_n", (2, 2, 2))(
                                    ConvBlock([64, 64, 64], actv=LeakyReLU, norm=BatchNorm3d,
                                                momentum=args.norm_momentum)
                                ),
                                ConvBlock([128, 64, 32], actv=LeakyReLU, norm=BatchNorm3d,
                                            momentum=args.norm_momentum, drop=Dropout3d)
                            ),
                            ConvBlock([64, 32, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum,
                                        drop=Dropout3d)
                        ),
                        ConvBlock([32, 32, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum,
                                    drop=Dropout3d)
                    ),
                    EncDecConnection("avg", "up_n", (4, 4, 1), cat=False)(
                        ConvBlock([4, 16, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum),
                        EncDecConnection("max", "up_n", (2, 2, 2))(
                            ConvBlock([16, 32, 32], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum),
                            EncDecConnection("max", "up_n", (2, 2, 2))(
                                ConvBlock([32, 64, 64], actv=LeakyReLU, norm=BatchNorm3d,
                                            momentum=args.norm_momentum),
                                EncDecConnection("max", "up_n", (2, 2, 2))(
                                    ConvBlock([64, 64, 64], actv=LeakyReLU, norm=BatchNorm3d,
                                                momentum=args.norm_momentum)
                                ),
                                ConvBlock([128, 64, 32], actv=LeakyReLU, norm=BatchNorm3d,
                                            momentum=args.norm_momentum, drop=Dropout3d)
                            ),
                            ConvBlock([64, 32, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum,
                                        drop=Dropout3d)
                        ),
                        ConvBlock([32, 32, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum,
                                    drop=Dropout3d)
                    ),
                ),
                Cat(),
                ConvBlock([4 + 16 + 16, 32, 16, 2], kernel_size=(1, 1, 1), actv=LeakyReLU),
            ),
            inputs=[("wscan", torch.float32)],
            outputs=["localmvi"],
            cuda_rank=0,
            storage=args.models_path / (args.id + ".pt"),
        )
        relevance = Architecture(
            Sequential(
                Separate(
                    EncDecConnection("avg", "up_n", (2, 2, 1), cat=True)(
                        ConvBlock([4, 16, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum),
                        EncDecConnection("max", "up_n", (2, 2, 1))(
                            ConvBlock([16, 32, 32], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum),
                            EncDecConnection("max", "up_n", (2, 2, 2))(
                                ConvBlock([32, 64, 64], actv=LeakyReLU, norm=BatchNorm3d,
                                            momentum=args.norm_momentum),
                                EncDecConnection("max", "up_n", (2, 2, 2))(
                                    ConvBlock([64, 64, 64], actv=LeakyReLU, norm=BatchNorm3d,
                                                momentum=args.norm_momentum)
                                ),
                                ConvBlock([128, 64, 32], actv=LeakyReLU, norm=BatchNorm3d,
                                            momentum=args.norm_momentum, drop=Dropout3d)
                            ),
                            ConvBlock([64, 32, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum,
                                        drop=Dropout3d)
                        ),
                        ConvBlock([32, 32, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum,
                                    drop=Dropout3d)
                    ),
                    EncDecConnection("avg", "up_n", (4, 4, 1), cat=False)(
                        ConvBlock([4, 16, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum),
                        EncDecConnection("max", "up_n", (2, 2, 2))(
                            ConvBlock([16, 32, 32], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum),
                            EncDecConnection("max", "up_n", (2, 2, 2))(
                                ConvBlock([32, 64, 64], actv=LeakyReLU, norm=BatchNorm3d,
                                            momentum=args.norm_momentum),
                                EncDecConnection("max", "up_n", (2, 2, 2))(
                                    ConvBlock([64, 64, 64], actv=LeakyReLU, norm=BatchNorm3d,
                                                momentum=args.norm_momentum)
                                ),
                                ConvBlock([128, 64, 32], actv=LeakyReLU, norm=BatchNorm3d,
                                            momentum=args.norm_momentum, drop=Dropout3d)
                            ),
                            ConvBlock([64, 32, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum,
                                        drop=Dropout3d)
                        ),
                        ConvBlock([32, 32, 16], actv=LeakyReLU, norm=BatchNorm3d, momentum=args.norm_momentum,
                                    drop=Dropout3d)
                    ),
                ),
                Cat(),
                ConvBlock([4 + 16 + 16, 32, 16, 1], kernel_size=(1, 1, 1), actv=LeakyReLU),
            ),
            inputs=[("wscan", torch.float32)],
            outputs=["relevance"],
            cuda_rank=1,
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
        class CustomLoss(CrossEntropyLoss):
            def forward(self, input, relevance, target):
                input = torch.permute(input, (0, 2, 3, 4, 1))
                relevance = torch.permute(relevance, (0, 2, 3, 4, 1))
                input = torch.reshape(input, (-1, 2))
                relevance = torch.reshape(relevance, (-1, 1))
                target = target.broadcast_to(input.shape[0])
                return (
                    super(CustomLoss, self).forward(input, target)
                    + super(CustomLoss, self).forward(torch.exp(relevance)*input, target)
                )

        loss = Architecture(
            CustomLoss(),
            inputs=["localmvi", "relevance", ("mvi", torch.int64)],
            outputs=["loss"],
            cuda_rank=0,
        )

        mvi_model.to_device()
        relevance.to_device()
        loss.to_device()

        print("Using model:", repr(mvi_model))
        # args.arch = model.stream.summary

        # train_cases, valid_cases = px.split_trainables(args.sources_path, shuffle=True, offset=i)

        # queue = dataset.queue_generator(list(train_cases), length=8, clip=args.clip)
        train_cases, valid_cases = train_test_split([p.absolute() for p in args.preds_path.iterdir()], test_size=0.10)
        train_dataset = Dataset(train_cases)
        valid_dataset = Dataset(valid_cases)

        # if debug:
        #     shape = args.slice_shape
        #     shape = (shape[0], shape[1], 4 * shape[2] + random.randint(1, 6))
        #     valid_dataset = dataset.GeneratorDataset(
        #         dataset.debug_slice_gen(None, shape),
        #         buffer_size=5,
        #         turnover_size=1
        #     )
        # else:
        #     valid_dataset = dataset.GeneratorDataset(
        #         ({"case_path": case_path} for case_path in valid_cases),
        #         post_func=functools.partial(nibabelio.load, segm=True, clip=args.clip)
        #     )

        print(f"Training dataset has {len(train_dataset)} elements.")
        print(f"Each element has a wscan of shape {train_dataset[0]['wscan'].shape}.")
        # print(f"Validation dataset has {len(valid_dataset)} elements.")

        run = report.init(config=vars(args), id=args.id, group=args.group_id, mute=debug)
        args.start_time = time.time()
        try:
            train(mvi_model, relevance, loss=loss, metrics=loss, tds=train_dataset, vds=valid_dataset, args=args)
            # if not debug:
            #     global_dataset = dataset.GeneratorDataset(
            #         ({"case_path": args.sources_path / case_path} for case_path in sorted(px.iter_trainable(args.sources_path))),
            #         post_func=functools.partial(nibabelio.load, segm=True, clip=args.clip)
            #     )
            #     validation_round(model, metrics=metrics, ds=global_dataset, epoch=args.epochs, args=args)
        finally:
            run.finish()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, files: list[Path]):
        super(Dataset, self).__init__()
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return torch.load(self.files[item])


@torch.no_grad()
def prepare_dataset(args):
    hercoles = pd.read_csv("hercoles.csv")
    premodel = Architecture.rebuild(args.models_path / args.rebuild, device=args.device)
    premodel.stream.eval()
    for case_path in px.iter_registered(args.sources_path):
        bundle = nibabelio.load(args.sources_path / case_path, segm=False, clip=args.clip)
        row = hercoles.loc[hercoles['ID_Paziente'] == bundle["name"]]
        if len(row)!=1:
            continue
        # print("Name:", bundle["name"], "Row len:", len(row))
        # print("MVI:", hercoles.loc[hercoles['ID_Paziente'] == bundle["name"]]["MVI"].to_numpy())
        bundle = apply(
            premodel,
            bundle,
            args.slice_shape,
            args
        )
        x, y, z = bundle["xw_start"], bundle["yw_start"], bundle["zw_start"]
        xl, yl, zl = 256, 256, 32
        
        torch.save(
            dict(
                name=bundle["name"],
                wscan=bundle["scan"][:, x:x + xl, y:y + yl, z:z + zl],
                mvi=torch.tensor(hercoles.loc[hercoles['ID_Paziente'] == bundle["name"]]["MVI"].to_numpy(), dtype=torch.int64).squeeze(-1)
            ),
            f"{args.preds_path / (bundle['name'] + '.pt')}"
        )


def train_epoch(*models: Architecture, loss: Architecture, ds: Dataset, epoch, optimizer, args):
    """Assuming model is single stream."""
    round_loss = 0
    optimizer.zero_grad()
    assert args.batch_size == 1
    dl = DataLoader(ds, batch_size=args.batch_size)
    for key, data in enumerate(dl):
        # key = int(data["keys"][0])
        for model in models:
            model.forward(data)
        loss.forward(data)
        data["loss"] /= args.grad_accumulation_steps
        data["loss"].backward()
        round_loss += data["loss"].item()
        # round_scores.update({key: data["loss"].item()})
        if (key + 1) % args.grad_accumulation_steps == 0 or key + 1 == len(ds):
            optimizer.step()
            optimizer.zero_grad()
    # ds.drop(round_scores)
    mean_loss = round_loss * args.grad_accumulation_steps / len(ds)
    print(
        f"Training epoch {epoch + 1}/{args.epochs}. "
        f"Loss: {mean_loss:.2e}. "
    )
    report.append({"loss": mean_loss})


@torch.no_grad()
def validation_round(*models: Architecture, metrics: Architecture, ds: Dataset, epoch=0, args):
    """Assuming model is single stream."""
    round_loss = 0
    dl = DataLoader(ds, batch_size=1)
    for data in dl:
        name = data["name"][0]
        for model in models:
            model.forward(data)
        # print(data)
        metrics.forward(data)
        round_loss += data["loss"].item()
        print(
            f"{name:>12}: "
            # f"{data['predmvi']} <> {data['mvi']}"
            f"{data['loss'].item():.3e} loss"
        )
    mean_loss = round_loss / len(ds)
    print(
        f"Validation epoch {epoch + 1}/{args.epochs}. "
        f"Loss: {mean_loss:.2e}. "
    )
    report.append({"valid_loss": mean_loss})


def train(*models: Architecture, loss: Architecture, metrics: Architecture, tds: Dataset, vds: Dataset, args):
    optimizer = AdaBelief(
        [p for model in models for p in model.stream.parameters()],
        lr=args.lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decouple=False,
        rectify=False,
        print_change_log=False,
    )
    for model in models:
        model.stream.eval()
    validation_round(*models, metrics=metrics, ds=vds, epoch=0, args=args)
    for epoch in range(args.epochs):
        for model in models:
            model.stream.train()
        train_epoch(*models, loss=loss, ds=tds, epoch=epoch, optimizer=optimizer, args=args)
        if (epoch + 1) % 20 == 0:
            for model in models:
                model.stream.eval()
            validation_round(*models, metrics=metrics, ds=vds, epoch=epoch + 1, args=args)
            # model.save()
            # args.norm_momentum = 0.4 * 0.01 + 0.6 * args.norm_momentum
            # model.stream.apply(set_norm(args.norm_momentum))
            # tds.buffer_size += args.buffer_increment
            # args.grad_accumulation_steps = (tds.buffer_size + 3) // 4
    report.append({})


def apply(model, items: dict, slice_shape: tuple[int, ...], args) -> dict:
    input_names = tuple(input__.name for input__ in model.inputs)
    inputs = tuple(input__(items).cpu().numpy() for input__ in model.inputs)
    height = inputs[0].shape[-1]
    assert all(input.shape[-1] == height for input in inputs)
    for starts, *_inputs in slicing.slices(inputs, shape=slice_shape, mode="overlapping", indices=True):
        for array in _inputs:
            array.shape = 1, *array.shape
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
