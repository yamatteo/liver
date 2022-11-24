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

import dataset
import nibabelio
import path_explorer as px
import report
from models import *
from train import validation_round, train

debug = not torch.cuda.is_available()
rich.reconfigure(width=180)


if __name__ == '__main__':
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
        group_id=f"LiverTumor{int(time.time()) // 120:06X}",
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

        if args.rebuild:
            model = Architecture.rebuild(args.rebuild)
        else:
            model = Architecture(
                Sequential(
                    Separate(
                        Identity(),
                        Sequential(
                            AvgPool3d(kernel_size=(2, 2, 1)),
                            ConvBlock([4, 16, 16], actv="LeakyReLU"),
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
                            ConvBlock([4, 16, 16], actv="LeakyReLU"),
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
                    ConvBlock([36, 32, 16, 2], kernel_size=(1, 1, 1), actv="LeakyReLU"),
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
            optimizer = AdaBelief(
                model.stream.parameters(),
                lr=args.lr,
                eps=1e-8,
                betas=(0.9, 0.999),
                weight_decouple=False,
                rectify=False,
                print_change_log=False,
            )
            train(model, loss=loss, metrics=metrics, tds=train_dataset, vds=valid_dataset, args=args)
        finally:
            del train_dataset, valid_dataset
            run.finish()
        try:
            queue.send(None)
            queue.send(True)
        except Exception as err:
            print(err)
