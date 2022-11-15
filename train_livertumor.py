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
        buffer_increment=5 if debug else 2,
        buffer_size=40 if debug else 40,
        debug=debug,
        device=torch.device("cpu") if debug else torch.device("cuda:0"),
        epochs=40 if debug else 200,
        grad_accumulation_steps=10,
        finals=2,
        id=f"LiverTumor{int(time.time()) // 120:06X}",
        lr=0.002,
        models_path=Path("../saved_models") if debug else Path('/gpfswork/rech/otc/uiu95bi/saved_models'),
        n_samples=4,
        norm_momentum=0.9,
        slice_shape=(32, 32, 8) if debug else (512, 512, 32),
        sources_path=Path('/gpfswork/rech/otc/uiu95bi/sources'),
        turnover_ratio=0.05,
    )
    print(f"Run with options: {vars(args)}")

    model = Wrapper(
        Sequential(
            Stream(AsTensor, dtype=torch.float32),
            SkipConnection(
                Stream(AvgPool3d, kernel_size=(2, 2, 1)),
                ConvBlock([4, 32, 32], actv="LeakyReLU"),
                Stream(FoldNorm3d, (16, 16, 8), num_features=32, momentum=0.9),
                SkipConnection(
                    Stream(MaxPool3d, kernel_size=(2, 2, 1)),
                    ConvBlock([32, 64, 64], actv="LeakyReLU"),
                    Stream(FoldNorm3d, (8, 8, 8), num_features=64, momentum=0.9),
                    SkipConnection(
                        Stream(MaxPool3d, kernel_size=(2, 2, 2)),
                        ConvBlock([64, 128, 128], actv="LeakyReLU", norm="InstanceNorm3d"),
                        # Stream(FoldNorm3d, (4, 4, 4), num_features=64, momentum=0.9),
                        ConvBlock([128, 64, 64], actv="LeakyReLU", norm="InstanceNorm3d"),
                        # Stream(FoldNorm3d, (4, 4, 4), num_features=32, momentum=0.9),
                        Stream(Dropout3d),
                        Stream(Upsample, scale_factor=(2, 2, 2), mode='nearest'),
                        dim=1,
                    ),
                    ConvBlock([128, 64, 32], actv="LeakyReLU", norm="InstanceNorm3d"),
                    # Stream(FoldNorm3d, (8, 8, 8), num_features=16, momentum=0.9),
                    Stream(Dropout3d),
                    Stream(Upsample, scale_factor=(2, 2, 1), mode='nearest'),
                    dim=1,
                ),
                ConvBlock([64, 32, 28], actv="LeakyReLU", norm="InstanceNorm3d"),
                # Stream(InstanceNorm3d, 12),
                Stream(Dropout3d),
                Stream(Upsample, scale_factor=(2, 2, 1), mode='trilinear'),
                dim=1,
            ),
            ConvBlock([32, 16, 16, 2], kernel_size=(1, 1, 1), actv="LeakyReLU"),
        ),
        inputs=["scan"],
        outputs=["pred"],
        rank=0,
        storage=args.models_path / "last.pth",
    )

    losses = Wrapper(
        Sequential(
            Parallel(
                Stream("Identity"),
                Stream("Clamp", 0, 1),
            ),
            Separate(
                Stream("CrossEntropyLoss"),
                Stream("Recall", argmax_input_dim=1),
            ),
        ),
        inputs=["pred", "segm"],
        outputs=["cross", "recall"],
        rank=0,
    )

    model.to_device()
    losses.to_device()

    print("Using model:", repr(model))
    args.arch = model.stream.summary

    train_cases, valid_cases = px.split_trainables(args.sources_path)
    train_cases = random.sample(train_cases, len(train_cases))

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
            post_func=functools.partial(nibabelio.load, train=True, clip=(-300, 400))
        )

    print(f"Training dataset has {len(train_dataset)} elements in buffer.")
    print(
        f"Each element has a scan of shape {train_dataset[0]['scan'].shape} "
        f"and a segm of shape {train_dataset[0]['segm'].shape}"
    )
    print(f"They are slices taken from a population of {len(list(train_cases))} cases.")
    print(f"Validation dataset has {len(valid_dataset)} elements.")

    run = report.init(config=vars(args), id=args.id, mute=debug)
    args.start_time = time.time()
    try:
        optimizer = AdaBelief(
            model.parameters(),
            lr=args.lr,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=False,
            rectify=False,
            print_change_log=False,
        )
        validation_round(model, losses, ds=valid_dataset, args=args)
        train(model, losses, tds=train_dataset, vds=valid_dataset, args=args)
        # train(ddp_model, train_dataset, train_loader, gpu, args)
        # train(model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, args=args)
    except:
        print(traceback.format_exc())
    finally:
        queue.send(True)
        del train_dataset, valid_dataset
        run.finish()
    assert None is True, "Crash & burn!"
