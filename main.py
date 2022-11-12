import functools
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

# import idr_torch as idr
import torch
import yaml
from adabelief_pytorch import AdaBelief
from torch import nn

# from torchvision import datasets, transforms
import dataset
import models
import nibabelio
import path_explorer as px
import report
from rich import print

from train import train_epoch, validation_round, train
from models import *

debug = Path(".env").exists()
res = 64 if debug else 512

if __name__ == '__main__':
    args = SimpleNamespace(
        batch_size=1,
        buffer_size=3 if debug else 40,
        # class_weights=[1, 5, 10],
        # dry_run=False,
        device=torch.device("cpu") if debug else torch.device("cuda:0"),
        epochs=20 if debug else 200,
        grad_accumulation_steps=10,
        finals=2,
        id=f"HOGWILD{int(time.time()) // 120:09X}",
        local_rank=0,  # idr.local_rank,
        lr=0.002,
        models_path=Path("../saved_models") if debug else Path('/gpfswork/rech/otc/uiu95bi/saved_models'),
        n_samples=4,
        norm_momentum=0.9,
        rank=0,  # idr.rank,
        # seed=1,
        # sgd_momentum=0.9,
        slice_shape=(32, 32, 8) if debug else (512, 512, 32),
        sources_path=Path('/gpfswork/rech/otc/uiu95bi/sources'),
        staging_size=1,
        warmup_size=0,
    )

    run = report.init(config=vars(args), id=args.id, mute=debug)
    print(f"Run with options: {vars(args)}")

    # dist.init_process_group(backend='nccl', init_method='env://', world_size=idr.size, rank=idr.rank)

    print("Try building model...")
    model = Wrapper(
        Sequential(
            Stream("as_tensor", dtype=torch.float32),
            ConvBlock("conv", [4, 16, 16], actv="ELU", norm="InstanceNorm3d"),
            SkipCat(
                Sequential(
                    Stream("MaxPool3d", kernel_size=(2, 2, 2)),
                    ConvBlock("conv", [16, 32, 32], actv="ELU", norm="InstanceNorm3d"),
                    ConvBlock("conv", [32, 16, 16], actv="ELU", norm="InstanceNorm3d", drop="Dropout3d"),
                    Stream("Upsample", scale_factor=(2, 2, 2), mode='nearest')
                ),
                dim=1,
            ),
            ConvBlock("conv", [32, 16, 16], actv="ELU", norm="InstanceNorm3d", drop="Dropout3d"),
            ConvBlock("pconv", [16, 8, 4, 2], actv="ReLU"),
        ),
        inputs=["scan"],
        outputs=["pred"],
        rank=0,
        storage=args.models_path/"last.pth",
    )
    losses = Wrapper(
        Sequential(
            Separated(
                Stream("Identity"),
                Stream("Clamp", 0, 1),
            ),
            Split(
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
    # models.set_momentum(model, args.norm_momentum)

    print("Using model:", repr(model))
    args.arch = model.stream.repr_dict
    train_cases, valid_cases = px.split_trainables(args.sources_path)
    train_cases = random.sample(train_cases, len(train_cases))

    queue = dataset.queue_generator(list(train_cases), 5)
    train_dataset = dataset.GeneratorDataset(
        dataset.debug_slice_gen(None, args.slice_shape) if debug else dataset.train_slice_gen(queue, args),
        buffer_size=args.buffer_size,
        staging_size=args.staging_size
    )

    if debug:
        shape = args.slice_shape
        shape = (shape[0], shape[1], 4 * shape[2] + random.randint(1, 6))
        valid_dataset = dataset.GeneratorDataset(
            dataset.debug_slice_gen(None, shape),
            buffer_size=5,
            staging_size=1
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
    print(f"Validation dataset has {len(valid_dataset)} elements.")

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset,
    #     num_replicas=idr.size,
    #     rank=idr.rank,
    # )
    #
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=True,
    #     sampler=train_sampler,
    # )

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
        validation_round(model, ds=valid_dataset, args=args)
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
