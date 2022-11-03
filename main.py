import functools
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

# import idr_torch as idr
import torch
import yaml

# from torchvision import datasets, transforms
import dataset
import models
import nibabelio
import path_explorer as px
import report
from rich import print
from train import train

debug = Path(".env").exists()
res = 64 if debug else 512

if __name__ == '__main__':
    args = SimpleNamespace(
        arch="architecture.yaml" if debug else "/gpfswork/rech/otc/uiu95bi/livertumor/architecture.yaml",
        batch_size=1,
        buffer_size=5 if debug else 40,
        # class_weights=[1, 5, 10],
        # dry_run=False,
        epochs=200,
        grad_accumulation_steps=8,
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
        slice_height=16 if debug else 48,
        sources_path=Path('/gpfswork/rech/otc/uiu95bi/sources'),
        staging_size=1,
        warmup_size=0,
    )

    run = report.init(config=vars(args), id=args.id, mute=debug)
    print(f"Run with options: {vars(args)}")

    # dist.init_process_group(backend='nccl', init_method='env://', world_size=idr.size, rank=idr.rank)

    # try:
    print("Try rebuilding model...")
    model = models.build(args.arch, models_path=args.models_path)
    # except:
        # print(traceback.format_exc())
        # models_dict = models.build(args.arch)
        # models_dict["shrink"].load_state_dict({f"module.{key}": value for key, value in torch.load(args.models_path / "shrink.pth").items()})
    args.arch = yaml.load(open(args.arch, 'r').read(), yaml.Loader)
    # torch.cuda.set_device(idr.local_rank)
    # gpu = torch.device("cuda")
    # model = model.to(gpu)
    if debug:
        model.to_cpu()
    else:
        model.to_cuda()
    models.set_momentum(model, args.norm_momentum)

    print("Using model:", repr(model))

    train_cases, valid_cases = px.split_trainables(args.sources_path)
    train_cases = random.sample(train_cases, len(train_cases))

    queue = dataset.queue_generator(list(train_cases), 5)
    train_dataset = dataset.GeneratorDataset(
        dataset.debug_slice_gen(None, args.slice_height) if debug else dataset.train_slice_gen(queue, args),
        buffer_size=args.buffer_size,
        staging_size=args.staging_size
    )

    if debug:
        valid_dataset = dataset.GeneratorDataset(
            dataset.debug_slice_gen(None, 4*args.slice_height+random.randint(1, 6)),
            buffer_size=5,
            staging_size=1
        )
    else:
        valid_dataset = dataset.GeneratorDataset(
            ({"case_path":case_path} for case_path in valid_cases),
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
        # train(ddp_model, train_dataset, train_loader, gpu, args)
        train(model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, args=args)
    except:
        print(traceback.format_exc())
    finally:
        queue.send(True)
        del train_dataset, valid_dataset
        run.finish()
    assert None is True, "Crash & burn!"
