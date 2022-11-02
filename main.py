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
from report import print
from train import train

if __name__ == '__main__':
    assert torch.cuda.is_available()
    args = SimpleNamespace(
        arch="/gpfswork/rech/otc/uiu95bi/hogwild_liver/architecture.yaml",
        batch_size=1,
        buffer_size=40,
        class_weights=[1, 5, 10],
        dry_run=False,
        epochs=200,
        grad_accumulation_steps=8,
        finals=2,
        id=f"HOGWILD{int(time.time()) // 120:09X}",
        local_rank=0,  # idr.local_rank,
        lr=0.002,
        models_path=Path('/gpfswork/rech/otc/uiu95bi/saved_models'),
        n_samples=4,
        norm_momentum=0.9,
        rank=0,  # idr.rank,
        # seed=1,
        # sgd_momentum=0.9,
        slice_height=48,
        sources_path=Path('/gpfswork/rech/otc/uiu95bi/sources'),
        staging_size=2,
        warmup_size=0,
    )

    run = report.init(config=vars(args), id=args.id)
    print(f"Run with options: {vars(args)}", once=True)

    # dist.init_process_group(backend='nccl', init_method='env://', world_size=idr.size, rank=idr.rank)

    try:
        print("Try rebuilding model...")
        models_dict = models.rebuild(args.models_path / "last.pth")
    except:
        print(traceback.format_exc())
        models_dict = models.build(args.arch)
        models_dict["shrink"].load_state_dict({f"module.{key}": value for key, value in torch.load(args.models_path / "shrink.pth").items()})
    args.arch = yaml.load(open(args.arch, 'r').read(), yaml.Loader)
    shrink = models_dict["shrink"]
    model = models_dict["lrlo_model"]
    # torch.cuda.set_device(idr.local_rank)
    # gpu = torch.device("cuda")
    # model = model.to(gpu)
    shrink.to_cuda()
    model.to_cuda()
    models.set_momentum(model, args.norm_momentum)

    print("Using model:", repr(model), once=True)

    train_cases, valid_cases = px.split_trainables(args.sources_path)
    train_cases = random.sample(train_cases, len(train_cases))

    queue = dataset.queue_generator(list(train_cases), 5)
    train_dataset = dataset.GeneratorDataset(
        dataset.train_slice_gen(queue, args),
        buffer_size=args.buffer_size,
        staging_size=args.staging_size
    )

    valid_dataset = dataset.GeneratorDataset(
        iter(valid_cases),
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
        train(shrink=shrink, model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, args=args)
    except:
        print(traceback.format_exc())
    finally:
        queue.send(True)
        del train_dataset, valid_dataset
        run.finish()
    assert None is True, "Crash & burn!"
