import heapq
import multiprocessing as mp
from collections import namedtuple
from itertools import cycle
from pathlib import Path
from typing import Iterator

import torch.utils.data
from rich import print

# import idr_torch as idr
import nibabelio


class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator: Iterator, *, buffer_size: int = None, staging_size: int = None, post_func=None):
        super(GeneratorDataset, self).__init__()
        self.generator = generator
        if buffer_size is None:
            self.buffer = list(generator)
            self.buffer_size = len(self.buffer)
            self.staging_size = None
        else:
            self.buffer = []
            self.buffer_size = buffer_size
            self.staging_size = staging_size
            self.fill()
        self.post_func = post_func

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, i: int):
        if self.post_func is not None:
            return self.post_func(self.buffer[i])
        return {"keys": i, **self.buffer[i]}

    def fill(self, size=None):
        size = size or self.buffer_size
        for _ in range(size):
            if len(self.buffer) >= size:
                break
            self.buffer.append(next(self.generator))

    def drop(self, scores: dict[int, float]):
        smallest = heapq.nsmallest(len(self.buffer) - self.buffer_size + self.staging_size, list(scores.keys()),
                                   lambda i: scores[i])
        smallest = reversed(sorted(smallest))
        for k in smallest:
            del self.buffer[k]
        self.fill()

    def warmup(self, size, evaluator):
        item_score = namedtuple('item_score', ['item', 'score'])
        buffer = []
        # print(f"{idr.rank}-{idr.local_rank}".ljust(12, ' '), f"Warming up, {size} item to process.")
        print(f"Warming up, {size} item to process.")
        for i in range(size):
            item = next(self.generator)
            score = evaluator(item)
            buffer.append(item_score(item, score))
            if len(buffer) > self.buffer_size:
                buffer = sorted(buffer, key=lambda t: -t.score)[:self.buffer_size]
        self.buffer = [t.item for t in buffer]


def put(q, case_path, deform):
    # print("Loading", case_path)
    bundle = nibabelio.load(case_path, train=True, clip=(-300, 400))
    if deform:
        print("Applying elastic deformation to", case_path)
        bundle = bundle.deformed()
    q.put(dict(
        scan=bundle.scan.detach().numpy(),
        segm=bundle.segm.detach().numpy()
    ))


def queue_generator(case_list: list[Path], length=2):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = []
    num_cases = len(case_list)
    case_list = cycle(case_list)  # To not run out of inputs
    stop = None
    computed = 0
    while stop is None:
        if len(procs) < length:
            p = ctx.Process(target=put, args=(q, next(case_list), computed >= num_cases))
            procs.append(p)
            p.start()
            computed += 1
            continue
        stop = yield q.get()
        alive = []
        for p in procs:
            try:
                p.close()
            except ValueError:
                alive.append(p)
        procs = alive
    print("Queue received stop signal!")
    for p in procs:
        p.join(5)
        p.terminate()
    q.close()
    print("Queue is closed.")
    yield


def train_slice_gen(queue, args):
    for bundle_dict in queue:
        for slice in nibabelio.Bundle(**bundle_dict).slices(args.slice_height, args.slice_height // 2):
            yield dict(
                scan=slice.scan,
                segm=slice.segm,
            )
