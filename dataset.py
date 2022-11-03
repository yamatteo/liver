import heapq
import multiprocessing as mp
import queue
from collections import namedtuple
from itertools import cycle
from pathlib import Path
from typing import Iterator

import elasticdeform
import numpy as np
import torch.utils.data
from rich import print

# import idr_torch as idr
import nibabelio
from slicing import slices


debug = Path(".env").exists()
res = 64 if debug else 512

class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, generator: Iterator[dict], *, buffer_size: int = None, staging_size: int = None, post_func=None):
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
            return {"keys": i, **self.post_func(**self.buffer[i])}
        return {"keys": i, **self.buffer[i]}

    def fill(self, size=None):
        size = size or self.buffer_size
        for _ in range(size):
            if len(self.buffer) >= size:
                break
            self.buffer.append(next(self.generator))

    def drop(self, scores: dict[int, float]):
        smallest = heapq.nsmallest(
            len(self.buffer) - self.buffer_size + self.staging_size,
            list(scores.keys()),
            lambda i: scores[i]
        )
        if 0 not in smallest:
            smallest.append(0)
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


def onehot(x, n=None, dtype=np.float32, is_batch=True):
    x = np.asarray(x)
    n = np.max(x) + 1 if n is None else n
    x = np.eye(n, dtype=dtype)[x]
    if is_batch:
        return np.transpose(x, (0, 4, 1, 2, 3))
    else:
        return np.transpose(x, (3, 0, 1, 2))


def deformed(bundle) -> dict:
    # print({k:v.shape if isinstance(v, np.ndarray) else v for k, v in bundle.items()})
    scan = bundle["scan"]
    is_batch = scan.ndim == 5
    segm = onehot(bundle["segm"], is_batch=is_batch)
    axis = (2, 3, 4) if is_batch else (1, 2, 3)
    # print("scan", scan.shape, "segm", segm.shape)
    scan, segm = elasticdeform.deform_random_grid(
        [scan, segm],
        sigma=np.broadcast_to(np.array([4, 4, 1]).reshape([3, 1, 1, 1]), [3, 5, 5, 5]),
        points=[5, 5, 5],
        axis=[axis, axis],
    )
    segm = np.argmax(segm, axis=1) if is_batch else np.argmax(segm, axis=0)
    return dict(bundle, scan=scan, segm=segm)


def put(q, case_path, deform, clip=(-300, 400)):
    # print("Loading", case_path)
    bundle = nibabelio.load(case_path, train=True, clip=clip)
    if deform:
        print("Applying elastic deformation to", bundle["name"])
        bundle = deformed(bundle)
    q.put(bundle)


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
            # p = ctx.Process(target=put, args=(q, next(case_list), True))
            p = ctx.Process(target=put, args=(q, next(case_list), computed >= num_cases))
            procs.append(p)
            p.start()
            computed += 1
            continue
        try:
            stop = yield q.get(timeout=5)
        except queue.Empty:
            pass
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
        for scan, segm in slices(bundle_dict["scan"], bundle_dict["segm"], shape=(res, res, args.slice_height), stride=(res, res, args.slice_height // 2)):
            yield dict(
                scan=scan,
                segm=segm,
            )


def debug_slice_gen(_, height):
    while True:
        yield dict(
            scan=np.random.randn(4, res, res, height),
            segm=np.random.randint(0, 2, [res, res, height]),
        )
