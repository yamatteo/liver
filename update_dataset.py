# from __future__ import annotations
#
# from multiprocessing import Pool
# from pathlib import Path
# from types import SimpleNamespace
# from typing import Iterable
#
# import elasticdeform
# import numpy as np
# import torch
# from rich import print
#
# from nibabelio import repeat, slices, split_iter_trainables, load_registration_data, _load_ndarray, onehot, argmax
# from slicing import halfstep_z_slices as hzs
#
# env = SimpleNamespace()
# env.backend = "wandb"
# env.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# env.models_path = Path("/gpfswork/rech/otc/uiu95bi/saved_models")
# env.sources_path = Path("/gpfswork/rech/otc/uiu95bi/sources")
#
# print("Defining generators.")
# train_cases, valid_cases = split_iter_trainables(env.sources_path)
#
#
# def process_case(index, case_path):
#     if f"{index:05}.done" in [ r.name for r in env.dataset_path.glob("*.done") ]:
#         print(f"Index {index} is done, skipping.")
#         return
#     print(f"Index {index}. Processing case {case_path}")
#     _, bottom, top, _ = load_registration_data(case_path)
#     scan = np.stack([
#         _load_ndarray(case_path / f"registered_phase_{phase}.nii.gz")
#         for phase in ["b", "a", "v", "t"]
#     ])
#     scan = scan[..., bottom:top]
#     np.clip(scan, a_min=-300, a_max=400, out=scan)
#     scan = scan.astype(np.float32)
#
#     segm = _load_ndarray(case_path / f"segmentation.nii.gz")
#     assert np.all(segm < 3), "segmentation has indices above 2"
#     segm = segm[..., bottom:top]
#     segm = segm.astype(np.int64)
#
#     print(f"Index {index}. Applying random elastic deformation...")
#     segm = onehot(segm)
#     assert scan.ndim == segm.ndim == 4, \
#         f"Scan and segm are expected to be of shape [C, X, Y, Z], got {scan.shape}, {segm.shape}."
#     scan, segm = elasticdeform.deform_random_grid(
#         [scan, segm],
#         sigma=np.broadcast_to(np.array([6, 6, 2]).reshape([3, 1, 1, 1]), [3, 5, 5, 5]),
#         points=[5, 5, 5],
#         axis=[(1, 2, 3), (1, 2, 3)],
#     )
#     segm = argmax(segm)
#
#     print(f"Index {index}. Saving slices ffrom {case_path}.")
#     for j, (scan_slice, segm_slice) in enumerate(hzs(scan, segm, length=8)):
#         torch.save(
#             dict(
#                 scan=torch.tensor(scan_slice),
#                 segm=torch.tensor(segm_slice),
#             ),
#             env.dataset_path / f"{index:05}{j:03}.pt")
#
#     torch.save(0, env.dataset_path / f"{index:05}.done")
#
# if __name__ == '__main__':
#     env.dataset_path = Path("/gpfswork/rech/otc/uiu95bi/dataset/train")
#     with Pool(10) as p:
#         print("Storing train dataset.")
#         p.starmap(process_case, enumerate(repeat(list(train_cases), stop_after=10)))
#     env.dataset_path = Path("/gpfswork/rech/otc/uiu95bi/dataset/valid")
#     with Pool(10) as p:
#         print("Storing valid dataset.")
#         p.starmap(process_case, enumerate(valid_cases))
#
# # train_slices = slices(repeat(train_cases, stop_after=10), deform=True, length=4, train=True)
# # valid_slices = slices(valid_cases, deform=False, length=4, train=True)
# #
# #
# # def store_generator(generator: Iterable, path: Path):
# #     i = 0
# #     for x in iter(generator):
# #         print(f"Saving {i:08}.pt")
# #         torch.save(x, path / f"{i:08}.pt")
# #         i += 1
# #
# #
# # def load(case_path: Path, train: bool = False, clip: tuple[int, int] = None):
# #     print(f"Loading {case_path}...")
# #     _, bottom, top, _ = load_registration_data(case_path)
# #     scan = np.stack([
# #         _load_ndarray(case_path / f"registered_phase_{phase}.nii.gz")
# #         for phase in ["b", "a", "v", "t"]
# #     ])
# #     scan = scan[..., bottom:top]
# #     if clip:
# #         np.clip(scan, *clip, out=scan)
# #     scan = scan.astype(np.float32)
# #
# #     if train:
# #         segm = _load_ndarray(case_path / f"segmentation.nii.gz")
# #         assert np.all(segm < 3), "segmentation has indices above 2"
# #         segm = segm[..., bottom:top]
# #         segm = segm.astype(np.int64)
# #
# #
# # store_generator(valid_slices, env.dataset_path / "valid")
# # print("Storing train dataset.")
# # store_generator(train_slices, env.dataset_path / "train")
